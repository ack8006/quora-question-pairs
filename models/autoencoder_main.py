from __future__ import print_function

import sys

import argparse
import random
import itertools
import time
import math
import data
from autoencoder_data import Data
import pickle
import clusters

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader

from models import BiLSTM, EmbeddingAutoencoder

parser = argparse.ArgumentParser(description='PyTorch Quora RNN/LSTM Language Model')
parser.add_argument('--datadir', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--supplement', type=str, default=None,
                    help='unlabeled supplemental data')
parser.add_argument('--max_sentences', type=int, default=1000000,
                    help='max num of sentences to train on')
parser.add_argument('--max_supplement', type=int, default=1000000,
                    help='max num of supplemental sentences to train on')

# Network size (embed size, sentence size, nlayers, etc)
parser.add_argument('--din', type=int, default=30,
                    help='length of LSTM')
parser.add_argument('--demb', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--dhid', type=int, default=100,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--squash_size', type=int, default=40,
                    help='sentence embedding squash size')

# D-loss (distance loss) tuning parameters
parser.add_argument('--dloss_factor', type=float, default=1.0,
                    help='distance loss scaling')
parser.add_argument('--dloss_shift', type=int, default=4,
                    help='when should dloss gating reach 0.5')
parser.add_argument('--dloss_slope', type=float, default=1.0,
                    help='how quickly dloss goes from 0...1')

# S-loss (supplemental loss) tuning parameters
parser.add_argument('--sloss_factor', type=float, default=0.1,
                    help='supplemental loss scaling')
parser.add_argument('--sloss_shift', type=int, default=4,
                    help='when should sloss gating reach 0.5')
parser.add_argument('--sloss_slope', type=float, default=1.0,
                    help='when should sloss gating reach 0.5')

# K-loss (KL-divergence loss, VAE) tuning parameters
parser.add_argument('--kloss_factor', type=float, default=1.0,
                    help='supplemental loss scaling')
parser.add_argument('--kloss_shift', type=int, default=10,
                    help='when should kloss gating reach 0.5')
parser.add_argument('--kloss_slope', type=float, default=1.0,
                    help='when should kloss gating reach 0.5')

# Training parameters.
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')

# Labeled training sample generation.
parser.add_argument('--seed_size', type=int, default=10,
                    help='batch generation: how many seed clusters per batch')
parser.add_argument('--take_clusters', type=int, default=10,
                    help='batch generation: how many seed points per seed clusters')

parser.add_argument('--more_dropout', action='store_true',
                    help='activate dropout on the embedding layers')
parser.add_argument('--epochs', type=int, default=2,
                    help='epochs to train against.')
parser.add_argument('--batchsize', type=int, default=25, metavar='N',
                    help='batch size')
parser.add_argument('--valid_batches', type=int, default=50,
                    help='number of validation set batches')
parser.add_argument('--batches', type=int, default=300,
                    help='max batches in an epoch')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='optimizer weight decay')
parser.add_argument('--vocabsize', type=int, default=20000,
                    help='how many words to get from glove')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='print more debugging information.')
parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_to', type=str,  default='autoencoder_3.pt',
                    help='path to save the final model')
args = parser.parse_args()

def kl_div_with_std_norm(mean, logvar):
    return torch.squeeze(torch.sum(logvar.exp() + mean * mean - logvar - 1, 1) / 2)

def logistic(slope, shift, x):
    gate0 = slope * (x - shift)
    return 1.0 / (1.0 + math.exp(-gate0))

def scheduler(slope, shift, factor):
    '''Creates a "sigmoid scheduler", a sequence of values that follow a
    scaled and shifted sigmoid function.'''
    return (factor * logistic(slope, shift, x) for x in itertools.count())

def cache(x, batchsize=250):
    '''Given an iterator, precompute some of its entries.'''
    cache = [] # Batch of batches.
    for item in x:
        if len(cache) == batchsize:
            for c in cache:
                yield c
            cache = []
        cache.append(item)
    for c in cache:
        yield c

def generate_labeled(args, qid, clusters_list, questions):
    '''Generates training batches.'''
    rlookup = {qid: i for i, qid in enumerate(qid)}
    def batch():
        for batch_idx, (batch_qids, mtx) in enumerate(
                clusters.iterate_epoch(clusters_list, args)):
            if args.batches > 0 and batch_idx > args.batches:
                return

            batch_qs = questions[torch.LongTensor([
                rlookup[qid] for qid in batch_qids])]

            # Yield input, duplicate matrix
            if args.cuda:
                batch_qs = batch_qs.cuda()
                mtx = mtx.cuda()
            yield (batch_qs, mtx)

    for e in range(args.epochs):
        yield batch()

def generate_train(args, data):
    return generate_labeled(
            args, data.qid_train, data.train_clusters, data.questions_train)

def generate_valid(args, data):
    return generate_labeled(
            args, data.qid_valid, data.valid_clusters, data.questions_valid)

def generate_supplement(args, data):
    questions = data.questions_supplement
    indices = range(len(questions))
    cd = lambda x: x if not args.cuda else x.cuda()

    while True:
        print(len(indices))
        np.random.shuffle(indices)
        for batch in xrange(0, len(indices), args.batchsize): # Seed size
            if len(indices) - batch != args.batchsize:
                # Skip to keep batch size constant; indices will get shuffled
                # anyway.
                continue
            batch_indices = indices[batch:(batch + args.batchsize)]
            yield cd(questions[torch.LongTensor(batch_indices)])


eye = torch.eye(200)
if args.cuda:
    eye = eye.cuda()

def normalize(log_prob, duplicate_matrix):
    B = duplicate_matrix.size(0)
    duplicate_matrix = duplicate_matrix.float()
    my_eye = Variable(eye[:B, :B])
    non_duplicate_matrix = (1 - duplicate_matrix) - my_eye

    # Calculate dist from logprob
    sumexp = log_prob.exp() + 1e-8
    sumexp = sumexp - (my_eye * sumexp)
    logsumexp = sumexp.sum(dim=1).log()
    log_prob = log_prob - logsumexp.repeat(1, B)
    return log_prob, duplicate_matrix, non_duplicate_matrix

def distance_loss(log_prob, duplicate_matrix):
    '''Args:
        log_prob: B*B sized array of log probabilities.
        duplicate_matrix: B*B array of is_duplicates.'''
    log_prob, duplicate_matrix, non_duplicate_matrix = \
        normalize(log_prob, duplicate_matrix)

    # When mask matrix is multiplied before exponentiating, the '0' becomes '1',
    # so finding the minimum is easy. Vice versa for finding max.
    has_dup = duplicate_matrix.sum(dim=1).gt(0)
    min_dup = (log_prob * duplicate_matrix).min(dim=1)[0].exp()[has_dup]
    max_non = (log_prob.exp() * non_duplicate_matrix).max(dim=1)[0][has_dup]
    max_dup = (log_prob.exp() * duplicate_matrix).max(dim=1)[0][has_dup]
    #print(max_dup - min_dup)
    #print(torch.stack([max_dup, min_dup, max_non], 1))

    # Gap loss between lest likely duplicate and most likely non-duplicate.
    gap = 1 + max_non - min_dup
    separation = max_dup - max_non
    #print(torch.stack([gap, separation], 1))
    return gap.mean(), separation.mean()

def measure(log_prob, duplicate_matrix, dups, nondups):
    '''Return the un-normalized log probabilities'''
    B = log_prob.size(0)
    for i in range(B):
        for j in range(i+1,B):
            if duplicate_matrix[i, j]:
                dups.append(log_prob[i, j].data[0])
            else:
                nondups.append(log_prob[i, j].data[0])


def noise(args):
    stdev = 1.0
    batch_size = args.batchsize
    cd = lambda x: x if not args.cuda else x.cuda()
    for_size = {}
    def generate_noise(size):
        if size not in for_size:
            def gen():
                while True:
                    batch = [Variable(cd(torch.randn(size)) * stdev)
                        for i in xrange(batch_size)]
                    for b in batch:
                        yield b
            for_size[size] = gen()
        return next(for_size[size])
    # GN: function that takes a size and returns a noise vector with the given
    # stdev, from a batch.
    return generate_noise


def cut(string):
    if len(string) > 100:
        return string[:97] + '...'
    return string


def main():
    data = Data(args)
    embedding = data.glove.module
    bilstm_encoder = BiLSTM(args.demb, args.dhid, args.nlayers, args.dropout)
    bilstm_decoder = BiLSTM(args.demb, args.dhid, args.nlayers, args.dropout)
    emb_dropout = 0.0
    if args.more_dropout:
        emb_dropout = args.dropout
    model = EmbeddingAutoencoder(embedding, bilstm_encoder, bilstm_decoder,
        embed_size=args.squash_size, cuda=args.cuda, dropout=emb_dropout)
    print(model)

    reconstruction_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
            [param for param in model.parameters()
                if param.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    # Input: B x W LongTensor
    # Duplicate_matrix: B x B ByteTensor
    print('Starting.')

    # Decaying average stats.
    recent_rloss = 0
    recent_kl = 0
    recent_dloss = 0
    recent_sloss = 0
    recent_kl_s = 0
    recent_sep = 0
    add_to_average = lambda r, v: 0.9 * r + 0.1 * v

    train_loader = generate_train(args, data)
    valid_loader = generate_valid(args, data)
    noiser = noise(args)
    supplement_loader = None
    dloss_schedule = scheduler(args.dloss_slope, args.dloss_shift, args.dloss_factor)
    sloss_schedule = scheduler(args.sloss_slope, args.sloss_shift, args.sloss_factor)
    kloss_schedule = scheduler(args.kloss_slope, args.kloss_shift, args.kloss_factor)
    if data.questions_supplement is not None:
        supplement_loader = generate_supplement(args, data)

    try:
        total_batchcount = 0
        # The math below makes the average accurate at low batch numbers.
        fmt = lambda f: '{:.6f}'.format(f / (1 - 0.9**(total_batchcount)))[:8]
        for (eid, (batches, valids)) in enumerate(itertools.izip(
                train_loader, valid_loader)):
            model.train()
            total_cost = 0
            first_batch = True
            print('Precomputing batches')
            cur_batches = cache(batches) # Precompute the batch.
            dloss_factor = next(dloss_schedule)
            sloss_factor = next(sloss_schedule)
            kloss_factor = next(kloss_schedule)
            print('Epoch {} start, dloss*={:.6f}, sloss*={:.6f}, kloss*={:.6f}'.\
                    format(eid, dloss_factor, sloss_factor, kloss_factor))
            batchcount = 0
            for ind, (input, duplicate_matrix) in enumerate(cur_batches):
                batchcount = ind + 1
                total_batchcount += 1
                start_time = time.time()
                input = Variable(input)
                model.zero_grad()
                bsz = input.size(0)

                # RUN THE MODEL FOR THIS BATCH.
                if args.cuda and not input.is_cuda:
                    input = input.cuda()
                auto, mean, logvar, log_prob = model(input, noiser)
                rloss = bsz * reconstruction_loss(
                        auto.view(-1, args.vocabsize), input.view(-1))
                dloss, separation = distance_loss(
                        log_prob, Variable(duplicate_matrix))
                kl = kl_div_with_std_norm(mean, logvar).mean()

                sloss = 0.0
                kl_s = 0.0
                if supplement_loader is not None:
                    # Supplemental loss.
                    supp = Variable(next(supplement_loader))
                    auto_s, mean_s, logvar_s, _ = model(supp, noiser, False)
                    sloss = bsz * reconstruction_loss(
                            auto_s.view(-1, args.vocabsize), supp.view(-1))
                    kl_s = kl_div_with_std_norm(mean_s, logvar_s).mean()

                if args.debug and first_batch:
                    print('INPUT:', input)
                    print('DUPLICATES:', duplicate_matrix)
                    print('LOGPROB:', log_prob)
                    first_batch = False

                # Assemble the complete loss function.
                loss = rloss + kloss_factor * kl
                if supplement_loader is not None:
                    loss += sloss_factor * (sloss + kloss_factor * kl_s)
                loss += dloss_factor * dloss

                loss.backward()
                clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()

                total_cost += loss.data[0]
                recent_rloss = add_to_average(recent_rloss, rloss.data[0])
                recent_kl = add_to_average(recent_kl, kl.data[0])
                recent_sloss = add_to_average(recent_sloss, sloss.data[0])
                recent_kl_s = add_to_average(recent_kl_s, kl_s.data[0])
                recent_dloss = add_to_average(recent_dloss, dloss.data[0])
                recent_sep = add_to_average(recent_sep, separation.data[0])

                #if ind > 100:
                    #return  # for testing only
                if total_batchcount % args.loginterval == 0 \
                        and total_batchcount > 0:
                    elapsed = time.time() - start_time
                    # Each 0.1234/0.7356 loss is reconstruction/kl-div
                    print('Epoch {} | Batch {:5d} | ms/batch {:5.2f} | '
                            'losses r{}/{} s{}/{} d{} (sep {})'.format(
                                eid, total_batchcount,
                                elapsed * 1000.0 / args.loginterval,
                                fmt(recent_rloss), fmt(recent_kl),
                                fmt(recent_sloss), fmt(recent_kl_s),
                                fmt(recent_dloss), fmt(recent_sep)))

            # Run model on validation set.
            model.eval()
            tvdloss = []
            tvseparation = []
            dups = []
            nondups = []
            sentences = [] # Original sentences and their replicas
            for ind, (input, duplicate_matrix) in enumerate(valids):
                if ind > args.valid_batches:
                    break
                input = Variable(input)
                idx = random.randint(0, len(input) - 1) # which sentence?
                orig_str = data.to_str(input[idx].data)

                # Run on validation set.
                if args.cuda and not input.is_cuda:
                    input = input.cuda()
                auto, mean_s, logvar_v, log_prob = model(input, noiser)
                reconstruction = data.sample_str(auto[idx].data)
                vdloss, vseparation =\
                        distance_loss(log_prob, Variable(duplicate_matrix))
                measure(log_prob, duplicate_matrix, dups, nondups)
                tvdloss.append(vdloss.data[0])
                tvseparation.append(vseparation.data[0])
                sentences.append((orig_str, reconstruction))
            dups = np.exp(np.array(dups))
            nondups = np.exp(np.array(nondups))
            threshold = (min(dups) + max(nondups)) / 2

            true_positives = sum(dups > threshold)
            false_positives = sum(nondups > threshold)
            true_negatives = sum(nondups <= threshold)
            false_negatives = sum(dups <= threshold)
            confusion_matrix = np.array([
                [true_positives, false_negatives],
                [false_positives, true_negatives]
                ])
            print('Validation confusion matrix:')
            print(confusion_matrix)

            print('Average loss: {:.6f} | Valid dloss: {:.6f} | Valid sep: {:.6f}'
                    .format(total_cost / batchcount, np.mean(tvdloss), np.mean(tvseparation)))
            np.random.shuffle(sentences)
            # Print a few sentences and their reconstructions
            print('Some reconstructions:')
            for orig, reconst in sentences[:3]:
                print('  ' + cut(orig))
                print('    => ' + cut(reconst))
            print('-' * 110)
            with open(args.save_to, 'wb') as f:
                torch.save(model, f)
    finally:
        print('Done. Saving cpu model')
        with open('cpu_' + args.save_to, 'wb') as f:
            mcpu = model.cpu()
            torch.save(mcpu, f)


if __name__ == '__main__':
    main()
