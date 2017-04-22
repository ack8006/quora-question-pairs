from __future__ import print_function

import sys

import argparse
import itertools
import time
import math
import data
from autoencoder_data import Data
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader

from models import BiLSTM, EmbeddingAutoencoder

def logistic(slope, shift, x):
    gate0 = slope * (x - shift)
    return 1.0 / (1.0 + math.exp(-gate0))

def scheduler(config):
    '''Creates a "sigmoid scheduler", a sequence of values that follow a
    scaled and shifted sigmoid function.'''
    slope, shift = config
    return (logistic(slope, shift, x) for x in itertools.count())

def generate_supplement(args, questions):
    indices = range(len(questions))
    cd = lambda x: x if not args.cuda else x.cuda()

    while True:
        np.random.shuffle(indices)
        for batch in xrange(0, len(indices), args.batchsize): # Seed size
            if len(indices) - batch != args.batchsize:
                # Skip to keep batch size constant; indices will get shuffled
                # anyway.
                continue
            batch_indices = indices[batch:(batch + args.batchsize)]
            yield cd(questions[torch.LongTensor(batch_indices)])

def cache(x, batchsize=250):
    cache = [] # Batch of batches.
    for item in x:
        if len(cache) == batchsize:
            for c in cache:
                yield c
            cache = []
        cache.append(item)
    for c in cache:
        yield c


def generate(args, data, clusters_list):
    def batch():
        for batch_qs, mtx in clusters.iterate_epoch(clusters_list, args):
            if args.batches > 0 and dup_batch / seed_size > args.batches:
                return

            # Yield input, duplicate matrix
            if args.cuda:
                batch_qs = batch_qs.cuda()
                mtx = mtx.cuda()
            yield (batch_qs, mtx)

    print('Analysis done. Ready to generate batches.')
    for e in range(args.epochs):
        yield batch()


eye = torch.eye(200)
if args.cuda:
    eye = eye.cuda()
def distance_loss(log_prob, duplicate_matrix, eye):
    '''Args:
        log_prob: B*B sized array of log probabilities.
        duplicate_matrix: B*B array of is_duplicates.'''
    B = duplicate_matrix.size(0)
    duplicate_matrix = duplicate_matrix.float()
    my_eye = Variable(eye[:B, :B])
    non_duplicate_matrix = (1 - duplicate_matrix) - my_eye

    # Calculate dist from logprob
    prob = log_prob.exp() + 1e-8
    prob = prob - (my_eye * prob)
    prob = prob / prob.sum(dim=1).repeat(1, B)

    #print(duplicate_matrix)
    #print(non_duplicate_matrix)

    pd = prob * duplicate_matrix
    # Set to 1 all cells that aren't probability of duplicate so we can
    # take the min value.
    probability_dup = pd + (1 - duplicate_matrix)
    probability_non = prob * non_duplicate_matrix

    has_dup = duplicate_matrix.sum(dim=1).gt(0)
    min_dup = probability_dup.min(dim=1)[0][has_dup]
    max_dup = pd.max(dim=1)[0][has_dup]
    max_non = probability_non.max(dim=1)[0][has_dup]

    # Gap loss between lest likely duplicate and most likely non-duplicate.
    return (1 + max_non - min_dup).mean(), (max_dup - max_non).mean()


def noise(args):
    stdev = args.noise_stdev
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


def main():
    parser = argparse.ArgumentParser(description='PyTorch Quora RNN/LSTM Language Model')
    parser.add_argument('--datadir', type=str, default='../data',
                        help='location of the data corpus')
    parser.add_argument('--supplement', type=str, default=None,
                        help='unlabeled supplemental data')
    parser.add_argument('--max_sentences', type=int, default=1000000,
                        help='max num of sentences to train on')
    parser.add_argument('--max_supplement', type=int, default=1000000,
                        help='max num of supplemental sentences to train on')
    parser.add_argument('--din', type=int, default=30,
                        help='length of LSTM')
    parser.add_argument('--demb', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--dhid', type=int, default=100,
                        help='humber of hidden units per layer')
    parser.add_argument('--dout', type=int, default=2,
                        help='number of output classes')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--noise_stdev', type=float, default=0.05,
                        help='noise distribution standard deviation')
    parser.add_argument('--sloss_factor', type=float, default=0.1,
                        help='supplemental loss scaling')
    parser.add_argument('--dloss_factor', type=float, default=1.0,
                        help='distance loss scaling')
    parser.add_argument('--dloss_shift', type=int, default=4,
                        help='when should dloss gating reach 0.5')
    parser.add_argument('--sloss_shift', type=int, default=4,
                        help='when should sloss gating reach 0.5')
    parser.add_argument('--sloss_slope', type=float, default=1.0,
                        help='when should sloss gating reach 0.5')
    parser.add_argument('--dloss_slope', type=float, default=1.0,
                        help='how quickly dloss goes from 0...1')
    parser.add_argument('--embinit', type=str, default='random',
                        help='embedding weight initialization type')
    parser.add_argument('--squash_size', type=int, default=40,
                        help='sentence embedding squash size')
    parser.add_argument('--seed_size', type=int, default=10,
                        help='how many seed points from which to sample duplicates')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--more_dropout', action='store_true',
                        help='activate dropout on the embedding layers')
    parser.add_argument('--epochs', type=int, default=2,
                        help='epochs to train against.')
    parser.add_argument('--batchsize', type=int, default=25, metavar='N',
                        help='batch size')
    parser.add_argument('--batches', type=int, default=300,
                        help='max batches in an epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='optimizer weight decay')
    parser.add_argument('--vocabsize', type=int, default=20000,
                        help='how many words to get from glove')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save_to', type=str,  default='autoencoder_3.pt',
                        help='path to save the final model')
    args = parser.parse_args()

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
    recent_dloss = 0
    recent_sloss = 0
    recent_sep = 0

    try:
        for (eid, batches) in enumerate(train_loader):
            model.train()
            total_cost = 0
            first_batch = True
            print('Precomputing batches')
            cur_batches = cache(batches) # Precompute the batch.
            dloss_gate = logistic(args.dloss_slope, args.dloss_shift, eid)
            dloss_factor = args.dloss_factor * dloss_gate
            sloss_factor = args.sloss_factor * logistic(
                args.dloss_slope, args.sloss_shift, eid)
            print('Epoch {} start, dloss_factor = {:.6f}, sloss_factor={:.6f}'.\
                    format(eid, dloss_factor, sloss_factor))
            batchcount = 0
            for ind, (input, duplicate_matrix) in enumerate(cur_batches):
                batchcount = ind + 1
                start_time = time.time()
                input = Variable(input)
                model.zero_grad()
                bsz = input.size(0)

                # RUN THE MODEL FOR THIS BATCH.
                if args.cuda and not input.is_cuda:
                    input = input.cuda()
                supp = None
                if supplement_loader is not None:
                    supp = Variable(next(supplement_loader))
                auto, log_prob, supp_auto = \
                    model(input, noise(args), supp)
                rloss = bsz * reconstruction_loss(
                        auto.view(-1, args.vocabsize), input.view(-1))
                dloss, separation = distance_loss(
                        log_prob, Variable(duplicate_matrix), eye)
                loss = rloss + dloss_factor * dloss

                sloss = 0.0
                if supp is not None:
                    sloss = bsz * reconstruction_loss(
                            supp_auto.view(-1, args.vocabsize), supp.view(-1))
                    loss = loss + sloss_factor * sloss

                if first_batch:
                    #print(input)
                    #print(duplicate_matrix)
                    #print(prob)
                    first_batch = False

                loss.backward()
                clip_grad_norm(model.parameters(), args.clip)

                if optimizer:
                    optimizer.step()
                else:
                    for p in model.parameters():
                        p.data.add_(-args.lr, p.grad.data)

                total_cost += loss.data[0]
                recent_rloss = 0.9 * recent_rloss + 0.1 * rloss.data[0]
                recent_dloss = 0.9 * recent_dloss + 0.1 * dloss.data[0]
                recent_sloss = 0.9 * recent_sloss + 0.1 * sloss.data[0]
                recent_sep = 0.9 * recent_sep + 0.1 * separation.data[0]

                #if ind > 100:
                    #return  # for testing only
                if ind % args.loginterval == 0 and ind > 0:
                    cur_loss = total_cost / (ind * args.batchsize)
                    elapsed = time.time() - start_time
                    print('Epoch {} | {:5d}/{} Batches | ms/batch {:5.2f} | '
                            'losses r{:.6f} s{:.6f} d{:.6f} (sep {:.6f})'.format(
                                eid, ind, args.batches,
                                elapsed * 1000.0 / args.loginterval,
                                recent_rloss, recent_sloss, recent_dloss, recent_sep))

            print('Average loss: {:.6f}'.format(total_cost / batchcount))
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
