from __future__ import print_function

import sys

import argparse
import random
import itertools
import time
import math
import data
from autoencoder_data import ClassifyData
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

parser = argparse.ArgumentParser(description='Autoencoder Quora Classifier')
parser.add_argument('--datadir', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--autoencoder', type=str, default='anti_collapse_10.pt',
                    help='location of the data corpus')
parser.add_argument('--max_sentences', type=int, default=1000000,
                    help='max num of sentences to train on')

# Network size (embed size, sentence size, nlayers, etc)
parser.add_argument('--din', type=int, default=30,
                    help='length of sentences')
parser.add_argument('--vocabsize', type=int, default=20000,
                    help='how many words to get from glove')

# Training parameters.
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')

# Classifier parameters.
parser.add_argument('--mlp', action='store_true',
                    help='feed length based features into MLP')
parser.add_argument('--mode', type=str, default='distance',
                    help='feed length based features into MLP')
parser.add_argument('--projection_dim', type=int, default=2,
                    help='dimension of subspaces to project vectors to.')
parser.add_argument('--n_projections', type=int, default=100,
                    help='how many projections to do.')

parser.add_argument('--epochs', type=int, default=2,
                    help='epochs to train against.')
parser.add_argument('--batchsize', type=int, default=25, metavar='N',
                    help='batch size')
parser.add_argument('--valid_batches', type=int, default=50,
                    help='number of validation set batches')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='print more debugging information.')
parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_to', type=str,  default='ae_classifier_0.pt',
                    help='path to save the final model')

args = parser.parse_args()

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

def generate_labeled(args, triplets, questions):
    '''Generates Q1, Q2, Y tensors in epochs.'''
    def epoch():
        np.random.shuffle(triplets)
        for batch_idx in xrange(0, len(triplets), args.batchsize):
            if batch_idx + args.batchsize > len(triplets)
                return # Do the remainder next epoch

            batch = triplets[batch_idx:(batch_idx + args.batchsize)]
            qid1 = torch.LongTensor([t[0] for t in batch])
            qid2 = torch.LongTensor([t[1] for t in batch])
            y = torch.ByteTensor([t[2] for t in batch])
            q1 = questions[qid1]
            q2 = questions[qid2]

            # Yield input, duplicate matrix
            if args.cuda:
                q1 = q1.cuda()
                q2 = q2.cuda()
                y = y.cuda()
            yield (q1, q2, y)

    for e in range(args.epochs):
        yield epoch()

def generate_train(args, data):
    return generate_labeled(
            args, data.train_triplets, data.questions_train)

def generate_valid(args, data):
    return generate_labeled(
            args, data.valid_triplets, data.questions_valid)

def noise(args):
    stdev = 1.0
    batch_size = args.batchsize
    cd = lambda x: x if not args.cuda else x.cuda()
    for_size = {}
    def generate_noise(size):
        if size not in for_size:
            print('New noise size:', size)
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
    data = ClassifyData(args)
    autoencoder = torch.load(args.autoencoder)
    # Lock the autoencoder
    for param in autoencoder.parameters():
        param.requires_grad = False

    model = AutoencoderClassifier(
        autoencoder,
        mode=args.mode,
        projection_dim=args.projection_dim,
        n_projections=args.n_projections,
        dropout=args.dropout,
        use_mlp=args.use_mlp)
    print(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr)

    # Input: B x W LongTensor
    # Duplicate_matrix: B x B ByteTensor
    print('Starting.')

    # Decaying average stats.
    recent_loss = 0
    add_to_average = lambda r, v: 0.9 * r + 0.1 * v

    train_loader = generate_train(args, data)
    valid_loader = generate_valid(args, data)

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
            cur_valids = cache(valids, batchsize=args.valid_batches)

            print('Epoch {} start'.format(eid))
            batchcount = 0
            for ind, (q1, q2, y) in enumerate(cur_batches):
                batchcount = ind + 1
                total_batchcount += 1
                start_time = time.time()

                q1 = Variable(q1)
                q2 = Variable(q2)
                y = Variable(y)
                model.zero_grad()
                bsz = input.size(0)

                # RUN THE MODEL FOR THIS BATCH.
                score = model(q1, q2)
                loss = criterion(score, y)

                if args.debug and first_batch:
                    print('INPUT:', q1, q2, y)
                    first_batch = False

                # Assemble the complete loss function.
                loss.backward()
                clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()

                total_cost += loss.data[0]
                recent_loss = add_to_average(recent_loss, loss.data[0])

                #if ind > 100:
                    #return  # for testing only
                if total_batchcount % args.loginterval == 0 \
                        and total_batchcount > 0:
                    elapsed = time.time() - start_time
                    # Each 0.1234/0.7356 loss is reconstruction/kl-div
                    print('Epoch {} | Batch {:5d} | ms/batch {:5.2f} | loss {}'\
                            .format(
                                eid, total_batchcount,
                                elapsed * 1000.0 / args.loginterval,
                                fmt(recent_loss)))

            # Run model on validation set.
            model.eval()
            tvloss = 0.0
            batches = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for ind, (q1, q2, y) in enumerate(valids):
                if ind > args.valid_batches:
                    break
                input = Variable(input)
                idx = random.randint(0, len(input) - 1) # which sentence?
                orig_str = data.to_str(input[idx].data)

                # Run on validation set.
                q1 = Variable(q1)
                q2 = Variable(q2)
                y = Variable(y)

                # RUN THE MODEL FOR THIS BATCH.
                score = model(q1, q2)
                loss = criterion(score, y)
                tvloss += loss.data[0]
                batches += 1

                # Create a confusion matrix.
                pred = (score.data.numpy() > 0.5).astype(np.int32)
                y = y.data.numpy()

                true_positives += sum(y * pred)
                false_positives += sum((1 - y) * pred)
                true_negatives += sum((1 - y) * (1 - pred))
                false_negatives += sum(y * (1 - pred))

            confusion_matrix = np.array([
                [true_positives, false_negatives],
                [false_positives, true_negatives]
                ])
            print('Validation confusion matrix:')
            print(confusion_matrix)

            acc = (true_positives + true_negatives) / np.sum(confusion_matrix)
            print('Valid loss: {:.6f} | Valid acc: {:.6f}'
                    .format(tvloss / batches, acc)

            with open(args.save_to, 'wb') as f:
                torch.save(model, f)
    finally:
        print('Done. Saving cpu model')
        with open('cpu_' + args.save_to, 'wb') as f:
            mcpu = model.cpu()
            torch.save(mcpu, f)


if __name__ == '__main__':
    main()
