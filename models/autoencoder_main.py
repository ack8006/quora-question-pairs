from __future__ import print_function

import sys

import argparse
import itertools
import time
import math
import data
import pickle
import spacy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader

from models import BiLSTM, EmbeddingAutoencoder

nlp = spacy.load('en', parser=False)

# data_path = '../data/train.csv'
# d_in = 30
# d_emb = 50
# cuda = False
# batch_size = 20
# epochs = 20
# d_out = 2
# d_hid = 50
# n_layers = 1
# optimizer = True
# lr = 0.05
# dropout = 0.0
# clip = 0.25
# vocab_size = 20000
# cuda = False
# log_interval = 200


def load_data(args, path, glove, limit=1000000):
    print('Loading Data')
    data = pd.read_csv(path, encoding='utf-8')
    data.columns = ['qid', 'question']

    train_data = data.iloc[:int(len(data)*0.8)]

    print('Cleaning and Tokenizing')
    qid, q = clean_and_tokenize(args, train_data, glove.dictionary, limit)

    return qid, q

def clean_and_tokenize(args, train_data, dictionary, limit):
    def to_indices(words):
        ql = [dictionary.get(str(w).lower(), dictionary['<unk>']) for w in words]
        qv = np.ones(args.din, dtype=int) * dictionary['<pad>'] # all padding
        qv[:len(ql)] = ql[:args.din] # set values
        return qv

    qids = []
    qs = []
    processed = 0
    print('Reading max:', limit)
    for example in train_data.itertuples():
        if processed % 10000 == 0:
            print('processed {0}'.format(processed))
        if processed > limit:
            break
        tokens = nlp(example.question, parse=False)
        qids.append(example.qid)
        qs.append(to_indices(tokens))
        processed += 1
    qst = torch.LongTensor(np.stack(qs, axis=0))
    qidst = torch.LongTensor(qids)
    return qidst, qst

class LoadedGlove:
    def __init__(self, glove):
        self.dictionary = glove[0]
        self.lookup = glove[1]
        self.module = glove[2]

def load_glove(args):
    # Returns dictionary, lookup, embed
    print('loading Glove')
    glove = data.load_embeddings(
            '{1}/glove.6B.{0}d.txt'.format(args.demb, args.glovedata),
            max_words=args.vocabsize)
    return LoadedGlove(glove)

def generate_supplement(args, questions):
    indices = range(len(questions))
    cd = lambda x: x if not args.cuda else x.cuda()

    while True:
        np.random.shuffle(indices)
        for batch in xrange(0, len(indices), args.batchsize): # Seed size
            batch_indices = indices[batch:(batch + args.batchsize)]
            yield cd(questions[torch.LongTensor(batch_indices)])
def cache(x, batchsize=250):
    cache = []
    for item in x:
        if len(cache) == batchsize:
            for c in cache:
                yield c
            cache = []
        cache.append(item)
    for c in cache:
        yield c


def generate(args, qids, questions):
    print(qids[:6])
    print(questions[:6])
    duplicates = pd.read_csv(args.duplicates)
    duplicates.columns = ['qid1', 'qid2']

    # What duplicates are available.
    all_qids = list(qids)
    set_qids = set(qids)
    duplist = ((t.qid1, t.qid2) for t in duplicates.itertuples())
    duplist = filter(lambda (q1, q2): q1 in set_qids and q2 in set_qids, duplist)
    print('{0} possible dupes'.format(len(duplist)))

    # Is (q1, q2) a duplicate.
    dupset = set(duplist)
    # For q1, what are its duplicates.
    dup_lookup = {qid: [] for qid in all_qids}
    idx_lookup = {qid: i for i, qid in enumerate(all_qids)}
    for qid1, qid2 in duplist:
            dup_lookup[qid1].append(qid2)

    print('Generating seeds...')
    seeds = []
    seeds_seen = {}
    for qid, dups in dup_lookup.items():
        if len(dups) == 0:
            continue
        if qid in seeds_seen:
            continue
        seeds_seen[qid] = True
        for d in dups:
            seeds_seen[d] = True
        seeds.append((qid, dups))
    print('Seeds:', len(seeds))

    def batch():
        np.random.shuffle(seeds)
        np.random.shuffle(all_qids)
        seed_size = args.seed_size
        qids_iter = itertools.cycle(all_qids)
        for dup_batch in xrange(0, len(seeds), seed_size): # Seed size
            if args.batches > 0 and dup_batch / seed_size > args.batches:
                return
            # Get a selection of duplicates as the seed.
            seed_ids = []
            seed_id_pairs = []
            already = {}
            for qid1, selection in seeds[dup_batch:(dup_batch + seed_size)]:
                qid2 = np.random.choice(selection)
                seed_ids.append(qid1)
                seed_id_pairs.append(qid2)
                already[qid1] = True
                already[qid2] = True
            # From the seeds, find other ids that are duplicates.
            extra = []
            for qid in seed_ids:
                dups = dup_lookup[qid]
                np.random.shuffle(dups)
                for d in dups:
                    if d not in already:
                        already[d] = True
                        extra.append(d)

            np.random.shuffle(extra)
            batch = seed_ids + seed_id_pairs + extra[:(
                args.batchsize - len(seed_ids) * 2)]
            while len(batch) < args.batchsize:
                qid = next(qids_iter)
                if qid not in already:
                    already[qid] = True
                    batch.append(qid)
            np.random.shuffle(batch)

            mtx = np.zeros((len(batch), len(batch)), dtype=np.int32)
            for i, q1 in enumerate(batch):
                for j, q2 in enumerate(batch):
                    mtx[i,j] = (q1, q2) in dupset
            #print(mtx.sum(axis=1))

            # Yield input, duplicate matrix
            indices = [idx_lookup[qid] for qid in batch]
            mtx = torch.from_numpy(mtx)
            batch_qs = questions[torch.LongTensor(indices)]
            if args.cuda:
                batch_qs = batch_qs.cuda()
                mtx = mtx.cuda()
            yield (batch_qs, mtx)

    print('Analysis done. Ready to generate batches.')
    for e in range(args.epochs):
        yield batch()


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

    # Hinge loss between lest likely duplicate and most likely non-duplicate.
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
    parser.add_argument('--data', type=str, default='../data/all_questions.csv',
                        help='location of the data corpus')
    parser.add_argument('--supplement', type=str, default=None,
                        help='unlabeled supplemental data')
    parser.add_argument('--duplicates', type=str, default='../data/unique_duplicates.csv',
                        help='location of the data corpus')
    parser.add_argument('--glovedata', type=str, default='../data/',
                        help='location of the pretrained glove embeddings')
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

    assert args.demb in (50, 100, 200, 300)
    glove = load_glove(args)
    qid, questions = load_data(args, args.data, glove, args.max_sentences)
    train_loader = generate(args, qid, questions)

    supplement_loader = None
    if args.supplement is not None:
        sid, supplement = load_data(args, args.supplement, glove, args.max_supplement)
        supplement_loader = cache(generate_supplement(args, supplement))

    embedding = glove.module
    bilstm_encoder = BiLSTM(args.demb, args.dhid, args.nlayers, args.dropout)
    bilstm_decoder = BiLSTM(args.demb, args.dhid, args.nlayers, args.dropout)
    model = EmbeddingAutoencoder(embedding, bilstm_encoder, bilstm_decoder,
        embed_size=args.squash_size, cuda=args.cuda, dropout=args.dropout)

    reconstruction_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
            [param for param in model.parameters()
                if param.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    def logistic(slope, shift, x):
        gate0 = slope * (x - shift)
        return 1.0 / (1.0 + math.exp(-gate0))

    # Input: B x W LongTensor
    # Duplicate_matrix: B x B ByteTensor
    print('Starting.')
    recent_rloss = 0
    recent_dloss = 0
    recent_sloss = 0
    recent_sep = 0
    eye = torch.eye(200)
    if args.cuda:
        eye = eye.cuda()
    try:
        for (eid, batches) in enumerate(train_loader):
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
                recent_rloss = 0.95 * recent_rloss + 0.05 * rloss.data[0]
                recent_dloss = 0.95 * recent_dloss + 0.05 * dloss.data[0]
                recent_sloss = 0.95 * recent_sloss + 0.05 * sloss.data[0]
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
            model.cpu()
            torch.save(model, f)


if __name__ == '__main__':
    main()
