from __future__ import print_function

import sys

import argparse
import time
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


def load_data(args, glove):
    print('Loading Data')
    data = pd.read_csv(args.data, encoding='utf-8')
    data.columns = ['qid', 'question']
    duplicates = pd.read_csv(args.duplicates)

    train_data = data.iloc[:int(len(data)*0.8)]

    print('Cleaning and Tokenizing')
    qid, q = clean_and_tokenize(args, train_data, glove.dictionary)
    if args.cuda:
        qid = qid.cuda()
        q = q.cuda()

    return qid, q

def clean_and_tokenize(args, train_data, dictionary):
    def to_indices(words):
        ql = [dictionary.get(str(w).lower(), dictionary['<unk>']) for w in words]
        qv = np.ones(args.din, dtype=int) * dictionary['<pad>'] # all padding
        qv[:len(ql)] = ql[:args.din] # set values
        return qv

    qids = []
    qs = []
    processed = 0
    print('Reading max:', args.max_sentences)
    for example in train_data.itertuples():
        if processed % 10000 == 0:
            print('processed {0}'.format(processed))
        if processed > args.max_sentences:
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
    dup_lookup = {qid: [] for qid in list(qids)}
    idx_lookup = {qid: i for i, qid in enumerate(list(qids))}
    for qid1, qid2 in duplist:
            dup_lookup[qid1].append(qid2)

    qids_list = list(qids)
    def batch():
        np.random.shuffle(duplist)
        seed_size = 5
        for dup_batch in xrange(0, len(duplist), seed_size): # Seed size
            if args.batches > 0 and dup_batch / seed_size > args.batches:
                return
            # Get a selection of duplicates as the seed.
            seed_ids = [qid1 for qid1, qid2 in
                    duplist[dup_batch:(dup_batch + seed_size)]]
            # From the seeds, find other ids that are duplicates.
            matching_duplicates = (dup_lookup[qid][:10] for qid in seed_ids)
            # Flatten those IDs to a list.
            dups_for_id = list(x for t in matching_duplicates for x in t)

            batch = (seed_ids + dups_for_id)
            np.random.shuffle(batch)
            batch = batch[:args.batchsize]
            while len(batch) < args.batchsize:
                batch.append(np.random.choice(qids_list))

            mtx = np.zeros((len(batch), len(batch)), dtype=np.int32)
            for i, q1 in enumerate(batch):
                for j, q2 in enumerate(batch):
                    mtx[i,j] = (q1, q2) in dupset

            # Yield input, duplicate matrix
            indices = [idx_lookup[qid] for qid in batch]
            yield questions[torch.LongTensor(indices).cuda()], torch.from_numpy(mtx).cuda()

    for e in range(args.epochs):
        yield batch()


def distance_loss(dist, duplicate_matrix):
    '''Args:
        dist: B*B sized array of differences.
        duplicate_matrix: B*B array of is_duplicates.'''
    B = duplicate_matrix.size(0)
    duplicate_matrix = duplicate_matrix.float()

    probability_dup = dist * duplicate_matrix + (1 - duplicate_matrix)
    probability_non = dist * (1 - duplicate_matrix)

    min_dup = probability_dup.min(dim=1)[0]
    max_non = probability_non.max(dim=1)[0]

    #probability_dup = (dist * duplicate_matrix).mean(dim=1)
    #probability_non = (dist * (1 - duplicate_matrix)).mean(dim=1)

    #min_dup = probability_dup
    #max_non = probability_non

    # Hinge loss between lest likely duplicate and most likely non-duplicate.
    return (1 + max_non - min_dup).mean()


def main():
    parser = argparse.ArgumentParser(description='PyTorch Quora RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/all_questions.csv',
                        help='location of the data corpus')
    parser.add_argument('--duplicates', type=str, default='../data/unique_duplicates.csv',
                        help='location of the data corpus')
    parser.add_argument('--glovedata', type=str, default='../data/',
                        help='location of the pretrained glove embeddings')
    parser.add_argument('--max_sentences', type=int, default=1000000,
                        help='max num of sentences to train on')
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
    parser.add_argument('--embinit', type=str, default='random',
                        help='embedding weight initialization type')
    parser.add_argument('--decinit', type=str, default='random',
                        help='decoder weight initialization type')
    parser.add_argument('--hidinit', type=str, default='random',
                        help='recurrent hidden weight initialization type')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='epochs to train against.')
    parser.add_argument('--batchsize', type=int, default=25, metavar='N',
                        help='batch size')
    parser.add_argument('--batches', type=int, default=300,
                        help='max batches in an epoch')
    parser.add_argument('--vocabsize', type=int, default=20000,
                        help='how many words to get from glove')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='',
                        help='path to save the final model')
    args = parser.parse_args()

    assert args.demb in (50, 100, 200, 300)
    glove = load_glove(args)
    qid, questions = load_data(args, glove)
    train_loader = generate(args, qid, questions)

    embedding = glove.module
    bilstm_encoder = BiLSTM(args.demb, args.dhid, args.nlayers, args.dropout)
    bilstm_decoder = BiLSTM(args.dhid, args.dhid, args.nlayers, args.dropout)
    model = EmbeddingAutoencoder(embedding, bilstm_encoder, bilstm_decoder)

    if args.cuda:
        model.cuda()

    reconstruction_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
            [param for param in model.parameters()
                if param.requires_grad], lr=args.lr, weight_decay=0.000001)

    # model_config = '\t'.join([str(x) for x in (torch.__version__, args.clip, args.nlayers, args.din, args.demb, args.dhid, 
    #                     args.embinit, args.freezeemb, args.decinit, args.hidinit, args.dropout, args.optimizer, args.lr, args.vocabsize)])
    # print('Pytorch | Clip | #Layers | InSize | EmbDim | HiddenDim | EncoderInit | FreezeEmb | DecoderInit | WeightInit | Dropout | Optimizer| LR | VocabSize')
    # print(model_config)

    # Input: B x W LongTensor
    # Duplicate_matrix: B x B ByteTensor
    print('Starting.')
    for (eid, batches) in enumerate(train_loader):
        total_cost = 0
        first_batch = True
        for ind, (input, duplicate_matrix) in enumerate(batches):
            start_time = time.time()
            input = Variable(input)
            model.zero_grad()
            bsz = input.size(0)

            # RUN THE MODEL FOR THIS BATCH.
            if args.cuda and not input.is_cuda:
                input = input.cuda()
            auto, prob = model(input)
            rloss = bsz * reconstruction_loss(
                    auto.view(-1, args.vocabsize), input.view(-1))
            dloss = distance_loss(prob, Variable(duplicate_matrix))
            loss = rloss + dloss

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

            if ind % args.loginterval == 0 and ind > 0:
                cur_loss = total_cost / (ind * args.batchsize)
                elapsed = time.time() - start_time
                print('Epoch {} | {:5d}/{} Batches | ms/batch {:5.2f} | '
                        'loss {:.6f} {:.6f}'.format(
                            eid, ind, args.batches,
                            elapsed * 1000.0 / args.loginterval,
                            rloss.data[0], dloss.data[0]))

        print('-' * 89)
        with open('autoencoder.pt', 'wb') as f:
            torch.save(model, f)
    with open('autoencoder_cpu.pt', 'wb') as f:
        model.cpu()
        torch.save(model, f)


if __name__ == '__main__':
    main()
