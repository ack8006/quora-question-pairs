import sys
sys.path.append('../models/text/')

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from torchtext import data

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from models import LSTMModel



def load_data(args, glove):
    print('Loading Data')
    data = pd.read_csv(args.data, encoding='utf-8')
    data.columns = ['qid1', 'qid2']

    train_data = data.iloc[:int(len(data)*0.8)]
    valid_data = data.iloc[int(len(data)*0.8):]

    print('Cleaning and Tokenizing')
    qid, q = clean_and_tokenize(args, train_data, glove.dictionary)

    return qid, q

def clean_and_tokenize(args, train_data, dictionary):
    def to_indices(words):
        ql = [dictionary.get(str(w), dictionary['<unk>']) for w in words]
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


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/train.csv',
                        help='location of the data corpus')
    parser.add_argument('--glovedata', type=str, default='../data/glove.6B',
                        help='location of the pretrained glove embeddings')
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--embinit', type=str, default='random',
                        help='embedding weight initialization type')
    parser.add_argument('--decinit', type=str, default='random',
                        help='decoder weight initialization type')
    parser.add_argument('--hidinit', type=str, default='random',
                        help='recurrent hidden weight initialization type')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batchsize', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--vocabsize', type=int, default=200000,
                        help='random seed')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--freezeemb', action='store_false',
                        help='freezes embeddings')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='',
                        help='path to save the final model')
    args = parser.parse_args()


    X, y, X_val, y_val, q_field = load_data(args.data, args.din, args.vocabsize, args.cuda, train_split=0.8)

    print('Generating Data Loaders')
    #X.size len(train_data),1,2,fix_length
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, 
                                batch_size=args.batchsize, 
                                shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=args.batchsize,
                                shuffle=False)


    ntokens = len(q_field.vocab.itos)
    # print(ntokens)
    glove_embeddings = None
    if args.embinit == 'glove':
        assert args.demb in (50, 100, 200, 300)
        glove_embeddings = get_glove_embeddings(args.glovedata, q_field.vocab.stoi, ntokens, args.demb)
    

    autoencoder = torch.load('autoencoder.pt')
    model = LSTMModel(args.din, args.dhid, args.nlayers, args.dout, args.demb, args.vocabsize, 
                        args.dropout, args.embinit, args.hidinit, args.decinit, glove_embeddings,
                        args.freezeemb, args.cuda)

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_config = '\t'.join([str(x) for x in (torch.__version__, args.clip, args.nlayers, args.din, args.demb, args.dhid, 
                        args.embinit, args.freezeemb, args.decinit, args.hidinit, args.dropout, args.optimizer, args.lr, args.vocabsize)])

    print('Pytorch | Clip | #Layers | InSize | EmbDim | HiddenDim | EncoderInit | EncoderGrad | DecoderInit | WeightInit | Dropout | Optimizer| LR | VocabSize')
    print(model_config)

    for epoch in range(args.epochs):
        model.train()
        total_cost = 0
        start_time = time.time()
        for ind, (qs, duplicate) in enumerate(train_loader):
            if args.cuda:
                qs = qs.cuda()
                duplicate = duplicate.cuda()
            duplicate = Variable(duplicate)
            model.zero_grad()
            pred = model(qs[:, 0, 0, :].long(), qs[:, 0, 1, :].long())
            loss = criterion(pred, duplicate)
            loss.backward()
            clip_grad_norm(model.parameters(), args.clip)

            if optimizer:
                optimizer.step()
            else:
                for p in model.parameters():
                    p.data.add_(-args.lr, p.grad.data)

            total_cost += loss.data[0]

            if ind % args.loginterval == 0 and ind > 0:
                # cur_loss = total_cost / (ind * args.batchsize)
                cur_loss = loss.data[0] / args.batchsize
                elapsed = time.time() - start_time
                print('| Epoch {:3d} | {:5d}/{:5d} Batches | ms/batch {:5.2f} | '
                        'Loss {:.6f}'.format(
                            epoch, ind, len(X) // args.batchsize,
                            elapsed * 1000.0 / args.loginterval, cur_loss))
                start_time = time.time()

        model.eval()
        train_correct, train_total = 0, 0
        for ind, (qs, duplicate) in enumerate(train_loader):
            if args.cuda:
                qs = qs.cuda()
            out = model(qs[:, 0, 0, :].long(), qs[:, 0, 1, :].long())
            # out = model(qs[:, 0, 0, :].long().cuda(), qs[:, 0, 1, :].long().cuda())
            pred = out.data.cpu().numpy().argmax(axis=1)
            train_correct += np.sum(pred == duplicate.cpu().numpy())   
            train_total += len(pred)
        train_acc = train_correct / train_total 

        val_correct, val_total = 0, 0
        for ind, (qs, duplicate) in enumerate(valid_loader):
            if args.cuda:
                qs = qs.cuda()
            out = model(qs[:, 0, 0, :].long(), qs[:, 0, 1, :].long())
            # out = model(qs[:, 0, 0, :].long().cuda(), qs[:, 0, 1, :].long().cuda())
            pred = out.data.cpu().numpy().argmax(axis=1)
            val_correct += np.sum(pred == duplicate.cpu().numpy())
            val_total += len(pred)
        acc = val_correct/val_total

        print('Epoch: {} | Train Loss: {:.4f} | Train Accuracy: {:.4f} | Val Accuracy: {:.4f}'.format(
            epoch, total_cost, train_acc, acc))
        print('-' * 89)


if __name__ == '__main__':
    main()
