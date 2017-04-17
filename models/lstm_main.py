import sys
sys.path.append('../utils/')

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from data import TacoText

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from models import LSTMModel



def load_data(data_path, d_in, vocab_size, train_split = 0.90):
    print('Loading Data')
    train_data = pd.read_csv(data_path)
    val_data = train_data.iloc[int(len(train_data)*train_split):]
    train_data = train_data.iloc[:int(len(train_data)*train_split)]
    # val_data = train_data.iloc[1000:1100]
    # train_data = train_data.iloc[:1000]

    print('Cleaning and Tokenizing')
    q1, q2, y = clean_and_tokenize(train_data)
    q1_val, q2_val, y_val = clean_and_tokenize(val_data)

    corpus = TacoText(vocab_size, lower=True)
    corpus.gen_vocab(q1+q2+q2_val+q1_val)

    print('Padding and Shaping')
    X, y = pad_and_shape(corpus, q1, q2, y, len(train_data), d_in)
    X_val, y_val = pad_and_shape(corpus, q1_val, q2_val, y_val, len(val_data), d_in)

    return X, y, X_val, y_val, corpus


def clean_and_tokenize(data):
    q1 = list(data['question1'].map(str))
    q2 = list(data['question2'].map(str))
    y = list(data['is_duplicate'])
    q1 = [word_tokenize(x) for x in q1]
    q2 = [word_tokenize(x) for x in q2]
    return q1, q2, y


def pad_and_shape(corpus, q1, q2, y, num_samples, d_in):
    X = torch.Tensor(num_samples, 1, 2, d_in).long()
    X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, d_in)).long()
    X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, d_in)).long()
    y = torch.from_numpy(np.array(y)).long()
    return X, y


def get_glove_embeddings(file_path, corpus, ntoken, nemb):
    file_name = '/glove.6B.{}d.txt'.format(nemb)
    f = open(file_path+file_name, 'r')
    embeddings = torch.nn.init.xavier_uniform(torch.Tensor(ntoken, nemb))
    for line in f:
        split_line = line.split()
        word = split_line[0]
        if word in corpus:
            embedding = torch.Tensor([float(val) for val in split_line[1:]])
            # embeddings[corpus.dictionary.word2idx[word]] = embedding
            embeddings[corpus[word]] = embedding
    return embeddings


def evaluate(model, data_loader, cuda):
    correct, total = 0, 0
    for ind, (qs, duplicate) in enumerate(data_loader):
        out = model(qs[:, 0, 0, :], qs[:, 0, 1, :])
        pred = out.data.max(1)[1]
        if cuda:
            pred = pred.cuda()
            duplicate = duplicate.cuda()
        correct += (pred == duplicate).sum()
        total += len(pred)
    return correct / total 


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


    X, y, X_val, y_val, corpus = load_data(args.data, args.din, args.vocabsize, train_split=0.9)

    if args.cuda:
        X, y = X.cuda(), y.cuda()
        X_val, y_val = X_val.cuda(), y_val.cuda()

    print('Generating Data Loaders')
    #X.size len(train_data),1,2,fix_length
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, 
                                batch_size=args.batchsize, 
                                shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=args.batchsize,
                                shuffle=False)

    ntokens = len(corpus)
    glove_embeddings = None
    if args.embinit == 'glove':
        assert args.demb in (50, 100, 200, 300)
        glove_embeddings = get_glove_embeddings(args.glovedata, corpus.dictionary.word2idx, ntokens, args.demb)

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

    best_val_acc = 0.78
    for epoch in range(args.epochs):
        model.train()
        total_cost = 0
        start_time = time.time()
        cur_loss = 0
        for ind, (qs, duplicate) in enumerate(train_loader):
            model.zero_grad()
            pred = model(qs[:, 0, 0, :], qs[:, 0, 1, :])
            if args.cuda:
            	pred = pred.cuda()
            	duplicate = duplicate.cuda()
            duplicate = Variable(duplicate)
            loss = criterion(pred, duplicate)
            loss.backward()
            clip_grad_norm(model.parameters(), args.clip)

            if optimizer:
                optimizer.step()
            else:
                for p in model.parameters():
                    p.data.add_(-args.lr, p.grad.data)

            total_cost += loss.data[0]
            cur_loss += loss.data[0]

            if ind % args.loginterval == 0 and ind > 0:
                cur_loss = loss.data[0] / args.loginterval
                elapsed = time.time() - start_time
                print('| Epoch {:3d} | {:5d}/{:5d} Batches | ms/batch {:5.2f} | '
                        'Loss {:.6f}'.format(
                            epoch, ind, len(X) // args.batchsize,
                            elapsed * 1000.0 / args.loginterval, cur_loss))
                start_time = time.time()
                cur_loss = 0

        model.eval()
        train_acc = evaluate(model, train_loader, args.cuda())
        val_acc = evaluate(model, valid_loader, args.cuda())
        if args.save and (val_acc > best_val_acc):
            torch.save(model, args.save)
            torch.save(model.state_dict(), args.save + ".state_dict")
            with open(args.save + ".state_dict.config", "w") as f:
                f.write(model_config)


        print('Epoch: {} | Train Loss: {:.4f} | Train Accuracy: {:.4f} | Val Accuracy: {:.4f}'.format(
            epoch, total_cost, train_acc, val_acc))
        print('-' * 89)


if __name__ == '__main__':
    main()
