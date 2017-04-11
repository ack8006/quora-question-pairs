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
from torchtext import datasets

from nltk.tokenize import word_tokenize

# import pandas as pd
# from torch.nn.utils import clip_grad_norm

from models import LSTMModel

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


def load_data(data_path, d_in, vocab_size, cuda, train_split = 0.8):
    print('Loading Data')
    train_data = pd.read_csv(data_path)
    val_data = train_data.iloc[int(len(train_data)*train_split):]
    train_data = train_data.iloc[:int(len(train_data)*train_split)]

    print('Cleaning and Tokenizing')
    q1, q2, y = clean_and_tokenize(train_data)
    q1_val, q2_val, y_val = clean_and_tokenize(val_data)

    question_field = data.Field(sequential=True, use_vocab=True, lower=True,
                                fix_length=d_in)
    question_field.build_vocab(q1 + q2 + q1_val + q2_val, {'max_size': vocab_size})

    device = -1
    if cuda:
        device = 1

    print('Padding and Shaping')
    X, y = pad_and_shape(question_field, q1, q2, y, len(train_data), device)
    X_val, y_val = pad_and_shape(question_field, q1_val, q2_val, y_val, len(val_data), device)

    return X, y, X_val, y_val


def clean_and_tokenize(data):
    q1 = list(data['question1'].map(str).apply(str.lower))
    q2 = list(data['question2'].map(str).apply(str.lower))
    y = list(data['is_duplicate'])
    q1 = [word_tokenize(x) for x in q1]
    q2 = [word_tokenize(x) for x in q2]
    return q1, q2, y


def pad_and_shape(field, q1, q2, y, num_samples, cuda):
    q1_pad_num = field.numericalize(field.pad(q1), device=cuda)
    q2_pad_num = field.numericalize(field.pad(q2), device=cuda)
    X = torch.Tensor(1, 2, d_in, num_samples)
    X[0, 0, :, :] = q1_pad_num.data
    X[0, 1, :, :] = q2_pad_num.data
    X.transpose_(0, 3).transpose_(2, 3).transpose_(1, 2)
    y = torch.from_numpy(np.array(y)).long()
    return X, y


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/train.csv',
                        help='location of the data corpus')
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
                        help='encoder weight initialization type')
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
    parser.add_argument('--vocabsize', type=int, default=20000,
                        help='random seed')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='',
                        help='path to save the final model')
    args = parser.parse_args()



    X, y, X_val, y_val = load_data(args.data, args.din, args.vocabsize, args.cuda, train_split=0.8)

    print('Generating Data Loaders')
    #X.size len(train_data),1,2,fix_length
    train_loader = DataLoader(TensorDataset(X, y), 
                                batch_size=args.batchsize, 
                                shuffle=True)
    # train_loader2 = DataLoader(TensorDataset(X, y),
    #                             batch_size = len(X),
    #                             shuffle=False)
    valid_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=len(X_val),
                                shuffle=False)

    # vocab_size = len(question_field.vocab.itos)

    model = LSTMModel(args.din, args.dhid, args.nlayers, args.dout, args.demb, args.vocabsize, 
                        args.dropout, args.embinit, args.hidinit, args.decinit)

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_cost = 0
        for ind, (qs, duplicate) in enumerate(train_loader):
            start_time = time.time()
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

            if (ind * args.batchsize) % args.log_interval == 0 and ind > 0:
                cur_loss = total_cost / (ind * args.batchsize)
                elapsed = time.time() - start_time
                print('| Epoch {:3d} | {:5d}/{:5d} Batches | ms/batch {:5.2f} | '
                        'Loss {:.6f}'.format(
                            epoch, ind, len(X) // args.batchsize,
                            elapsed * 1000.0 / args.log_interval, cur_loss))

        for ind, (qs, duplicate) in enumerate(valid_loader):
            model.eval()
            out = model(qs[:, 0, 0, :].long(), qs[:, 0, 1, :].long())
            pred = out.data.numpy().argmax(axis=1)
            valid_loss = criterion(pred, duplicate).data[0]
            acc = np.mean(pred == duplicate.numpy())

            print('Epoch: {} | Train Loss: {:.4f} | Valid Loss: {:.4f} | Accuracy: {:.4f}'.format(
                epoch, total_cost, valid_loss, acc))
        print('-' * 89)


if __name__ == '__main__':
    main()
