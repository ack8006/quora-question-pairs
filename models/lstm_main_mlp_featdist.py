import sys

import argparse
import time
import pickle as pkl
import functools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.metrics import log_loss

from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from models2 import LSTMModelMLPFeatDist
sys.path.append('../utils/')
from data import TacoText
from preprocess import clean_and_tokenize, pad_and_shape
from pipeline import pipeline


def get_glove_embeddings(file_path, corpus, ntoken, nemb):
    file_name = '/glove.6B.{}d.txt'.format(nemb)
    f = open(file_path+file_name, 'r')
    embeddings = torch.nn.init.xavier_uniform(torch.Tensor(ntoken, nemb))
    for line in f:
        split_line = line.split()
        word = split_line[0]
        if word in corpus:
            embedding = torch.Tensor([float(val) for val in split_line[1:]])
            embeddings[corpus[word]] = embedding
    return embeddings


def evaluate(model, data_loader, cuda, d_in, n_feat):

    correct, total = 0, 0
    pred_list = []
    true_list = []
    for ind, (qs, duplicate) in enumerate(data_loader):
        out = model(qs[:, 0, 0, :].long(), qs[:, 0, 1, :].long(), qs[:, 0, 2, :n_feat])
        pred = out.data.max(1)[1]
        if cuda:
            pred = pred.cuda()
            duplicate = duplicate.cuda()
        correct += (pred == duplicate).sum()
        total += len(pred)
        pred_list += list(out.exp()[:, 1].data.cpu().numpy())
        true_list += list(duplicate.cpu().numpy())
    return (correct / total), log_loss(true_list, pred_list, eps=1e-7)

def feature_gen(x):
    f = []
    f.append(abs(len(x[0]) - len(x[1])))  #WCDifference
    wic = len(set(x[0]).intersection(set(x[1])))
    f.append(wic) #NumWordsInCommon
    uw = len(set(x[0]).union(set(x[1])))
    f.append(uw) #Num unique words
    f.append(wic/uw) #Jaccard
    f.append(f[3]/len(set(x[0])))  #Pct Overlap Q1
    f.append(int((f[3]/len(set(x[0]))) < 0.1))
    f.append(int((f[3]/len(set(x[0]))) < 0.2))
    f.append(int((f[3]/len(set(x[0]))) < 0.3))
    f.append(int((f[3]/len(set(x[0]))) < 0.4))
    f.append(int((f[3]/len(set(x[0]))) < 0.5))
    f.append(f[3]/len(set(x[1])))  #Pct Overlap Q2
    f.append(int((f[3]/len(set(x[1]))) < 0.1))
    f.append(int((f[3]/len(set(x[1]))) < 0.2))
    f.append(int((f[3]/len(set(x[1]))) < 0.3))
    f.append(int((f[3]/len(set(x[1]))) < 0.4))
    f.append(int((f[3]/len(set(x[1]))) < 0.5))
    f.append(int((wic/uw) < 0.1))
    f.append(int((wic/uw) < 0.2))
    f.append(int((wic/uw) < 0.3))
    f.append(int((wic/uw) < 0.4))
    f.append(int((wic/uw) < 0.5))
    f.append(int(x[0][0].lower() == x[1][0].lower()))
#     for q in ('who','what','when','where','why','how','which'):
#         f.append(int(x[0][0].lower() == q))
#         f.append(int(x[1][0].lower() == q))
    return f


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the data corpus')
    parser.add_argument('--presaved', action='store_true',
                        help='use presaved data')
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
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed')
    parser.add_argument('--vocabsize', type=int, default=200000,
                        help='random seed')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--pipeline', action='store_true',
                        help='use pipeline file')
    parser.add_argument('--psw', type=int, default=1,
                        help='remove stop words')
    parser.add_argument('--ppunc', action='store_true',
                        help='remove punctuation')
    parser.add_argument('--pntok', action='store_true',
                        help='use number tokens')
    parser.add_argument('--pkq', action='store_true',
                        help='keep question words')
    parser.add_argument('--stem', action='store_true',
                        help='use stemmer')
    parser.add_argument('--lemma', action='store_true',
                        help='use lemmatizer')
    parser.add_argument('--freezeemb', action='store_false',
                        help='freezes embeddings')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--loginterval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='',
                        help='path to save the final model')
    args = parser.parse_args()


    if not args.presaved:
        pipe = None
        if args.pipeline:
            stemmer, lemmatizer = None, None
            if args.stem:
                stemmer = SnowballStemmer('english')
            elif args.lemma:
                lemmatizer = WordNetLemmatizer()

            pipe = functools.partial(pipeline, 
                                    rm_stop_words=args.psw, 
                                    rm_punc=args.ppunc, 
                                    number_token=args.pntok, 
                                    keep_questions=args.pkq,
                                    stemmer=stemmer,
                                    lemmatizer=lemmatizer)

        corpus = TacoText(args.vocabsize, lower=True, vocab_pipe=pipe)
        print('Loading Data')
        # train_data = pd.read_csv(args.data)
        #Shuffle order of training data

        # train_data = train_data.reindex(np.random.permutation(train_data.index))
        # val_data = train_data.iloc[int(len(train_data) * 0.9):]
        # train_data = train_data.iloc[:int(len(train_data) * 0.9)]
        train_data = pd.read_csv('../data/train_data_shuffle.csv')
        val_data = pd.read_csv('../data/val_data_shuffle.csv')

        print('Cleaning and Tokenizing')
        q1, q2, y = clean_and_tokenize(train_data, corpus)
        q1_val, q2_val, y_val = clean_and_tokenize(val_data, corpus)

        train_feat = list(map(feature_gen, zip(q1, q2)))
        val_feat = list(map(feature_gen, zip(q1_val, q2_val)))
        scalar = preprocessing.StandardScaler()
        train_feat = scalar.fit_transform(train_feat)
        val_feat = scalar.transform(val_feat)

        print('Piping Data')
        q1 = corpus.pipe_data(q1)
        q2 = corpus.pipe_data(q2)
        q1_val = corpus.pipe_data(q1_val)
        q2_val = corpus.pipe_data(q2_val)

        corpus.gen_vocab(q1 + q2 + q2_val + q1_val)

        n_feat = train_feat.shape[1]
        d_in = args.din
        feat_max = int(np.max([n_feat, d_in]))

        X = torch.Tensor(len(train_data), 1, 3, feat_max)
        X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, feat_max)).long()
        X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, feat_max)).long()
        X[:, 0, 2, :n_feat] = torch.from_numpy(np.array(train_feat))
        y = torch.from_numpy(np.array(y)).long()

        X_val = torch.Tensor(len(val_data), 1, 3, feat_max)
        X_val[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1_val, feat_max)).long()
        X_val[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2_val, feat_max)).long()
        X_val[:, 0, 2, :n_feat] = torch.from_numpy(np.array(val_feat))
        y_val = torch.from_numpy(np.array(y_val)).long()

        torch.save(X, '../data/X_featd.t')
        torch.save(y, '../data/y_featd.t')
        torch.save(X_val, '../data/X_val_featd.t')
        torch.save(y_val, '../data/y_val_featd.t')
        with open('../data/corpus_featd.pkl', 'wb') as corp_f:
            pkl.dump(corpus, corp_f, protocol=pkl.HIGHEST_PROTOCOL)

    else:
        n_feat = 22
        d_in = args.din
        print('Loading Presaved Data')
        X = torch.load(args.data + 'X_featd.t')
        y = torch.load(args.data + 'y_featd.t')
        X_val = torch.load(args.data + 'X_val_featd.t')
        y_val = torch.load(args.data + 'y_val_featd.t')
        with open('../data/corpus_featd.pkl', 'rb') as f:
            corpus = pkl.load(f)


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

    model = LSTMModelMLPFeatDist(args.din, args.dhid, args.nlayers, args.dout, args.demb, n_feat, args.vocabsize, 
                        args.dropout, args.embinit, args.hidinit, args.decinit, glove_embeddings,
                        args.cuda)

    if args.cuda:
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_config = '\t'.join([str(x) for x in (torch.__version__, args.clip, args.nlayers, args.din, args.demb, args.dhid, 
                        args.embinit, args.decinit, args.hidinit, args.dropout, args.optimizer, args.lr, args.vocabsize,
                        args.pipeline, args.psw, args.ppunc, args.pntok, args.pkq, args.stem, args.lemma)])

    print('Pytorch | Clip | #Layers | InSize | EmbDim | HiddenDim | EncoderInit | DecoderInit | WeightInit | Dropout | Optimizer| LR | VocabSize | pipeline | stop | punc | ntoken | keep_ques | stem | lemma')
    print(model_config)

    # best_val_acc = 0.78
    best_ll = 0.5
    for epoch in range(args.epochs):
        model.train()
        total_cost = 0
        start_time = time.time()
        cur_loss = 0
        for ind, (qs, duplicate) in enumerate(train_loader):
            model.zero_grad()
            pred = model(qs[:, 0, 0, :d_in].long(), qs[:, 0, 1, :d_in].long(), qs[:, 0, 2, :n_feat])
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

        train_acc, train_ll = evaluate(model, train_loader, args.cuda, d_in, n_feat)
        val_acc, val_ll = evaluate(model, valid_loader, args.cuda, d_in, n_feat)
        # if args.save and (val_acc > best_val_acc):
        if args.save and (val_ll < best_ll):
            with open(args.save + '_corpus.pkl', 'wb') as corp_f:
                pkl.dump(corpus, corp_f, protocol=pkl.HIGHEST_PROTOCOL)
            torch.save(model.cpu(), args.save)
            torch.save(model.cpu().state_dict(), args.save + ".state_dict")
            with open(args.save + ".state_dict.config", "w") as f:
                f.write(model_config)
            best_ll = val_ll
            if args.cuda:
                model.cuda()


        print('Epoch: {} | Train Loss: {:.4f} | Train Accuracy: {:.4f} | Val Accuracy: {:.4f} | Train LL: {:.4f} | Val LL: {:.4f}'.format(
            epoch, total_cost, train_acc, val_acc, train_ll, val_ll))
        print('-' * 89)


if __name__ == '__main__':
    main()
