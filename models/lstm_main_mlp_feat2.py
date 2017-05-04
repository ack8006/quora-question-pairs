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
from nltk.corpus import stopwords

from models2 import LSTMModelMLPFeat
sys.path.append('../utils/')
from data import TacoText
from preprocess import clean_and_tokenize, pad_and_shape


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

def word_match_share(row, stops):
    q1words = {}
    q2words = {}
    for word in row[0]:
        if word not in stops:
            q1words[word] = 1
    for word in row[1]:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def wc_ratio(row):
    l1 = len(row[0]) * 1.0 
    l2 = len(row[1])
    if l2 == 0:
        return 0.0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_ratio_unique(row):
    l1 = len(set(row[0])) * 1.0
    l2 = len(set(row[1]))
    if l2 == 0:
        return 0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row[0]) if x not in stops]) - len([x for x in set(row[1]) if x not in stops]))

def char_diff(row):
    return abs(len(''.join(row[0])) - len(''.join(row[1])))

def char_ratio(row):
    l1 = len(''.join(row[0])) 
    l2 = len(''.join(row[1]))
    if l2 == 0:
        return 0
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
def char_diff_unique_stop(row, stops):
    return abs(len(''.join([x for x in set(row[0]) if x not in stops])) - len(''.join([x for x in set(row[1]) if x not in stops])))

def feature_gen(x):
    stops = set(stopwords.words("english"))
    f = []
    f.append(word_match_share(x, stops)) #word share match
    f.append(abs(len(x[0]) - len(x[1])))  #WCDifference
    f.append(abs(len(set(x[0])) - len(set(x[1]))))  #WCDifferenceUnique
    f.append(wc_ratio(x)) #Word Count Ratio
    f.append(wc_ratio_unique(x))  #Unique wordcount ratio
    f.append(wc_diff_unique_stop(x, stops)) 
    f.append(char_diff(x)) #Character Difference
    f.append(char_ratio(x)) #Character Ratio
    f.append(char_diff_unique_stop(x, stops)) #Character Ratio
    wic = len(set(x[0]).intersection(set(x[1])))
    f.append(wic) #NumWordsInCommon
    uw = len(set(x[0]).union(set(x[1])))
    f.append(uw) #Num unique words
    if uw == 0: 
        uw = 1
    pct_overlap = 0
    if len(set(x[0])) > 0:
        pct_overlap = wic / len(set(x[0]))
    f.append(pct_overlap)  #Pct Overlap Q1
    f.append(int(pct_overlap < 0.1))
    f.append(int(pct_overlap < 0.2))
    f.append(int(pct_overlap < 0.3))
    f.append(int(pct_overlap < 0.4))
    f.append(int(pct_overlap < 0.5))
    pct_overlap = 0
    if len(set(x[1])) > 0:
        pct_overlap = wic / len(set(x[1]))
    f.append(pct_overlap)  #Pct Overlap Q2
    f.append(int(pct_overlap < 0.1))
    f.append(int(pct_overlap < 0.2))
    f.append(int(pct_overlap < 0.3))
    f.append(int(pct_overlap < 0.4))
    f.append(int(pct_overlap < 0.5))
    f.append(wic/uw) #Jaccard
    f.append(int((wic/uw) < 0.1))
    f.append(int((wic/uw) < 0.2))
    f.append(int((wic/uw) < 0.3))
    f.append(int((wic/uw) < 0.4))
    f.append(int((wic/uw) < 0.5))
    if x[0] and x[1]:
        f.append(int(x[0][0].lower() == x[1][0].lower())) #same start word
    else:
        f.append(0)
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
    parser.add_argument('--reweight', action='store_true',
                        help='reweight loss function')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batchsize', type=int, default=2000, metavar='N',
                        help='batch size')
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed')
    parser.add_argument('--vocabsize', type=int, default=200000,
                        help='random seed')
    parser.add_argument('--optimizer', action='store_true',
                        help='use ADAM optimizer')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--loginterval', type=int, default=20, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='',
                        help='path to save the final model')
    args = parser.parse_args()


    pipe = None
    corpus = TacoText(args.vocabsize, lower=True, vocab_pipe=pipe)
    train_data = pd.read_csv('../data/train.csv')
    train_data = train_data.fillna(' ')
    train_data = train_data.sample(frac=1)
    valid_data = train_data.iloc[int(0.9 * len(train_data)):]
    train_data = train_data.iloc[:int(0.9 * len(train_data))]

    print('Downsampling')
    #downsample
    pos_valid = valid_data[valid_data['is_duplicate'] == 1]
    neg_valid = valid_data[valid_data['is_duplicate'] == 0]
    p = 0.19
    pl = len(pos_valid)
    tl = len(pos_valid) + len(neg_valid)
    val = int(pl - (pl - p * tl) / ((1 - p)))
    pos_valid = pos_valid.iloc[:int(val)]
    valid_data = pd.concat([pos_valid, neg_valid])

    print('Splitting Train')
    q1 = list(train_data['question1'].map(str))
    q2 = list(train_data['question2'].map(str))
    y = list(train_data['is_duplicate'])
    q1 = [x.lower().split() for x in q1]
    q2 = [x.lower().split() for x in q2]

    print('Splitting Valid')
    q1_val = list(valid_data['question1'].map(str))
    q2_val = list(valid_data['question2'].map(str))
    y_val = list(valid_data['is_duplicate'])
    q1_val = [x.lower().split() for x in q1_val]
    q2_val = [x.lower().split() for x in q2_val]

    corpus.gen_vocab(q1 + q2 + q2_val + q1_val)

    train_feat = list(map(feature_gen, zip(q1, q2)))
    val_feat = list(map(feature_gen, zip(q1_val, q2_val)))
    scalar = preprocessing.StandardScaler()
    train_feat = scalar.fit_transform(train_feat)
    val_feat = scalar.transform(val_feat)

    n_feat = train_feat.shape[1]
    d_in = args.din
    feat_max = int(np.max([n_feat, d_in]))

    X = torch.Tensor(len(train_data), 1, 3, feat_max)
    X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, feat_max)).long()
    X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, feat_max)).long()
    X[:, 0, 2, :n_feat] = torch.from_numpy(np.array(train_feat))
    y = torch.from_numpy(np.array(y)).long()

    X_val = torch.Tensor(len(valid_data), 1, 3, feat_max)
    X_val[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1_val, feat_max)).long()
    X_val[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2_val, feat_max)).long()
    X_val[:, 0, 2, :n_feat] = torch.from_numpy(np.array(val_feat))
    y_val = torch.from_numpy(np.array(y_val)).long()

    # torch.save(X, '../data/X_feat.t')
    # torch.save(y, '../data/y_feat.t')
    # torch.save(X_val, '../data/X_val_feat.t')
    # torch.save(y_val, '../data/y_val_feat.t')
    # with open(args.save + '_corpus_feat.pkl', 'wb') as corp_f:
    #     pkl.dump(corpus, corp_f, protocol=pkl.HIGHEST_PROTOCOL)

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

    model = LSTMModelMLPFeat(args.din, args.dhid, args.nlayers, args.dout, args.demb, n_feat, args.vocabsize, 
                        args.dropout, args.embinit, args.hidinit, args.decinit, glove_embeddings,
                        args.cuda)

    if args.cuda:
        model.cuda()

    if args.reweight:
        w_tensor = torch.Tensor([1.309028344, 0.472001959])
        if args.cuda:
            w_tensor = w_tensor.cuda()
        criterion = nn.NLLLoss(weight=w_tensor)
    else:
        criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_config = '\t'.join([str(x) for x in (torch.__version__, args.clip, args.nlayers, args.din, args.demb, args.dhid, 
                        args.embinit, args.decinit, args.hidinit, args.dropout, args.optimizer, args.lr, args.vocabsize,
                        )])

    print('Pytorch | Clip | #Layers | InSize | EmbDim | HiddenDim | EncoderInit | DecoderInit | WeightInit | Dropout | Optimizer| LR | VocabSize | pipeline | stop | punc | ntoken | keep_ques | stem | lemma')
    print(model_config)

    # best_val_acc = 0.78
    best_ll = 0.4
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

    del train_loader, valid_loader, X, y, X_val, y_val

    print('Reloading Best Modelgit status')
    model = torch.load(args.save)
    model.cuda()

    print('LOADING TEST DATA')
    test_data = pd.read_csv('../data/test.csv')
    test_data = test_data.fillna(' ')
    q1 = list(test_data['question1'].map(str))
    q2 = list(test_data['question2'].map(str))
    y = list(test_data['is_duplicate'])
    q1 = [x.lower().split() for x in q1]
    q2 = [x.lower().split() for x in q2]

    print('GENERATING TEST FEATURES')
    test_feat = list(map(feature_gen, zip(q1, q2)))
    test_feat = scalar.transform(test_feat)

    n_feat = test_feat.shape[1]
    d_in = args.din
    feat_max = int(np.max([n_feat, d_in]))

    X = torch.Tensor(len(test_data), 1, 3, feat_max)
    X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, feat_max)).long()
    X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, feat_max)).long()
    X[:, 0, 2, :n_feat] = torch.from_numpy(np.array(test_feat))
    y = torch.from_numpy(np.array(y)).long()

    X = X.cuda()
    y = y.cuda()

    test_loader = DataLoader(TensorDataset(X, y),
                                batch_size=500,
                                shuffle=False)

    print('PREDICTING')
    pred_list = []
    for ind, (qs, _) in enumerate(test_loader):
        out = model(qs[:, 0, 0, :d_in].long(), qs[:, 0, 1, :d_in].long(), qs[:, 0, 2, :n_feat])
        pred_list += list(out.exp()[:, 1].data.cpu().numpy())

    with open('../predictions/'+ args.save +'.pkl', 'wb') as f:
        pkl.dump(f)

if __name__ == '__main__':
    main()
