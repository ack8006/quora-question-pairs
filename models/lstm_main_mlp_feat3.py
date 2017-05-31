import sys

import argparse
import time
import pickle as pkl

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
from preprocess import clean_and_tokenize, pad_and_shape, split_text


def get_glove_embeddings(file_path, corpus, ntoken, nemb):
    file_name = 'glove.840B.300d.txt'.format(nemb)
    f = open(file_path+file_name, 'r')
    embeddings = torch.nn.init.xavier_uniform(torch.Tensor(ntoken, nemb))
    for line in f:
        split_line = line.split()
        word = ' '.join(split_line[0:-300])
        if word in corpus:
            embedding = torch.Tensor([float(val) for val in split_line[-300:]])
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


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the data corpus')
    parser.add_argument('--presaved', action='store_true',
                        help='use presaved data')
    parser.add_argument('--glovedata', type=str, default='../data/',
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

    parser.add_argument('--clean', action='store_true',
                        help='clean text')
    parser.add_argument('--rm_stops', action='store_true',
                        help='remove stop words')

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
    train_data = pd.read_csv('../data/train_data_shuffle.csv')
    valid_data = pd.read_csv('../data/val_data_shuffle.csv')
    train_data = train_data.fillna(' ')
    valid_data = valid_data.fillna(' ')

    if args.reweight:
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

    print('Splitting Valid')
    q1_val = list(valid_data['question1'].map(str))
    q2_val = list(valid_data['question2'].map(str))
    y_val = list(valid_data['is_duplicate'])

    train_feat = pd.read_csv('../data/train_features_all_norm.csv')
    val_feat = train_feat.iloc[valid_data['id']].values
    train_feat = train_feat.iloc[train_data['id']].values

    print('Splitting Data')
    if args.clean:
        print('Cleaning Data')
        stops = None
        if args.rm_stops:
            stops = stops = set(stopwords.words("english"))
        q1 = [split_text(x, stops) for x in q1]
        q2 = [split_text(x, stops) for x in q2]
        q1_val = [split_text(x, stops) for x in q1_val]
        q2_val = [split_text(x, stops) for x in q2_val] 
    else:
        q1 = [x.lower().split() for x in q1]
        q2 = [x.lower().split() for x in q2]
        q1_val = [x.lower().split() for x in q1_val]
        q2_val = [x.lower().split() for x in q2_val]    

    print('Downsample Weight: ', np.mean(y_val))

    corpus.gen_vocab(q1 + q2 + q2_val + q1_val)

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

    num_train = len(X)

    del X, y, X_val, y_val, train_feat, val_feat, q1, q2, q1_val, q2_val

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
                        args.embinit, args.decinit, args.hidinit, args.dropout, args.optimizer, args.reweight, args.lr, args.vocabsize,
                        args.batchsize, args.clean, args.rm_stops)])

    print('Pytorch | Clip | #Layers | InSize | EmbDim | HiddenDim | EncoderInit | DecoderInit | WeightInit | Dropout | Optimizer | Reweight | LR | VocabSize | batchsize | Clean | Stops')
    print(model_config)

    # best_val_acc = 0.78
    best_ll = 0.3
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
                            epoch, ind, num_train // args.batchsize,
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

    del train_loader

    print('Reloading Best Model')
    model = torch.load(args.save)
    model.cuda()
    model.eval()


    print('RELOADING VALID')

    valid_data = pd.read_csv('../data/val_data_shuffle.csv')
    valid_data = valid_data.fillna(' ')

    q1_val = list(valid_data['question1'].map(str))
    q2_val = list(valid_data['question2'].map(str))
    y_val = list(valid_data['is_duplicate'])

    train_feat = pd.read_csv('../data/train_features_all_norm.csv')
    val_feat = train_feat.iloc[valid_data['id']].values

    if args.clean:
        print('Cleaning Data')
        stops = None
        if args.rm_stops:
            stops = stops = set(stopwords.words("english"))
        q1_val = [split_text(x, stops) for x in q1_val]
        q2_val = [split_text(x, stops) for x in q2_val] 
    else:
        q1_val = [x.lower().split() for x in q1_val]
        q2_val = [x.lower().split() for x in q2_val]  


    X_val = torch.Tensor(len(valid_data), 1, 3, feat_max)
    X_val[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1_val, feat_max)).long()
    X_val[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2_val, feat_max)).long()
    X_val[:, 0, 2, :n_feat] = torch.from_numpy(np.array(val_feat))
    y_val = torch.from_numpy(np.array(y_val)).long()


    if args.cuda:
        X_val, y_val = X_val.cuda(), y_val.cuda()

    valid_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=args.batchsize,
                                shuffle=False)

    del X_val, y_val, train_feat, val_feat, q1_val, q2_val, valid_data

    print('PREDICTING VALID')
    pred_list = []
    for ind, (qs, _) in enumerate(valid_loader):
        out = model(qs[:, 0, 0, :d_in].long(), qs[:, 0, 1, :d_in].long(), qs[:, 0, 2, :n_feat])
        pred_list += list(out.exp()[:, 1].data.cpu().numpy())

    with open('../predictions/'+ args.save +'_val.pkl', 'wb') as f:
        pkl.dump(pred_list, f, protocol=pkl.HIGHEST_PROTOCOL)

    if args.reweight:
        print('LOADING TEST DATA')
        test_data = pd.read_csv('../data/test.csv')
        test_data = test_data.fillna(' ')
        q1 = list(test_data['question1'].map(str))
        q2 = list(test_data['question2'].map(str))
        q1 = [x.lower().split() for x in q1]
        q2 = [x.lower().split() for x in q2]

        print('LOADING TEST FEATURES')
        test_feat = pd.read_csv('../data/test_features_all_norm.csv').values

        n_feat = test_feat.shape[1]
        d_in = args.din
        feat_max = int(np.max([n_feat, d_in]))

        X = torch.Tensor(len(test_data), 1, 3, feat_max)
        X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, feat_max)).long()
        X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, feat_max)).long()
        X[:, 0, 2, :n_feat] = torch.from_numpy(np.array(test_feat))
        y = torch.LongTensor(len(test_data)).zero_()

        if args.cuda:
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
            pkl.dump(pred_list, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
