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

d_in = 30
d_emb = 50
cuda = False
batch_size = 20
epochs = 20
d_out = 2
d_hid = 50
n_layers = 1
optimizer = True
lr = 0.05
dropout = 0.0
clip = 0.25
cuda = False
log_interval = 200


def main():
    print('Loading Data')
    train_data = pd.read_csv('../data/train.csv')
    val_data = train_data.iloc[int(len(train_data)*0.8):]
    # train_data = train_data.iloc[:int(len(train_data)*0.8)]
    train_data = train_data.iloc[:1000]
    q1 = list(train_data['question1'].map(str).apply(str.lower))
    q2 = list(train_data['question2'].map(str).apply(str.lower))
    y = list(train_data['is_duplicate'])
    # q1_val = list(val_data['question1'].map(str).apply(str.lower))
    # q2_val = list(val_data['question2'].map(str).apply(str.lower))
    # y_val = list(val_data['is_duplicate'])

    print('Shaping Data')
    q1 = [word_tokenize(x) for x in q1[:1000]]
    q2 = [word_tokenize(x) for x in q2[:1000]]
    # q1_val = [word_tokenize(x) for x in q1_val]
    # q2_val = [word_tokenize(x) for x in q2_val]

    question_field = data.Field(sequential=True, use_vocab=True, lower=True,
                            fix_length=d_in)
    # question_field.build_vocab(q1+q2+q1_val+q2_val)
    question_field.build_vocab(q1+q2)

    device = -1
    if cuda:
        device = 1

    q1_pad_num = question_field.numericalize(question_field.pad(q1), device=device)
    q2_pad_num = question_field.numericalize(question_field.pad(q2), device=device)

    print(q1_pad_num.size())
    X = torch.Tensor(1, 2, d_in, len(train_data))
    print(X.size())
    X[0,0,:,:] = q1_pad_num.data
    X[0,1,:,:] = q2_pad_num.data
    X.transpose_(0,3).transpose_(2,3).transpose_(1,2)
    y = torch.from_numpy(np.array(y)).long()

    print('Generating Data Loaders')
    #X.size len(train_data),1,2,fix_length
    train_loader = DataLoader(TensorDataset(X, y), 
                                batch_size=batch_size, 
                                shuffle=True)
    train_loader2 = DataLoader(TensorDataset(X, y),
                                batch_size = len(X),
                                shuffle=False)

    vocab_size = len(question_field.vocab.itos)

    model = LSTMModel(d_in, d_hid, n_layers, d_out, d_emb, vocab_size)

    if cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    for epoch in range(epochs):
        model.train()
        total_cost = 0
        for ind, (qs, duplicate) in enumerate(train_loader):
            start_time = time.time()
            duplicate = Variable(duplicate)
            model.zero_grad()
            pred = model(qs[:, 0, 0, :], qs[:, 0, 1, :])
            loss = criterion(pred, duplicate)
            loss.backward()
            clip_grad_norm(model.parameters(), clip)

            if optimizer:
                optimizer.step()
            else:
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)

            total_cost += loss.data[0]

            if (ind*batch_size) % log_interval == 0 and ind > 0:
                cur_loss = total_cost / (ind*batch_size)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:.6f}'.format(
                    epoch, ind, len(X) // batch_size,
                    elapsed * 1000.0 / log_interval, cur_loss))

        for ind, (qs, duplicate) in enumerate(train_loader2):
            model.eval()
            pred = model(qs[:, 0, 0, :], qs[:, 0, 1, :]).data.numpy().argmax(axis=1)

            print('Epoch: {} | Accuracy: {:.4f} | Train Loss: {:.4f}'.format(
                epoch, np.mean(pred == duplicate.numpy()), total_cost))
        print('-' * 89)

            





if __name__ == '__main__':
    main()








