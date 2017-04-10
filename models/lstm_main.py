import sys
sys.path.append('../models/text/')

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
epochs = 5
d_out = 2
d_hid = 50
n_layers = 1
lr = 0.01
dropout = 0.0
cuda = False


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

    vocab_size = len(question_field.vocab.itos)

    model = LSTMModel(d_in, d_hid, n_layers, d_out, d_emb, vocab_size)

    if cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for ind, (qs, dup) in enumerate(train_loader):
            model.zero_grad()
            print(model(qs[:, 0, 0, :], qs[:, 0, 1, :]))

            





if __name__ == '__main__':
    main()








