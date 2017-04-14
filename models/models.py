import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

#TODO
'''
- Batchnorm
'''

class FC(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(FC, self).__init__()

        self.dropout = dropout
        self.fc = nn.Linear(d_in, d_out, bias=True)


    def forward(self, X):
        X = F.dropout(X, p=self.dropout)
        X = self.fc(X)
        return F.log_softmax(X)


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        init_types[weight_init](self.fc.weight)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.linear1 = nn.Linear(d_in, d_hidden, bias=True)
        self.linear2 = nn.Linear(d_hidden, d_out, bias=True)


    def forward(self, X):
        X = F.dropout(self.linear1(X), p=self.dropout)
        X = F.dropout(self.linear2(X), p=self.dropout)
        return F.log_softmax(X)


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        init_types[weight_init](self.linear1.weight)
        init_types[weight_init](self.linear2.weight)


class BiLSTM(nn.Module):
    def __init__(self, d_emb, d_hid, n_layers, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(d_emb, 
                            d_hid, 
                            n_layers,
                            bias=True,
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)


    def forward(self, x, h):
        out, hid = self.lstm(x, h)
        #out, hidden, cell
        return out, hid[0], hid[1]


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        for layer in self.lstm._all_weights:
            for w in layer:
                if 'bias' not in w:
                    init_types[weight_init](getattr(self.lstm, w))


class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, freeze_emb, is_cuda):
        super(LSTMModel, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding1 = nn.Embedding(vocab, d_emb)
        self.embedding1.weight.requires_grad = freeze_emb
        self.bilstm_1 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        self.embedding2 = nn.Embedding(vocab, d_emb)
        self.embedding2.weight.requires_grad = freeze_emb
        self.bilstm_2 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        #(d_hid * directions * questions * layers)
        self.fc = FC(d_hid * 2 * 2 * n_layers, d_out, dropout)

        self.init_weights(emb_init, hid_init, dec_init, glove_emb)


    def init_weights(self, emb_init, hid_init, dec_init, glove_emb):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        if emb_init == 'glove' and glove_emb is not None:
            self.embedding1.weight.data = glove_emb
            self.embedding2.weight.data = glove_emb
        else:
            init_types[emb_init](self.embedding1.weight)
            init_types[emb_init](self.embedding2.weight)

        self.bilstm_1.init_weights(hid_init)
        self.bilstm_2.init_weights(hid_init)
        self.fc.init_weights(dec_init)


    def forward(self, X1, X2):
        X1 = Variable(X1)
        X2 = Variable(X2)

        emb1 = self.drop(self.embedding1(X1))
        h1 = self.init_hidden(X1.size()[0])
        out1, hid1, cell1 = self.bilstm_1(emb1, h1)

        emb2 = self.drop(self.embedding2(X2))
        h2 = self.init_hidden(X2.size()[0])
        out2, hid2, cell2 = self.bilstm_2(emb2, h2)

        #Concatenates Hidden State Directions and Layers
        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid2.size()[0]), 2)[0]

        #Concatenates Question Hidden States Together
        h_cat = torch.cat([hid1, hid2], 1)
        return self.fc(h_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)))








