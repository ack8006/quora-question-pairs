import functools

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

#TODO
'''
- Batchnorm
'''

class MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(MLP, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module('linear1', nn.Linear(d_in, 
                                                        d_out, 
                                                        bias=True))
        self.sequential.add_module('softmax', nn.LogSoftmax())


    def forward(self, X):
        out = self.sequential(X)
        return out


class BiLSTM(nn.Module):
    def __init__(self, d_emb, d_hid, n_layers, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(d_emb, 
                            d_hid, 
                            n_layers,
                            bias=True,
                            batch_first=True, 
                            bidirectional=True,
                            dropout = dropout)


    def forward(self, x, h):
        out, hid = self.lstm(x, h)
        #out, hidden, cell
        return out, hid[0], hid[1]


class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout=0.0,
                    init_type='random'):
        super(LSTMModel, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers

        self.drop = nn.Dropout(dropout)

        self.embedding1 = nn.Embedding(vocab, d_emb)
        self.bilstm_1 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        self.embedding2 = nn.Embedding(vocab, d_emb)
        self.bilstm_2 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        self.seq_3 = nn.Sequential()
        self.seq_3.add_module('drop', nn.Dropout(dropout))
        #d_hid * directions * questions * layers
        self.seq_3.add_module('mlp', MLP(d_hid*2*2*n_layers, d_out, dropout))

        self.init_weights('random')


    def init_weights(self, init_type):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        init_types[init_type](self.embedding1.weight)
        init_types[init_type](self.embedding2.weight)


    def forward(self, X1, X2):
        X1 = Variable(torch.from_numpy(X1.numpy()).long())
        X2 = Variable(torch.from_numpy(X2.numpy()).long())
        emb1 = self.drop(self.embedding1(X1))

        h1 = self.init_hidden(X1.size()[0])
        out1, hid1, cell1 = self.bilstm_1(emb1, h1)

        emb2 = self.drop(self.embedding2(X2))
        h2 = self.init_hidden(X2.size()[0])
        out2, hid2, cell2 = self.bilstm_2(emb2, h2)

        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid1.size()[0]), 2)[0]

        h_cat = torch.cat([hid1, hid2], 1)
        # h_cat = torch.cat([torch.cat([hid1[0], hid1[1]], 1), torch.cat([hid2[0], hid2[1]], 1)], 1)
        return self.seq_3(h_cat)


    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers*2, batch_size, self.d_hid)), 
                Variable(torch.zeros(self.n_layers*2, batch_size, self.d_hid)))








