import functools

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

#TODO
'''
- 
'''

class MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(MLP, self).__init__()
        self.sequential = nn.Sequential()
#BATCHNORM
        self.sequential.add_module('linear1', nn.Linear(d_in, 
                                                        d_out, 
                                                        bias=True))
        self.sequential.add_module('softmax', nn.LogSoftmax())

    def forward(self, X):
        out = self.sequential(X)
        return out


class BiLSTM(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, dropout):
        super(BiLSTM, self).__init__()
        # self.d_hid = d_hid
        # self.n_layers = n_layers
        self.lstm = nn.LSTM(d_in, 
                            d_hid, 
                            n_layers,
                            bias=True,
                            # batch_first=True, 
                            bidirectional=True,
                            dropout = dropout)

    def forward(self, x, h):
        # h0 = Variable(torch.zeros(self.n_layers*2, x.size()[0], self.d_hid))
        # c0 = Variable(torch.zeros(self.n_layers*2, x.size()[0], self.d_hid))
        # out, _ = self.lstm(x, (h0, c0))
        print(x.size(), h[0].size(), h[1].size())
        out, _ = self.lstm(x, h)
        return out


class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout=0.0,
                    init_type='random'):
        super(LSTMModel, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers

        self.drop = nn.Dropout(dropout)

        self.embedding1 = nn.Embedding(vocab, d_emb)
        self.bilstm_1 = BiLSTM(d_in, d_hid, n_layers, dropout)

        self.embedding2 = nn.Embedding(vocab, d_emb)
        self.bilstm_2 = BiLSTM(d_in, d_hid, n_layers, dropout)

        self.seq_3 = nn.Sequential()
        self.seq_3.add_module('drop', nn.Dropout(dropout))
        self.seq_3.add_module('mlp', MLP(d_hid*2, d_out, dropout))

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
        # emb1 = self.embedding1(X1)
        # emb1 = self.drop(emb1)

        h1 = self.init_hidden(X1.size()[0])
        out1, hid1 = self.bilstm_1(emb1, h1)
        # out1, hid1 = self.bilstm_1(emb1)

        emb2 = self.drop(self.embedding2(X2))
        h2 = self.init_hidden(X2.size()[0])
        out2, hid2 = self.bilstm_2(emb2, h2)
        # out2, hid2 = self.bilstm_2(emb2)

#***Perhaps should be 2 not 1
        print(torch.cat([out1, out2], 1).size())
        return self.seq_3(torch.cat[out1, out2], 1)


    def init_hidden(self, batch_size):
        print('dhid:', self.d_hid)
        print(Variable(torch.zeros(self.n_layers*2, batch_size, self.d_hid)).size())
        return (Variable(torch.zeros(self.n_layers*2, batch_size, self.d_hid)), 
                Variable(torch.zeros(self.n_layers*2, batch_size, self.d_hid)))








