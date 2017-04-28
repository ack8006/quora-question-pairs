import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from models2 import MLP


class BiGRU(nn.Module):
    def __init__(self, d_emb, d_hid, dropout):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(d_emb, 
                            d_hid, 
                            1,
                            bias=True,
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)


    def forward(self, x, h):
        out, hid = self.gru(x, h)
        return out, hid


    def init_weights(self, weight_init):
        if not weight_init:
            return
        init_types = {'random': functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        for layer in self.gru._all_weights:
            for w in layer:
                if 'bias' not in w:
                    init_types[weight_init](getattr(self.gru, w))


class ConvRNN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, d_emb, d_lin, vocab, dropout, emb_init, 
                    hid_init, dec_init, glove_emb, is_cuda):
        super(ConvRNN, self).__init__()

        self.d_hid = d_hid
        self.d_lin = d_lin
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab, d_emb)
        self.rnn = BiGRU(d_emb, d_hid, dropout)

        self.linear = nn.Linear(2*d_hid + d_emb, d_lin)
        self.mlp = MLP(2 * d_lin, 512, 256, d_out, dropout)

        self.init_weights(emb_init, hid_init, dec_init, glove_emb)


    def init_weights(self, emb_init, hid_init, dec_init, glove_emb):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        if emb_init == 'glove' and glove_emb is not None:
            self.embedding.weight.data = glove_emb
        else:
            init_types[emb_init](self.embedding.weight)

        self.rnn.init_weights(hid_init)
        self.mlp.init_weights(dec_init)


    def forward(self, X1, X2):
        X1 = Variable(X1)
        X2 = Variable(X2)

        #emb1 (batch, seq_len, embsize)
        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size(0))
        out1, _ = self.rnn(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size(0))
        out2, _ = self.rnn(emb2, h2)

        X1_cat = torch.cat([emb1, out1], 2)
        X2_cat = torch.cat([emb2, out2], 2)
        
        y1_i = Variable(torch.Tensor(X1_cat.size(0), X1_cat.size(1), self.d_lin))
        y2_i = Variable(torch.Tensor(X2_cat.size(0), X2_cat.size(1), self.d_lin))
        for ind in range(X1_cat.size(1)):
            y1_i[:, ind, :] = F.tanh(self.linear(X1_cat[:, ind, :]))
            y2_i[:, ind, :] = F.tanh(self.linear(X2_cat[:, ind, :]))
            
        enc1 = torch.max(y1_i, 1)[0][:, 0, :]
        enc2 = torch.max(y2_i, 1)[0][:, 0, :]
        
        s_cat = torch.cat([enc1, enc2], 1)
        if self.is_cuda:
            s_cat = s_cat.cuda()

        return self.mlp(s_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(2, batch_size, self.d_hid).cuda()))
        else:
            return (Variable(torch.zeros(2, batch_size, self.d_hid)))



