import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, d_in, d_hid1, d_hid2, d_out, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        print('MLP Dropout: ', dropout)

        self.linear1 = nn.Linear(d_in, d_hid1)
        self.linear2 = nn.Linear(d_hid1, d_hid2)
        self.linear3 = nn.Linear(d_hid2, d_out)

        self.mlp = nn.Sequential()
        self.mlp.add_module('linear1', self.linear1)
        self.mlp.add_module('relu1', nn.LeakyReLU(negative_slope=0.18))
        self.mlp.add_module('drop2', nn.Dropout(p=dropout))
        self.mlp.add_module('linear2', self.linear2)
        self.mlp.add_module('relu2', nn.LeakyReLU(negative_slope=0.18))
        self.mlp.add_module('drop3', nn.Dropout(p=dropout))
        self.mlp.add_module('linear3', self.linear3)
        self.mlp.add_module('logsoft', nn.LogSoftmax())


    def forward(self, X):
        return self.mlp(X)


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        init_types[weight_init](self.linear1.weight)
        init_types[weight_init](self.linear2.weight)
        init_types[weight_init](self.linear3.weight)


class MLP2(nn.Module):
    def __init__(self, d_in, d_hid1, d_hid2, d_hid3, d_out, dropout):
        super(MLP2, self).__init__()
        self.dropout = dropout
        print('MLP Dropout: ', dropout)

        self.linear1 = nn.Linear(d_in, d_hid1)
        self.linear2 = nn.Linear(d_hid1, d_hid2)
        self.linear3 = nn.Linear(d_hid2, d_hid3)
        self.linear4 = nn.Linear(d_hid3, d_out)

        self.mlp = nn.Sequential()
        self.mlp.add_module('linear1', self.linear1)
        self.mlp.add_module('relu1', nn.LeakyReLU(negative_slope=0.18))
        self.mlp.add_module('drop1', nn.Dropout(p=dropout))
        self.mlp.add_module('linear2', self.linear2)
        self.mlp.add_module('relu2', nn.LeakyReLU(negative_slope=0.18))
        self.mlp.add_module('drop2', nn.Dropout(p=dropout))
        self.mlp.add_module('linear3', self.linear3)
        self.mlp.add_module('drop3', nn.Dropout(p=dropout))
        self.mlp.add_module('linear4', self.linear4)
        self.mlp.add_module('logsoft', nn.LogSoftmax())


    def forward(self, X):
        return self.mlp(X)


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        init_types[weight_init](self.linear1.weight)
        init_types[weight_init](self.linear2.weight)
        init_types[weight_init](self.linear3.weight)
        init_types[weight_init](self.linear4.weight)


class BiGRU(nn.Module):
    def __init__(self, d_emb, d_hid, dropout, bidirectional):
        super(BiGRU, self).__init__()
        print('BiGRU Dropout: ', dropout)
        self.gru = nn.GRU(d_emb, 
                            d_hid, 
                            1,
                            bias=True,
                            batch_first=True, 
                            bidirectional=bidirectional,
                            dropout=dropout)


    def forward(self, x, h):
        return self.gru(x, h)


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


class BiLSTM(nn.Module):
    def __init__(self, d_emb, d_hid, dropout, bidirectional):
        super(BiLSTM, self).__init__()
        print('LSTM Dropout: ', dropout)
        self.lstm = nn.LSTM(d_emb, 
                            d_hid, 
                            1,
                            bias=True,
                            batch_first=True, 
                            bidirectional=bidirectional,
                            dropout=dropout)


    def forward(self, x, h):
        out, (hid, cell) = self.lstm(x, h)
        #out, hidden, cell
        return out, hid


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


class ConvRNN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, d_emb, d_lin, vocab, dropout, emb_init, 
                    hid_init, dec_init, glove_emb, is_cuda, rnn, bidirectional):
        super(ConvRNN, self).__init__()

        self.d_hid = d_hid
        self.d_lin = d_lin
        self.is_cuda = is_cuda
        self.dir = 1
        if bidirectional:
            self.dir = 2
        self.rnn_type = rnn

        self.embedding = nn.Embedding(vocab, d_emb)
        if rnn == 'gru':
            self.rnn = BiGRU(d_emb, d_hid, dropout, bidirectional)
        elif rnn == 'lstm':
            self.rnn = BiLSTM(d_emb, d_hid, dropout, bidirectional)

        self.linear = nn.Linear(self.dir * d_hid + d_emb, d_lin)
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
        if self.is_cuda:
            y1_i = y1_i.cuda()
            y2_i = y2_i.cuda()

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
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
        else:
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid)))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)))


class ConvRNNFeat(nn.Module):
    def __init__(self, d_in, d_hid, d_out, d_emb, d_lin, vocab, dropout, emb_init, 
                    hid_init, dec_init, glove_emb, is_cuda, rnn, bidirectional, nfeat):
        super(ConvRNNFeat, self).__init__()

        self.d_hid = d_hid
        self.d_lin = d_lin
        self.is_cuda = is_cuda
        self.dir = 1
        if bidirectional:
            self.dir = 2
        self.rnn_type = rnn

        self.embedding = nn.Embedding(vocab, d_emb)
        if rnn == 'gru':
            self.rnn = BiGRU(d_emb, d_hid, dropout, bidirectional)
        elif rnn == 'lstm':
            self.rnn = BiLSTM(d_emb, d_hid, dropout, bidirectional)

        self.linear = nn.Linear(self.dir * d_hid + d_emb, d_lin)
        self.mlp = MLP(nfeat + 2 * d_lin, 512, 256, d_out, dropout)

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


    def forward(self, X1, X2, feat):
        X1 = Variable(X1)
        X2 = Variable(X2)
        feat = Variable(feat)

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
        if self.is_cuda:
            y1_i = y1_i.cuda()
            y2_i = y2_i.cuda()

        for ind in range(X1_cat.size(1)):
            y1_i[:, ind, :] = F.tanh(self.linear(X1_cat[:, ind, :]))
            y2_i[:, ind, :] = F.tanh(self.linear(X2_cat[:, ind, :]))
            
        enc1 = torch.max(y1_i, 1)[0][:, 0, :]
        enc2 = torch.max(y2_i, 1)[0][:, 0, :]
        
        s_cat = torch.cat([feat, enc1, enc2], 1)
        if self.is_cuda:
            s_cat = s_cat.cuda()

        return self.mlp(s_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
        else:
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid)))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)))


class ConvRNNLSTMFeat(nn.Module):
    def __init__(self, d_in, d_hid, d_out, d_emb, d_lin, vocab, dropout, emb_init, 
                    hid_init, dec_init, glove_emb, is_cuda, rnn, bidirectional, nfeat):
        super(ConvRNNLSTMFeat, self).__init__()

        self.d_hid = d_hid
        self.d_lin = d_lin
        self.is_cuda = is_cuda
        self.dir = 1
        if bidirectional:
            self.dir = 2
        self.rnn_type = rnn

        self.embedding = nn.Embedding(vocab, d_emb)
        if rnn == 'gru':
            self.rnn = BiGRU(d_emb, d_hid, dropout, bidirectional)
        elif rnn == 'lstm':
            self.rnn = BiLSTM(d_emb, d_hid, dropout, bidirectional)

        self.linear = nn.Linear(self.dir * d_hid + d_emb, d_lin)
        # NFeatures + 2 Linear Layers + (Dir * NQuestions * Hidden)
        self.mlp = MLP2(nfeat + (2 * d_lin) + (self.dir * 2 * d_hid), 512, 512, 256, d_out, dropout)

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


    def forward(self, X1, X2, feat):
        X1 = Variable(X1)
        X2 = Variable(X2)
        feat = Variable(feat)

        #emb1 (batch, seq_len, embsize)
        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size(0))
        out1, hid1 = self.rnn(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size(0))
        out2, hid2 = self.rnn(emb2, h2)

        X1_cat = torch.cat([emb1, out1], 2)
        X2_cat = torch.cat([emb2, out2], 2)
        
        y1_i = Variable(torch.Tensor(X1_cat.size(0), X1_cat.size(1), self.d_lin))
        y2_i = Variable(torch.Tensor(X2_cat.size(0), X2_cat.size(1), self.d_lin))
        if self.is_cuda:
            y1_i = y1_i.cuda()
            y2_i = y2_i.cuda()

        for ind in range(X1_cat.size(1)):
            y1_i[:, ind, :] = F.tanh(self.linear(X1_cat[:, ind, :]))
            y2_i[:, ind, :] = F.tanh(self.linear(X2_cat[:, ind, :]))
            
        enc1 = torch.max(y1_i, 1)[0][:, 0, :]
        enc2 = torch.max(y2_i, 1)[0][:, 0, :]

        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid2.size()[0]), 2)[0]
        
        s_cat = torch.cat([feat, enc1, enc2, hid1, hid2], 1)
        if self.is_cuda:
            s_cat = s_cat.cuda()

        return self.mlp(s_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid).cuda()))
        else:
            if self.rnn_type == 'lstm':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)), 
                        Variable(torch.zeros(self.dir, batch_size, self.d_hid)))
            elif self.rnn_type == 'gru':
                return (Variable(torch.zeros(self.dir, batch_size, self.d_hid)))


