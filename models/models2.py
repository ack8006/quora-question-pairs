import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

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
    def __init__(self, d_in, d_hid1, d_hid2, d_out, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        # self.bn1 = nn.BatchNorm1d(d_in)
        self.linear1 = nn.Linear(d_in, d_hid1, bias=True)
        # self.bn2 = nn.BatchNorm1d(d_hid1)
        self.linear2 = nn.Linear(d_hid1, d_hid2, bias=True)
        # self.bn3 = nn.BatchNorm1d(d_hid2)
        self.linear3 = nn.Linear(d_hid2, d_out, bias=True)


    def forward(self, X):
        # X = self.bn1(X)
        X = self.linear1(X)
        X = F.dropout(F.leaky_relu(X, negative_slope=1 / 5.5), p=self.dropout)
        # X = self.bn2(X)
        X = self.linear2(X)
        X = F.dropout(F.leaky_relu(X, negative_slope=1 / 5.5), p=self.dropout)
        # X = self.bn3(X)
        X = self.linear3(X)
        return F.log_softmax(X)


    def init_weights(self, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-0.1, b=0.1),
                        'constant': functools.partial(init.constant, val=0.1),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        init_types[weight_init](self.linear1.weight)
        init_types[weight_init](self.linear2.weight)
        init_types[weight_init](self.linear3.weight)


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
        out, (hid, cell) = self.lstm(x, h)
        #out, hidden, cell
        return hid


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
                    emb_init, hid_init, dec_init, glove_emb, is_cuda):
        super(LSTMModel, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab, d_emb)
        self.bilstm = BiLSTM(d_emb, d_hid, n_layers, dropout)

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
            self.embedding.weight.data = glove_emb
        else:
            init_types[emb_init](self.embedding.weight)

        self.bilstm.init_weights(hid_init)
        self.fc.init_weights(dec_init)


    def forward(self, X1, X2):
        X1 = Variable(X1)
        X2 = Variable(X2)

        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size()[0])
        hid1 = self.bilstm(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size()[0])
        hid2= self.bilstm(emb2, h2)

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



class LSTMModelMLP(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, is_cuda):
        super(LSTMModelMLP, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab, d_emb)
        self.bilstm = BiLSTM(d_emb, d_hid, n_layers, dropout)

        #(d_hid * directions * questions * layers)
        self.mlp = MLP(d_hid * 2 * 2 * n_layers, 512, 256, d_out, dropout)

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

        self.bilstm.init_weights(hid_init)
        self.mlp.init_weights(dec_init)


    def forward(self, X1, X2):
        X1 = Variable(X1)
        X2 = Variable(X2)

        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size()[0])
        hid1 = self.bilstm(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size()[0])
        hid2= self.bilstm(emb2, h2)

        #Concatenates Hidden State Directions and Layers
        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid2.size()[0]), 2)[0]

        #Concatenates Question Hidden States Together
        h_cat = torch.cat([hid1, hid2], 1)
        return self.mlp(h_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)))


class LSTMModelMLPFeat(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, dfeat, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, is_cuda):
        super(LSTMModelMLPFeat, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab, d_emb)
        self.bilstm = BiLSTM(d_emb, d_hid, n_layers, dropout)

        #(handcrafted + d_hid * directions * questions * layers)
        self.mlp = MLP(dfeat + d_hid * 2 * 2 * n_layers, 512, 256, d_out, dropout)

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

        self.bilstm.init_weights(hid_init)
        self.mlp.init_weights(dec_init)


    def forward(self, X1, X2, feats):
        X1 = Variable(X1)
        X2 = Variable(X2)
        feats = Variable(feats)

        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size()[0])
        hid1 = self.bilstm(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size()[0])
        hid2 = self.bilstm(emb2, h2)

        #Concatenates Hidden State Directions and Layers
        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid2.size()[0]), 2)[0]

        #Concatenates Question Hidden States Together
        h_cat = torch.cat([feats, hid1, hid2], 1)
        return self.mlp(h_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)))


class LSTMModelMLPFeatDist(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, dfeat, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, is_cuda):
        super(LSTMModelMLPFeatDist, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers
        self.is_cuda = is_cuda

        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab, d_emb)
        self.bilstm = BiLSTM(d_emb, d_hid, n_layers, dropout)

        #(handcrafted + d_hid * directions * questions * layers)
        self.mlp = MLP(dfeat + 5 + d_hid * 2 * 2 * n_layers, 512, 256, d_out, dropout)

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

        self.bilstm.init_weights(hid_init)
        self.mlp.init_weights(dec_init)


    def forward(self, X1, X2, feats):
        X1 = Variable(X1)
        X2 = Variable(X2)
        feats = Variable(feats)

        emb1 = self.embedding(X1)
        h1 = self.init_hidden(X1.size()[0])
        hid1 = self.bilstm(emb1, h1)

        emb2 = self.embedding(X2)
        h2 = self.init_hidden(X2.size()[0])
        hid2 = self.bilstm(emb2, h2)

        #Concatenates Hidden State Directions and Layers
        hid1 = torch.cat(torch.chunk(hid1, hid1.size()[0]), 2)[0]
        hid2 = torch.cat(torch.chunk(hid2, hid2.size()[0]), 2)[0]

        #Calculated Cos and Euclidian Distance Between Vectors
        dist = self.calculate_distances(hid1, hid2)

        #Concatenates Question Hidden States Together
        h_cat = torch.cat([feats, dist, hid1, hid2], 1)
        return self.mlp(h_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)))

    def calculate_distances(self, x1, x2):
        dim1 = x1.size(0)
        distances = torch.Tensor(dim1, 5).float()
        for d in range(dim1):
            distances[d, 0] = cosine(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy())
            distances[d, 1] = jaccard(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy())
            distances[d, 2] = canberra(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy())
            distances[d, 3] = minkowski(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy(), 3)
            distances[d, 4] = braycurtis(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy())
        if self.is_cuda:
            distances = distances.cuda()
        return Variable(distances)




