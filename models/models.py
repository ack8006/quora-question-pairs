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
        # self.sequential = nn.Sequential()
        # self.sequential.add_module('linear1', nn.Linear(d_in, 
        #                                                 d_out, 
        #                                                 bias=True))
        # self.sequential.add_module('softmax', nn.LogSoftmax())
        self.dropout = dropout

        self.fc = nn.Linear(d_in, d_out, bias=True)


    def forward(self, X):
        X = F.dropout(X, p=self.dropout)
        X = self.fc(X)
        return F.log_softmax(X)

        # out = self.sequential(X)
        # return out


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


class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, freeze_emb):
        super(LSTMModel, self).__init__()

        self.d_hid = d_hid
        self.n_layers = n_layers

        self.drop = nn.Dropout(dropout)

        self.embedding1 = nn.Embedding(vocab, d_emb)
        self.embedding1.weight.requires_grad = freeze_emb
        self.bilstm_1 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        self.embedding2 = nn.Embedding(vocab, d_emb)
        self.embedding2.weight.requires_grad = freeze_emb
        self.bilstm_2 = BiLSTM(d_emb, d_hid, n_layers, dropout)

        #(d_hid * directions * questions * layers)
        self.fc = FC(d_hid * 2 * 2 * n_layers, d_out, dropout)
        # self.seq_3 = nn.Sequential()
        # self.seq_3.add_module('drop', nn.Dropout(dropout))
        # #d_hid * directions * questions * layers
        # self.seq_3.add_module('mlp', FC(d_hid * 2 * 2 * n_layers, d_out, dropout))

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

        # init_types[hid_init](self.bilstm_1.weight_ih_l)
        # init_types[hid_init](self.bilstm_1.weight_hh_l)
        # init_types[hid_init](self.bilstm_2.weight_ih_l)
        # init_types[hid_init](self.bilstm_2.weight_hh_l)

        # init_types[dec_init](self.fc.weight)


    def forward(self, X1, X2):
        # X1 = Variable(torch.from_numpy(X1.numpy()).long())
        # X2 = Variable(torch.from_numpy(X2.numpy()).long())
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
        # return self.seq_3(h_cat)


    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)), 
                Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid)))


class EmbeddingAutoencoder(nn.Module):
    '''The autoencoder does two things: try to recode the sentence, and embed
    the encoding in a space that makes sense.

    To do this, it has two loss functions:
        - One component is just the reconstruction loss of the autoencoder.
        - The other one makes sure that sentence duplicates are embedded
          closely together.
    '''
    def __init__(self, word_embedding, bilstm_encoder, bilstm_decoder,
            dropout=0.0, glove=None):
        '''Args:
            word_embedding: nn.Embedding - Word IDs to embeddings
            bilstm_encoder: BiLSTM - Sequence to hidden state
            bilstm_decoder: BiLSTM - Hidden state to sequence of hidden states
            dropout: Float value that controls dropout aggressiveness.
            glove: Tensor containing GloVE vectors for init. Can be None.
        Dimensions must agree with each other.
        '''
        super(EmbeddingAutoencoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embedding = word_embedding
        self.bilstm_encoder = bilstm_encoder
        self.bilstm_decoder = bilstm_decoder
        # Times 2 because of bidirectional.
        self.fc_decoder = FC(self.bilstm_decoder.lstm.hidden_size
                * 2 * self.bilstm_decoder.lstm.num_layers,
                self.word_embedding.embedding_dim, dropout)

        assert self.word_embedding.embedding_dim == \
            self.bilstm_encoder.lstm.input_size
        assert self.bilstm_encoder.lstm.num_layers == \
            self.bilstm_decoder.lstm.num_layers
        assert self.bilstm_encoder.lstm.hidden_size == \
            self.bilstm_decoder.lstm.input_size
        assert self.bilstm_encoder.lstm.hidden_size * \
            self.bilstm_decoder.lstm.num_layers * 2 == \
            self.fc_decoder.fc.in_features
        assert self.fc_decoder.fc.out_features == \
            self.word_embedding.num_embeddings
        assert self.bilstm_encoder.lstm.batch_first
        assert self.bilstm_decoder.lstm.batch_first

    def encoder(self, X):
        '''Run X through the encoder part. Args:
            X: B x Seq_Len x D word embeddings
        Returns:
            hid: 2N x B x D encoded sentence for each batch.
            flat: B x 2ND tensor same as hid, but flattened.'''
        h0 = self.init_hidden(X.size(0))
        _, hid, _ = self.bilstm_encoder(X, h0) # Want only the hidden states.
        flat = torch.cat(torch.chunk(hid, hid.size(0)), 2)[0]
        return hid, flat

    def decoder(self, h0, X):
        '''Run X through the decoder part. Args:
            h0: 2N x B x D hidden states from encoder.
            X: B x Seq_Len x D word embeddings of the correct answer.
        Returns:
            output: B x Seq_Len X D output class probabilities.'''
        # B x S x ND
        out, _, _ = self.bilstm_decoder(X, h0)
        # S (B x ND)
        words = [w.squeeze() for w in torch.chunk(hid, hid.size(0), dim=1)]
        # S (B x D)
        words = [self.fc_decoder(w) for w in words]
        # B x S x D
        return torch.stack(words, dim=1)

    def distances(self, emb):
        '''Args:
            emb1: B x H embeddings
        Returns:
            dist: pairwise distances in a 1D tensor, that comes from the
                  right triangular distance matrix.'''
        B = emb.size(0)
        # If B = 5, then dist will have 4+3+2+1 = 5*4/2 components
        # If B = 4, then dist will have 3+2+1 = 4*3/2 components
        d = B*(B-1)/2
        dists = Variable(torch.FloatTensor(d))
        current = 0
        for row in range(B-1): # For B=5, go through rows 0,1,2,3 (skip 4)
            start = row + 1 # For B=4,row1 (2nd): start from 2 (1,2),(1,3)
            num = B - start # How many distances to compare
            diff = emb[start:] - emb[row].unsqueeze(0).repeat(num)
            dist = (diff * diff).sum(dim=1)
            dists[current:current+num] = dist
            current += num
        return dists

    def forward(self, X1):
        '''Args:
            X1: B x Seq_Len word tokens
        Returns:
            auto_X1: autoencoded X1 (B x Seq_Len x W) log-probabilities
            dist: pairwise distances between X1 and X2 FloatTensor Variable
                  of size B x B'''

        X1 = self.drop(self.word_embedding(X1))
        h1, emb1 = self.encoder(X1)
        auto_X1 = self.decoder(h1, X1)
        dist = self.distances(emb1)
        return auto_X1, dist

        



