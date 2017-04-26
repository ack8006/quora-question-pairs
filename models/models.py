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
    def __init__(self, d_in, d_hid1, d_hid2, d_out, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(d_in)
        self.linear1 = nn.Linear(d_in, d_hid1, bias=True)
        self.bn2 = nn.BatchNorm1d(d_hid1)
        self.linear2 = nn.Linear(d_hid1, d_hid2, bias=True)
        self.bn3 = nn.BatchNorm1d(d_hid2)
        self.linear3 = nn.Linear(d_hid2, d_out, bias=True)


    def forward(self, X):
        X = self.bn1(X)
        X = self.linear1(X)
        X = F.dropout(F.leaky_relu(X, negative_slope=1 / 5.5), p=self.dropout)
        X = self.bn2(X)
        X = self.linear2(X)
        X = F.dropout(F.leaky_relu(X, negative_slope=1 / 5.5), p=self.dropout)
        X = self.bn3(X)
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


class LSTMModelMLP(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, d_out, d_emb, vocab, dropout,
                    emb_init, hid_init, dec_init, glove_emb, freeze_emb, is_cuda):
        super(LSTMModelMLP, self).__init__()

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
        self.mlp = MLP(d_hid * 2 * 2 * n_layers, 512, 256, d_out, dropout)

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
        self.mlp.init_weights(dec_init)


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
        return self.mlp(h_cat)


    def init_hidden(self, batch_size):
        if self.is_cuda:
            return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()), 
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.d_hid).cuda()))
        else:
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
            dropout=0.0, embed_size=20, squash_size=50, glove=None, cuda=False,
            word_dropout=0.0, extra_noise=0.0):
        '''Args:
            word_embedding: nn.Embedding - Word IDs to embeddings
            bilstm_encoder: BiLSTM - Sequence to hidden state
            bilstm_decoder: BiLSTM - Hidden state to sequence of hidden states
            dropout: Float value that controls dropout aggressiveness.
            embed_size: Embedded vector size that gets sent to the decoder.
            squash_size: Dimensionality of vector distance calculation.
            glove: Tensor containing GloVE vectors for init. Can be None.
        Dimensions must agree with each other.
        '''
        super(EmbeddingAutoencoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embedding = word_embedding
        self.word_dropout = word_dropout
        self.extra_noise = extra_noise
        self.bilstm_encoder = bilstm_encoder
        self.bilstm_decoder = bilstm_decoder

        # Between encoder and decoder, do VAE.
        encoder_dim = self.bilstm_encoder.lstm.hidden_size \
                * 2 * self.bilstm_decoder.lstm.num_layers
        self.fc_mean = FC(encoder_dim, embed_size, 0.0)
        self.fc_logvar = FC(encoder_dim, embed_size, 0.0)
        self.fc_expand = FC(embed_size, encoder_dim, dropout)
        self.fc_squash = FC(embed_size, squash_size, dropout)

        self.fc_decoder = FC(self.bilstm_decoder.lstm.hidden_size * 2,
                self.word_embedding.num_embeddings, dropout)
        self.batchnorm = nn.BatchNorm1d(
                2 * self.bilstm_encoder.lstm.num_layers *
                self.bilstm_encoder.lstm.hidden_size)
        self.bn_expand = nn.BatchNorm1d(
                2 * self.bilstm_encoder.lstm.num_layers *
                self.bilstm_encoder.lstm.hidden_size)
        self.bn_squash = nn.BatchNorm1d(squash_size)
        self.batchnorm_decode = nn.BatchNorm1d(
                2 * self.bilstm_decoder.lstm.hidden_size)

        self.is_cuda = False
        if cuda:
            self.is_cuda = True
            self.cuda()

        assert self.word_embedding.embedding_dim == \
            self.bilstm_encoder.lstm.input_size
        assert self.word_embedding.embedding_dim == \
            self.bilstm_decoder.lstm.input_size
        assert self.bilstm_encoder.lstm.num_layers == \
            self.bilstm_decoder.lstm.num_layers
        assert self.bilstm_encoder.lstm.hidden_size == \
            self.bilstm_decoder.lstm.hidden_size
        assert self.bilstm_encoder.lstm.hidden_size * 2 == \
            self.fc_decoder.fc.in_features # Last layer only
        assert self.fc_decoder.fc.out_features == \
            self.word_embedding.num_embeddings
        assert self.bilstm_encoder.lstm.batch_first
        assert self.bilstm_decoder.lstm.batch_first

    def encoder(self, X):
        '''Run X through the encoder part. Args:
            X: B x Seq_Len x D word embeddings
        Returns:
            hn: B x 2NH encoded sentence for each batch.'''
        h0 = self.init_hidden(X.size(0), self.bilstm_encoder.lstm)
        _, hid, _ = self.bilstm_encoder(X, h0) # Want only the hidden states.

        # Rotate hid so batchnorm works. hn = 2n x b x h
        hn = hid.transpose(0, 1).contiguous().\
            view(hid.size(1), hid.size(0) * hid.size(2)) # b x 2nh
        hn = self.batchnorm(hn)
        return hn

    def decoder(self, expand, X):
        '''Run X through the decoder part. Args:
            h0: 2N x B x H hidden states from encoder.
            X: B x Seq_Len x D word embeddings of the correct answer.
        Returns:
            output: B x Seq_Len X D output class probabilities.'''
        # B x 2ND -> 2N x B x H
        n = self.bilstm_decoder.lstm.num_layers
        h = self.bilstm_decoder.lstm.hidden_size
        h1_noised = expand.view(-1, 2 * n, h).transpose(1, 0)

        # B x S x 2D
        h0 = self.init_hidden(X.size(0), self.bilstm_decoder.lstm)
        h0 = (h1_noised.contiguous(), h0[1])
        out, _, _ = self.bilstm_decoder(X, h0)

        # S (B x 2D)
        words = [w.squeeze() for w in torch.chunk(out, out.size(1), dim=1)]
        # S (B x 2D)
        words = [self.fc_decoder(self.batchnorm_decode(w)) for w in words]
        #words = [self.fc_decoder(w) for w in words]
        # B x S x 2D
        return torch.stack(words, dim=1)

    def pair_log_probabilities(self, emb):
        '''Args:
            emb1: B x H embeddings
        Returns:
            log_probs: B x B matrix. Cell (i, j) is (unnormalized) log p(j|i)
                  according to some likelihood function on pairs of data points.'''
        B = emb.size(0)
        log_probs = Variable(torch.FloatTensor(B, B))
        if emb.is_cuda:
            log_probs = log_probs.cuda()
        for row in range(B):
            diff = emb - emb[row].unsqueeze(0).repeat(B, 1) # B x H
            dist = (diff * diff).sum(dim=1) # B x 1
            log_probs[row] = -dist
        #print(log_probs[3])
        return log_probs

    def forward(self, X1, noise=None, calculate_dist=True):
        '''Args:
            X1: B x Seq_Len word tokens
            noise: noise generator to use in training
            Xs: Supplemental data for autoencoder
        Returns:
            auto_X1: autoencoded X1 (B x Seq_Len x W) log-probabilities
            prob: pairwise probabilities between X1 and X2 FloatTensor Variable
                  of size B x B'''
        Xin = X1
        X1 = self.drop(self.word_embedding(X1)) # B x S x D
        hn = self.encoder(X1) # B x 2NH

        # Compute variational sample
        mean = self.fc_mean(hn)
        logvar = self.fc_logvar(hn)
        emb1 = noise(mean.size()) * (logvar / 2).exp() + mean
        if self.extra_noise > 0:
            emb1.add_(noise(emb1.size()) * self.extra_noise)
        expand = self.bn_expand(self.fc_expand(emb1))

        X1d = X1
        if self.word_dropout > 0:
            # Dropout entire words during generation.
            mask = torch.Tensor(Xin.size()).uniform_(0, 1).unsqueeze(2)
            mask = mask.gt(self.word_dropout).float()
            if self.is_cuda:
                mask = mask.cuda()
            # Zero out the entire word.
            X1d = X1.mul(Variable(mask).repeat(1, 1, X1.size(2)))
        auto_X1 = self.decoder(expand, X1d)
        prob = None
        if calculate_dist:
            squash = self.bn_squash(self.fc_squash(emb1))
            prob = self.pair_log_probabilities(squash)
            
        return auto_X1, mean, logvar, prob

    def init_hidden(self, batch_size, lstm):
        layer_dir = lstm.num_layers * 2
        d_hid = lstm.hidden_size
        tup = (Variable(torch.zeros(layer_dir, batch_size, d_hid)), 
                Variable(torch.zeros(layer_dir, batch_size, d_hid)))
        if self.is_cuda:
            tup = (tup[0].cuda(), tup[1].cuda())
        return tup


class AutoencoderClassifier(nn.Module):
    '''Uses a pre-trained EmbeddingAutoencoder to classify duplicates.
    
    Has 3 modes:
        - distance: Encode sentences, decide on distance
        - lengths: decide on 3 signals, sentence lengths + distance
        - projections: lengths + projection lengths'''

    def __init__(self, autoencoder, mode='distance', projection_dim=3,
            n_projections=10, use_mlp=False, dropout=0.0):
        super(AutoencoderClassifier, self).__init__()

        self.mode = mode
        self.autoencoder = autoencoder
        # Maybe it's a good idea to lock the projection vectors at length 1.
        self.projection_dim = projection_dim
        self.n_projections = n_projections
        self.use_mlp = use_mlp

        d_emb = self.autoencoder.fc_squash.fc.out_features
        self.proj = nn.Linear(d_emb, projection_dim * n_projections, bias=False)

        d_all = 3 * d_emb + 2 * n_projections
        self.fc = nn.Linear(d_all, 1)
        self.mlp = MLP(d_all, 512, 256, 1, dropout)

    def get_embedding(self, X):
        X = self.autoencoder.drop(self.autoencoder.word_embedding(X)) # B x S x D
        hn = self.autoencoder.encoder(X) # B x 2NH
        mean = self.autoencoder.fc_mean(hn)
        squash = self.autoencoder.bn_squash(self.autoencoder.fc_squash(mean))
        return squash.detach()

    def projection(self, mu):
        cube = (mu.size(0), self.n_projections, self.projection_dim)
        proj1 = self.proj(mu1).view(cube) # BxPxD
        return proj1.pow(2).sum(dim=2) # BxP

    def forward(self, X1, X2):
        mu1 = self.get_embedding(X1) # BxE
        mu2 = self.get_embedding(X2) # BxE
        dist = (mu1 - mu2).pow(2) # BxE

        len1 = mu1.pow(2) # ||mu||^2 # BxE
        len2 = mu2.pow(2) # ||mu||^2 # BxE

        if self.mode == 'distance':
            # Do not use distance in decision.
            len1 = len1.mul(0)
            len2 = len1

        if self.mode == 'projections':
            proj1 = self.projection(mu1)
            proj2 = self.projection(mu2)
            projs = torch.cat([proj1, proj2], dim=1) # Bx2P
        else:
            projs = torch.zeros((X1.size(0), 2 * self.n_projections))

        all_features = torch.cat([pdist, len1, len2, projs], dim=1)
        if self.use_mlp:
            res = self.mlp(all_features)
        else:
            res = self.fc(all_features)
        return torch.sigmoid(res) # Between 0 and 1



