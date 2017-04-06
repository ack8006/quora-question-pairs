import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLP, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module("linear1", nn.Linear(d_in, d_out, bias=True))
        self.sequential.add_module("softmax", nn.LogSoftmax())

    def forward(self, X):
        h = self.sequential(X)
        return h


class RnnModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, out_size, rnn_layers, dropout):
        super(RnnModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hid_size, rnn_layers, dropout=dropout)
        self.mlp = MLP(hid_size, out_size)

    def init_hidden(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

