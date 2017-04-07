import torch.nn as nn
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, activation="relu"):
        super(FeedForward, self).__init__()
        self.model = nn.Sequential()
        linear_1 = nn.Linear(d_in, d_out)
        # He. normal initialization, ref: http://arxiv.org/abs/1502.01852
        linear_1.weight.data.normal_(mean=0, std=np.sqrt(2.0 / d_in))
        self.model.add_module("linear_1", linear_1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("activation " + str(activation) + " not supported")
        self.model.add_module("non_linearity_1", self.activation)

    def forward(self, X):
        return self.model(X)


class Attend(nn.Module):
    def __init__(self, max_length, n_input, n_hidden, dropout=0.0, activation="relu"):
        super(Attend, self).__init__()
        self.max_length = max_length
        self.model = nn.Sequential()
        """
        TODO: check if you have to add code for doing a time-series,
        ref: https://github.com/fchollet/keras/issues/1029
        """
        self.model.add_module("dropout_1", nn.Dropout(p=dropout))
        self.model.add_module(
            "attention_1",
            FeedForward(d_in=n_input, d_out=n_hidden, activation=activation)
        )
        self.model.add_module("dropout_2", nn.Dropout(p=dropout))
        self.model.add_module(
            "attention_2",
            FeedForward(d_in=n_hidden, d_out=n_hidden, activation=activation)
        )

    def forward(self, a, b):
        a_v = a.contiguous().view(a.size(0) * a.size(1), a.size(2))
        b_v = b.contiguous().view(b.size(0) * b.size(1), b.size(2))
        f_a = self.model(a_v)
        f_b = self.model(b_v)
        f_a = f_a.contiguous().view(a.size(0), a.size(1), a.size(2))
        f_b = f_b.contiguous().view(b.size(0), b.size(1), b.size(2))
        f_b_t = f_b.transpose(1, 2)
        out = f_a.bmm(f_b_t)
        return out