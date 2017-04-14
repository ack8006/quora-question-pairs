import os
from collections import Counter, defaultdict
import torch
import random
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = defaultdict(int)
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class TacoText(object):
    def __init__(self, vocab_size=None, lower=False, unk_token='<**unk**>',
                 pad_token='<**pad**>', vocab_pipe=None):
        self.dictionary = Dictionary()
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lower = lower
        if vocab_pipe is None:
            self.vocab_pipe = self.pipeline
        else:
            self.vocab_pipe = vocab_pipe

    def pipeline(self, x):
        if self.lower:
            x = x.lower()
        return x

    def gen_vocab(self, data):
        #Flattens List of List
        word_counts = Counter([self.pipeline(w) for s in data for w in s])
        #print('Total Words: ', len(word_counts))
        if self.vocab_size:
            word_counts = dict(word_counts.most_common(self.vocab_size))
        self.dictionary.add_word(self.unk_token)
        self.dictionary.add_word(self.pad_token)
        for w in word_counts.keys():
            self.dictionary.add_word(w)

    def pad(self, data, pad):
        return [[w for w in s][:pad]+[self.pad_token]*(pad-len(s)) for s in data]

    def numericalize(self, data):
        return np.array([[self.dictionary.word2idx[w] for w in s] for s in data])

    def pad_numericalize(self, data, pad):
        return self.numericalize(self.pad(data, pad))

    def __len__(self):
        return len(self.dictionary)
