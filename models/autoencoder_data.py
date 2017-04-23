'''Loads data for training the autoencoder.'''

import pandas as pd
import os
import spacy
import json
import data
import numpy as np
import torch
import itertools
import re
from functools import partial
from multiprocessing import Pool

nlp = spacy.load('en', parser=False)
strip_padding = re.compile('(?: <pad>)+ *$')

def load_data(args, path, glove, limit=1000000):
    '''Load a csv file containing (qid, question) rows'''
    print('Loading Data')
    data = pd.read_csv(path, encoding='utf-8')
    data.columns = ['qid', 'question']
    data = data[:limit]
    data['question'] = data['question'].astype(unicode)

    print('Cleaning and Tokenizing')
    qid, q = clean_and_tokenize(args, data, glove.dictionary)

    return qid, q

def to_indices_base(dictionary, din, words):
    '''From word to word index.'''
    ql = [dictionary.get(str(w).lower(), dictionary['<unk>']) for w in words]
    qv = np.ones(din, dtype=int) * dictionary['<pad>'] # all padding
    qv[:len(ql)] = ql[:din] # set values
    return qv

def clean_and_tokenize(args, data, dictionary):
    to_indices = partial(to_indices_base, dictionary, args.din)

    qs = []
    processed = 0
    qids = [example.qid for example in data.itertuples()]
    tokens = [example.question for example in data.itertuples()]
    nlps = list(nlp.pipe(tokens, parse=False, n_threads=4, batch_size=10000))
    print(list(nlps[0]))
    to_ind_pool = Pool(4)
    for qid, parsed in itertools.izip(qids, nlps):
        if processed % 20000 == 0:
            print('processed {0}'.format(processed))
        ind = to_indices(parsed)
        qs.append(ind)
        processed += 1
    assert len(qids) == len(data)
    assert len(qs) == len(data)
    # Questions tensor
    qst = torch.LongTensor(np.stack(qs, axis=0))
    # Question IDs tensor
    qidst = torch.LongTensor(qids)
    return qidst, qst

def filter_clusters(qids, clusters):
    qids = set(qids)
    cl0 = [[x for x in c if x in qids] for c in clusters]
    cl1 = [c for c in cl0 if len(c) > 1]
    return cl1

class LoadedGlove:
    def __init__(self, glove):
        self.dictionary = glove[0]
        self.lookup = glove[1]
        self.module = glove[2]

class Data:
    '''Dataset to load, and the dictionary used to read them.'''
    def __init__(self, args):
        f = lambda fname: os.path.join(args.datadir, fname)

        print('loading Glove')
        assert args.demb in (50, 100, 200, 300)
        self.glove = LoadedGlove(data.load_embeddings(
                f('glove.6B.{0}d.txt'.format(args.demb)),
                max_words=args.vocabsize))

        print('fetching train')
        self.qid_train, self.questions_train = \
                load_data(args, f('questions_train.csv'), self.glove, args.max_sentences)
        print('fetching valid')
        self.qid_valid, self.questions_valid = \
                load_data(args, f('questions_valid.csv'), self.glove, args.max_sentences)
        if args.supplement is not None:
            print('fetching unsupervised')
            _, self.questions_supplement = \
                    load_data(args, f(args.supplement), self.glove, args.max_supplement)
        else:
            self.qid_supplement, self.questions_supplement = None, None

        # Get clusters for train and valid
        self.train_clusters = filter_clusters(self.qid_train, json.load(
                open(f('train_clusters.json')))['clusters'])
        print('{} training clusters'.format(len(self.train_clusters)))
        self.valid_clusters = filter_clusters(self.qid_valid, json.load(
                open(f('valid_clusters.json')))['clusters'])
        print('{} validation clusters'.format(len(self.valid_clusters)))

    def to_str(self, ids):
        '''ids: LongTensor'''
        # Convert to string and strip trailing padding.
        return strip_padding.sub('',
                ' '.join(self.glove.lookup[w] for w in list(ids)))

    def sample_str(self, log_probs):
        # Greedy decoding
        chosen = log_probs.max(dim=1)[1].squeeze()
        # print(chosen)
        # print(chosen.size())
        return self.to_str(chosen)




