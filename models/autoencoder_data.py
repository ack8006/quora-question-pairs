'''Loads data for training the autoencoder.'''

import pandas as pd
import os
import spacy
import json
import data
import numpy as np
import torch
import itertools

nlp = spacy.load('en', parser=False)

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

def clean_and_tokenize(args, data, dictionary):
    def to_indices(words):
        '''From word to word index.'''
        ql = [dictionary.get(str(w).lower(), dictionary['<unk>']) for w in words]
        qv = np.ones(args.din, dtype=int) * dictionary['<pad>'] # all padding
        qv[:len(ql)] = ql[:args.din] # set values
        return qv

    qs = []
    processed = 0
    qids = [example.qid for example in data.itertuples()]
    tokens = (example.question for example in data.itertuples())
    nlps = nlp.pipe(tokens, parse=False, n_threads=16, batch_size=10000)
    for qid, tokens in itertools.izip(qids, nlps):
        if processed % 10000 == 0:
            print('processed {0}'.format(processed))
        qs.append(to_indices(tokens))
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
    cl1 = [c for c in clusters if len(c) > 0]
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
                    load_data(args, f('supplemental.csv'), self.glove, args.max_supplement)
        else:
            self.qid_supplement, self.questions_supplement = None, None

        # Get clusters for train and valid
        self.train_clusters = filter_clusters(self.qid_train, json.load(
                open(f('train_clusters.json')))['clusters'])
        self.valid_clusters = filter_clusters(self.qid_valid, json.load(
                open(f('valid_clusters.json')))['clusters'])



