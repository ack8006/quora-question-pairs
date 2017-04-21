'''Loads data for training the autoencoder.'''

import pandas as pd
import os
import spacy
import json

nlp = spacy.load('en', parser=False)

def load_data(args, path, glove, limit=1000000):
    '''Load a csv file containing (qid, question) rows'''
    print('Loading Data')
    data = pd.read_csv(path, encoding='utf-8')
    data.columns = ['qid', 'question']
    data = data[:limit]

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

    qids = []
    qs = []
    processed = 0
    for example in data.itertuples():
        if processed % 10000 == 0:
            print('processed {0}'.format(processed))
        tokens = nlp(example.question, parse=False)
        qids.append(example.qid)
        qs.append(to_indices(tokens))
        processed += 1
    # Questions tensor
    qst = torch.LongTensor(np.stack(qs, axis=0))
    # Question IDs tensor
    qidst = torch.LongTensor(qids)
    return qidst, qst

class LoadedGlove:
    def __init__(self, glove):
        self.dictionary = glove[0]
        self.lookup = glove[1]
        self.module = glove[2]

def load_glove(args):
    # Returns dictionary, lookup, embed
    print('loading Glove')
    glove = data.load_embeddings(
            '{1}/glove.6B.{0}d.txt'.format(args.demb, args.glovedata),
            max_words=args.vocabsize)
    return LoadedGlove(glove)

class Data:
    '''Dataset to load, and the dictionary used to read them.'''
    def __init__(self, args):
        f = lambda fname: os.path.join(args.datadir, f)

        self.glove = load_glove(args)
        self.qid_train, self.questions_train = \
                load_data(args, f('questions_train.csv'))
        self.qid_valid, self.questions_valid = \
                load_data(args, f('questions_valid.csv'))
        self.train_clusters = json.load(open(f('train_clusters.json')))
        self.valid_clusters = json.load(open(f('valid_clusters.json')))


