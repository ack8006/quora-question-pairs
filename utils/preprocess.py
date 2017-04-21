import numpy as np
import pandas as pd
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize



def load_data(data_path, corpus, d_in, train_split=0.90):
    '''
    Arguments:
    data_path - String - takes file path the train data dataframe
    d_in - INT - 
    '''
    print('Loading Data')
    train_data = pd.read_csv(data_path)
    #Shuffle order of training data
    train_data = train_data.reindex(np.random.permutation(train_data.index))
    val_data = train_data.iloc[int(len(train_data) * train_split):]
    train_data = train_data.iloc[:int(len(train_data) * train_split)]

    print('Cleaning and Tokenizing')
    q1, q2, y = clean_and_tokenize(train_data, corpus)
    q1_val, q2_val, y_val = clean_and_tokenize(val_data, corpus)

    corpus.gen_vocab(q1 + q2 + q2_val + q1_val)

    print('Padding and Shaping')
    X, y = pad_and_shape(corpus, q1, q2, y, len(train_data), d_in)
    X_val, y_val = pad_and_shape(corpus, q1_val, q2_val, y_val, len(val_data), d_in)

    return X, y, X_val, y_val


def clean_and_tokenize(data, corpus):
    q1 = list(data['question1'].map(str))
    q2 = list(data['question2'].map(str))
    y = list(data['is_duplicate'])
    q1 = [word_tokenize(x) for x in q1]
    q2 = [word_tokenize(x) for x in q2]

    print('Piping Data')
    print(q1[:5])
    q1 = corpus.pipe_data(q1)
    print(q1[:5])
    q2 = corpus.pipe_data(q2)

    return q1, q2, y


def pad_and_shape(corpus, q1, q2, y, num_samples, d_in):
    X = torch.Tensor(num_samples, 1, 2, d_in).long()
    X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, d_in)).long()
    X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, d_in)).long()
    y = torch.from_numpy(np.array(y)).long()
    return X, y