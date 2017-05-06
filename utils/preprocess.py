import re

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize



def load_data(data_path, corpus, d_in):
    '''
    Arguments:
    data_path - String - takes file path the train data dataframe
    d_in - INT - 
    '''
    print('Loading Data')
    train_data = pd.read_csv(data_path+'train_data_shuffle.csv')
    val_data = pd.read_csv(data_path+'val_data_shuffle.csv')
    train_data = train_data.fillna(' ')
    val_data = val_data.fillna(' ')
    #Shuffle order of training data

    # train_data = train_data.iloc[:1000]
    # val_data = val_data.iloc[:100]

    # train_data = train_data.reindex(np.random.permutation(train_data.index))
    # val_data = train_data.iloc[int(len(train_data) * train_split):]
    # train_data = train_data.iloc[:int(len(train_data) * train_split)]

    print('Cleaning and Tokenizing')
    q1, q2, y = clean_and_tokenize(train_data, corpus)
    q1_val, q2_val, y_val = clean_and_tokenize(val_data, corpus)

    print('Piping Data')
    q1 = corpus.pipe_data(q1)
    q2 = corpus.pipe_data(q2)
    q1_val = corpus.pipe_data(q1_val)
    q2_val = corpus.pipe_data(q2_val)

    corpus.gen_vocab(q1 + q2 + q2_val + q1_val)

    print('Padding and Shaping')
    X, y = pad_and_shape(corpus, q1, q2, y, len(train_data), d_in)
    X_val, y_val = pad_and_shape(corpus, q1_val, q2_val, y_val, len(val_data), d_in)

    return X, y, X_val, y_val


def clean_and_tokenize(data, corpus):
    q1 = list(data['question1'].map(str))
    q2 = list(data['question2'].map(str))
    y = list(data['is_duplicate'])
    q1 = [x.lower().split() for x in q1]
    q2 = [x.lower().split() for x in q2]

    return q1, q2, y


def pad_and_shape(corpus, q1, q2, y, num_samples, d_in):
    X = torch.Tensor(num_samples, 1, 2, d_in).long()
    X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, d_in)).long()
    X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, d_in)).long()
    y = torch.from_numpy(np.array(y)).long()
    return X, y


def split_text(x, stops=None):
    x = x.lower()
    x = x.replace("'ve", " have")
    x = x.replace("can't", "cannot")
    x = x.replace("n't", " not ")
    x = x.replace("'re", " are ")
    x = re.sub(r"(\d+)(k)", r"\g<1>000", x)
    x = x.replace("i'm", "i am")
    x = x.replace("'d", " would")
    x = x.replace("'ll", " will")
    x = x.replace("'s", " ")
    x = x.replace("’s", " ")
    x = x.replace(",", "")
    x = x.replace(".", "")
    for punc in list("-?(){}:/';"):
        x = x.replace(punc, " ")
    x = x.replace('"', '')
    x = x.replace("&", " and ")
    for punc in list("^+='*"):
        x = x.replace(punc, " "+punc+" ")
    x = x.replace(" e g ", " eg ")
    x = x.replace(" u s ", " america ")
    x = x.replace(" usa ", " america ")
    x = x.replace(" uk ", " england ")
    x = x.replace(" eu ", " europe ")
    x = x.replace('imrovement', ' improvement ')
    x = x.replace('programing', ' programming ')
    x = x.replace(' iiit', ' india institute technology')
    x = x.replace(' iii', ' 3')
    x = x.replace('kms', ' kilometers ')
    x = x.replace('khmp', ' kilometers per hour ')
    x = x.replace('[math]', ' [math] ')
    x = x.replace('[/math]', ' [math] ')
    x = x.replace('dissapear', ' disappear')
    x = x.replace('dissapoint', 'disappoint')
    x = x.replace('embarassing', 'embarrass')
    x = x.replace('begining', 'beginning')
    x = x.replace('buisness', 'business')
    x = x.replace('calender', 'calendar')
    x = x.replace('concious', 'conscious')
    x = x.replace('enviroment', 'environment')
    x = x.replace('existance', 'existence')
    x = x.replace('goverment', 'government')
    x = x.replace('independant', 'independent')
    x = x.replace('recieve', 'receive')
    x = x.replace('succesful', 'successful')
    
    x = [w.strip() for w in x.split()]
    if stops:
        x = [w for w in x if w not in stops]
        
    return x
