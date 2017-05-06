import argparse
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.externals import joblib



# max_features = 300000
# min_df = 50
# ngram_range = (1,10)


def fit_extractor(data, max_n_features, min_df, max_df, ngram_range, analyzer):
    char_extractor = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_n_features, 
                                      analyzer=analyzer, ngram_range=ngram_range, 
                                      binary=True, lowercase=True)
    char_extractor.fit(pd.concat((data.loc[:,'question1'], data.loc[:,'question2'])).unique().astype('U'))
    return char_extractor


def transform_data(extractor, data, test=False):
    q1 = extractor.transform(data.loc[:, 'question1'])
    q2 = extractor.transform(data.loc[:, 'question2'])
    X = -(q1 != q2).astype(int)
    return X


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--save', type=str, default='xgb_',
                        help='location of the data corpus')
    parser.add_argument('--nfeat', type=int, default=200000,
                        help='number of features')
    parser.add_argument('--minoccur', type=int, default=500,
                        help='number of features')
    parser.add_argument('--maxngram', type=int, default=5,
                        help='maximum number of ngrams')
    parser.add_argument('--ntype', type=str, default='char',
                        help='ngram type, char or word')
    parser.add_argument('--grid', action='store_true',
                        help='gridsearch')
    parser.add_argument('--debug', action='store_true',
                        help='debug')
    parser.add_argument('--reweight', action='store_true',
                        help='whether or not to reweight data')
    args = parser.parse_args()

    
    print('Loading Data')

    print('Extracting Features')
    train_data = pd.read_csv('../data/train_data_shuffle.csv')
    valid_data = pd.read_csv('../data/val_data_shuffle.csv')
    train_data = train_data.fillna(' ')
    valid_data = valid_data.fillna(' ')

    if args.debug:
        train_data = train_data.iloc[:100000]

    X_train, y_train = train_data.iloc[:,:-1], train_data['is_duplicate']
    X_valid, y_valid = valid_data.iloc[:,:-1], valid_data['is_duplicate']

    # X_train, X_valid, y_train, y_valid = train_test_split(train_data.iloc[:,:-1], train_data['is_duplicate'], test_size=0.1, random_state=4242)
    
    word_extractor = fit_extractor(X_train, args.nfeat, args.minoccur, 0.999, 
                                (1, args.maxngram), args.ntype)
    print('Total Features: ', len(word_extractor.vocabulary_))

    if args.reweight:
        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]

        X_train = pd.concat((neg_train, pos_train, neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.shape[0] + [0] * neg_train.shape[0])
        del pos_train, neg_train
        print("Mean target rate : ", y_train.mean())

        pos_valid = X_valid[y_valid == 1]
        neg_valid = X_valid[y_valid == 0]

        X_valid = pd.concat([neg_valid, pos_valid, neg_valid])
        y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.shape[0] + [0] * neg_valid.shape[0])
        del pos_valid, neg_valid
        print("Mean target rate : ", y_valid.mean())

    X_train = transform_data(word_extractor, X_train)
    X_valid = transform_data(word_extractor, X_valid)

    print('Running Logistic Regression')
    lr = LogisticRegression(C=0.03)
    lr.fit(X_train, y_train)
    train_pred = lr.predict_proba(X_train)
    val_pred = lr.predict_proba(X_valid)
    print('Train:', log_loss(y_train, train_pred))
    print('Val:', log_loss(y_valid, val_pred))
    with open('../predictions/'+args.save+'_lr.csv') as f:
        pkl.dump(val_pred, f)


if __name__ == '__main__':
    main()










