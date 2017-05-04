import argparse

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split



from sklearn.externals import joblib



# max_features = 300000
# min_df = 50
# ngram_range = (1,10)


def grid_search(d_train, d_valid, y_valid, params):
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    results = {}
    best_ll, best_key = 999, None
    for md in (5, 7, 9, 11):
        for ss in (0.4, 0.6, 0.8, 1.0):
            for eta in (0.01, 0.1, 0.2):
                for colsam in (0.6, 0.8, 1.0):
                    params['max_depth'] = md
                    params['subsample'] = ss
                    params['eta'] = eta
                    params['colsample_bytree'] = colsam
                    bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50)
                    key = (md, ss, eta, colsam)
                    ll = log_loss(y_valid, bst.predict(d_valid))
                    print(key, ll)
                    if ll < best_ll:
                        best_ll = ll
                        best_key = key
                    results[key] = (ll)
    print(best_key, best_ll)
    print('max_depth, subsample, eta, colsample_by_tree')
    for k, v in results.items():
        print(k, v)



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
    parser.add_argument('--minoccur', type=float, default=500,
                        help='number of features')
    parser.add_argument('--maxngram', type=int, default=5,
                        help='maximum number of ngrams')
    parser.add_argument('--ntype', type=str, default='char',
                        help='ngram type, char or word')
    parser.add_argument('--grid', action='store_true',
                        help='gridsearch')
    parser.add_argument('--debug', action='store_true',
                        help='debug')
    args = parser.parse_args()

    
    print('Loading Data')

    print('Extracting Features')
    train_data = pd.read_csv('../data/train.csv')
    train_data = train_data.fillna(' ')

    if args.debug:
        train_data = train_data.iloc[:100000]

    X_train, X_valid, y_train, y_valid = train_test_split(train_data.iloc[:,:-1], train_data['is_duplicate'], test_size=0.1, random_state=4242)
    word_extractor = fit_extractor(X_train, args.nfeat, args.minoccur, 0.999, 
                                (1, args.maxngram), args.ntype)
    print('Total Features: ', len(word_extractor.vocabulary_))

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

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 7
    params['subsample'] = 0.6
    params['base_score'] = 0.2

    if args.grid:
        grid_search(d_train, d_valid, y_valid, params)
    else:
        print(args.nfeat, args.minoccur, args.maxngram, args.ntype)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
        bst.save_model(args.save+'.mdl')
        test_data = pd.read_csv('../data/test.csv')
        test_data = test_data.fillna(' ')
        X_test = transform_data(word_extractor, test_data)
        d_test = xgb.DMatrix(X_test)
        p_test = bst.predict(d_test)
        sub = pd.DataFrame()
        sub['test_id'] = test_data['test_id']
        sub['is_duplicate'] = p_test
        sub = sub.set_index('test_id')
        sub.to_csv('../predictions/'+args.save+'.csv')


if __name__ == '__main__':
    main()










