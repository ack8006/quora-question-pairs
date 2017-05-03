import argparse
import functools

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True) #1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True) #4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    f = functools.partial(char_diff_unique_stop, stops=stops) 
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

#     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16
    
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17    

    return X


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--debug', action='store_true',
                        help='uses small dfs')
    parser.add_argument('--grid', action='store_true',
                        help='gridsearch')
    parser.add_argument('--downsample', type=float, default=0.0,
                            help='initial learning rate')
    parser.add_argument('--upsample', type=float, default=0.0,
                            help='initial learning rate')
    args = parser.parse_args()

    df_train = pd.read_csv('../data/train_features.csv', encoding="ISO-8859-1")
    x_train_ab = df_train.iloc[:, 2:-1]
    x_train_ab = x_train_ab.drop('euclidean_distance', axis=1)
    x_train_ab = x_train_ab.drop('jaccard_distance', axis=1)

    df_train = pd.read_csv('../data/train.csv')
    df_train = df_train.fillna(' ')

    if args.debug:
        x_train_ab = x_train_ab.iloc[:50000]
        df_train = df_train.iloc[:50000]

    # explore
    stops = set(stopwords.words("english"))

    df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
    df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    print('Building Features')
    x_train = build_features(df_train, stops, weights)
    x_train = pd.concat((x_train, x_train_ab), axis=1)
    y_train = df_train['is_duplicate'].values

    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    if args.downsample:
        print('Downsampling')
        assert args.downsample > 0.0 and args.downsample < .37
        p = args.downsample
        pl = len(pos_train)
        tl = len(pos_train) + len(neg_train)
        val = int(pl - (pl - p * tl)/((1 - p)))
        pos_train = pos_train.iloc[:int(val)]

    if args.upsample:
        p = args.upsample
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = pd.concat([neg_train, neg_train])
            scale -=1
        neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
        
    print(len(pos_train) / (len(pos_train) + len(neg_train)))

    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4242)

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 5
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    params['scale_pos_weight'] = 0.2

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    results = {}
    if args.grid:
        best_ll, best_key = 999, None
        for md in (5,6,7,8,9):
            for ss in (0.4, 0.6, 0.8, 1.0):
                for eta in (0.01, 0.1, 0.2):
                    for colsam in (0.6, 0.8, 1.0):
                        params['max_depth'] = md
                        params['subsample'] = ss
                        params['eta'] = eta
                        params['colsample_bytree'] = colsam
                        bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50)
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
    else:
        bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
        print(log_loss(y_valid, bst.predict(d_valid)))
        bst.save_model('XGB_handcrafted_5_06.mdl')


    if not args.debug:
        df_test = pd.read_csv('../data/test_features.csv', encoding = "ISO-8859-1")
        x_test_ab = df_test.iloc[:, 2:-1]
        x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
        x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)
        
        df_test = pd.read_csv('../data/test.csv')
        df_test = df_test.fillna(' ')

        df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
        df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())
        
        x_test = build_features(df_test, stops, weights)
        x_test = pd.concat((x_test, x_test_ab), axis=1)
        d_test = xgb.DMatrix(x_test)
        p_test = bst.predict(d_test)
        sub = pd.DataFrame()
        sub['test_id'] = df_test['test_id']
        sub['is_duplicate'] = p_test
        sub.to_csv('../predictions/4_xgb.csv')

if __name__ == '__main__':
    main()












