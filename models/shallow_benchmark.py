import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.externals import joblib



# max_features = 300000
# min_df = 50
# ngram_range = (1,10)

def fit_extractor(data, max_n_features, min_df, max_df, ngram_range):
    char_extractor = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_n_features, 
                                      analyzer='char', ngram_range=ngram_range, 
                                      binary=True, lowercase=True)
    char_extractor.fit(pd.concat((data.loc[:,'question1'], data.loc[:,'question2'])).unique().astype('U'))
    return char_extractor

def transform_data(extractor, data, test=False):
    q1 = extractor.transform(data.loc[:, 'question1'])
    q2 = extractor.transform(data.loc[:, 'question2'])
    X = -(q1 != q2).astype(int)
    y=None
    if not test:
        y = np.array(data.loc[:,'is_duplicate'])
    return X, y

def evaluate(y_train, y_val, train_pred, valid_pred):
    print('Train ', accuracy_score(y_train, train_pred > 0.5), log_loss(y_train, train_pred))
    print('Val ', accuracy_score(y_val, valid_pred > 0.5), log_loss(y_val, valid_pred))


def run_circut(X_train, y_train, feat):
    parameters = {'C': [0.01, 0.03, 0.05, 0.1, 0.5]}
    lr = LogisticRegression()
    gs = GridSearchCV(lr, parameters, scoring='neg_log_loss', cv=5)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('C: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, 'logistic_reg_{}.pkl'.format(feat))


    print('*' * 89)
    print('AdaBoost')
    parameters = {'n_estimators': (100, 500, 1000),
                    'learning_rate': (0.5, 1.0)}
    ab = AdaBoostClassifier()
    gs = GridSearchCV(ab, parameters, scoring='neg_log_loss', cv=5)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('N_estimators: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, 'adaboost_{}.pkl'.format(feat))


    print('*' * 89)
    print('Random Forest')
    parameters = {'n_estimators': (100, 500, 1000)}
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, parameters, scoring='neg_log_loss', cv=5)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('N_estimators: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, 'random_forest_{}.pkl'.format(feat))


    print('*' * 89)
    print('Gradient Boosting')
    parameters = {'n_estimators': (50, 100, 500, 1000)}
    rf = GradientBoostingClassifier()
    gs = GridSearchCV(rf, parameters, scoring='neg_log_loss', cv=5)
    gs.fit(X_train.toarray(), y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('N_estimators: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, 'gradient_boosting_{}.pkl'.format(feat))


def main():
    print('Loading Data')

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_data = train_data.fillna(' ')
    test_data = test_data.fillna(' ')
    # train_data = train_data.dropna(how="any").reset_index(drop=True)
    # test_data = test_data.dropna(how="any").reset_index(drop=True)

    # train_data = train_data.iloc[:100]
    # test_data = test_data.iloc[:100]

    print('Extracting Features')
    print(300000, 50, 0.999, (1,10))
    char_extractor = fit_extractor(train_data, 300000, 50, 0.999, (1,10))

    print('Transforming Data')
    X_train, y_train = transform_data(char_extractor, train_data)
    X_test, y_test = transform_data(char_extractor, test_data, test=True)
    with open('../data/train_X_300k.pkl', 'wb') as f:
        pkl.dump(X_train, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('../data/test_X_300k.pkl', 'wb') as f:
        pkl.dump(X_test, f, protocol=pkl.HIGHEST_PROTOCOL)
    del train_data, test_data, X_test, y_test

    run_circut(X_train, y_train, 300000)

    print('*' * 88)

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_data = train_data.fillna(' ')
    test_data = test_data.fillna(' ')

    print('Extracting Features')
    print(100000, 50, 0.999, (1,5))
    char_extractor = fit_extractor(train_data, 100000, 50, 0.999, (1,5))

    print('Transforming Data')
    X_train, y_train = transform_data(char_extractor, train_data)
    X_test, y_test = transform_data(char_extractor, test_data, test=True)
    with open('../data/train_X_100k.pkl', 'wb') as f:
        pkl.dump(X_train, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('../data/test_X_100k.pkl', 'wb') as f:
        pkl.dump(X_test, f, protocol=pkl.HIGHEST_PROTOCOL)
    del train_data, test_data, X_test, y_test

    run_circut(X_train, y_train, 100000)
    

if __name__ == '__main__':
    main()










