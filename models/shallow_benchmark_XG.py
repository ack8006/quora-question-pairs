import argparse

import numpy as np
import pickle as pkl

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib



# max_features = 300000
# min_df = 50
# ngram_range = (1,10)


def run_circut(X_train, y_train, save, feat):
    parameters = {'n_estimators': (250,), #(100, 500)
                    #'subsample'L (0.2, 0.4, 0.6, 0.8, 1.0),
                    'max_depth': (3,), #(3, 4, 5, 6)
                    'max_delta_step': (1,),
                    'base_score': (0.5,), #'base_score': (0.5, 0.65),
                    }
    xb = XGBClassifier()
    gs = GridSearchCV(xb, parameters, scoring='neg_log_loss', cv=2) #cv=5)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('Params: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, '{}{}.pkl'.format(save, feat))


def main():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--save', type=str, default='xgb_',
                        help='location of the data corpus')
    args = parser.parse_args()

    print('Loading Data')

    print('Extracting Features')
    print(300000, 50, 0.999, (1, 10))

    with open('../data/train_X_300k.pkl', 'rb') as f:
        X_train = pkl.load(f)
    with open('../data/train_y_300k.pkl', 'rb') as f:
        y_train = pkl.load(f)

    run_circut(X_train, y_train, args.save, 300000)


if __name__ == '__main__':
    main()










