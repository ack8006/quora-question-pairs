import argparse

import numpy as np
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib



# max_features = 300000
# min_df = 50
# ngram_range = (1,10)


def run_circut(X_train, y_train, save, feat):
    parameters = {'n_estimators': (200,)} #(100, 500)}
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, parameters, scoring='neg_log_loss', cv=2) #cv=5)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    for c, t_v, v_v in zip(results['params'], results['mean_train_score'], results['mean_test_score']):
        print('Params: ', c, ' MeanTrainScore', t_v, ' MeanTestScore: ', v_v)
    joblib.dump(gs.best_estimator_, '{}{}.pkl'.format(save, feat))


def main():
    parser = argparse.ArgumentParser(description='RandomForest ')
    parser.add_argument('--save', type=str, default='rf_',
                        help='location of the data corpus')
    args = parser.parse_args()

    print('Loading Data')
    print(300000, 50, 0.999, (1, 10))

    with open('../data/train_X_300k.pkl', 'rb') as f:
        X_train = pkl.load(f)
    with open('../data/train_y_300k.pkl', 'rb') as f:
        y_train = pkl.load(f)

    run_circut(X_train, y_train, args.save, 300000)


if __name__ == '__main__':
    main()


