import numpy as np
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from sklearn.externals import joblib


with open('../data/test_X_300k.pkl', 'rb') as f:
    X_test = pkl.load(f)

with open('logistic_reg_300000.pkl', 'rb') as f:
    model = joblib.load(f)


test_pred = model.predict_proba(X_test)
test_pred = test_pred[:, 1]

with open('logistic_reg_300k_pred.pkl', 'wb') as f:
    pkl.dump(test_pred, f)