import pickle as pkl

import pandas as pd
import torch

import sys
sys.path.append('../models/')
sys.path.append('../utils')
from models2 import LSTMModelMLP
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import word_tokenize


with open('../models/model_reweight1_mlp_corpus.pkl', 'rb') as f:
    corpus = pkl.load(f)

model = torch.load('model_reweight1_mlp')
model.cuda()

test_data = pd.read_csv('../data/test.csv', nrows=1000)
y = torch.LongTensor(len(test_data)).zero_()

q1 = list(test_data['question1'].map(str))
q2 = list(test_data['question2'].map(str))
q1 = [word_tokenize(x) for x in q1]
q2 = [word_tokenize(x) for x in q2]
q1 = corpus.pipe_data(q1)
q2 = corpus.pipe_data(q2)

X = torch.Tensor(len(test_data), 1, 2, 20).long()
X[:, 0, 0, :] = torch.from_numpy(corpus.pad_numericalize(q1, 20)).long()
X[:, 0, 1, :] = torch.from_numpy(corpus.pad_numericalize(q2, 20)).long()

X = X.cuda()
y = y.cuda()


test_dataset = TensorDataset(X, y)
test_loader = DataLoader(test_dataset, 
                            batch_size=100, 
                            shuffle=False)

pred_list = []
for ind, (qs, _) in enumerate(test_loader):
    out = model(qs[:, 0, 0, :], qs[:, 0, 1, :])
    pred_list += list(out.exp()[:, 1].data.cpu().numpy())


with open('../predictions/model_reweight_1.pkl', 'wb') as f:
	pkl.dump(f)








