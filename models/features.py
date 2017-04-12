'''Feature extraction algorithms.'''

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging

logger = logging.getLogger('features')


class Permute(nn.Module):
    def __init__(self, out_features, sentence_length=30,
            n_trials=2000, power=0.1, differentiable=False):
        super(Permute, self).__init__()
        self.n_trials = n_trials
        self.seq_len = sentence_length
        self.power = power
        self.differentiable = differentiable
        # Only used in differentiable mode. Correction term.
        self.bias = Parameter(torch.Tensor(out_features))

    def reset_parameters(self):
	self.bias.data.fill_(0)

    def forward(self, q1, q2):
        mean_q1 = q1.mean(dim=1)
        mean_q2 = q2.mean(dim=1)
        mean_dist = (mean_q1 - mean_q2).abs()

        q1q2 = torch.cat([q1, q2], 1) # B x W x D

        permutation = torch.FloatTensor(
            self.n_trials, 2 * self.seq_len).fill_(0) # N x W

        for trial in xrange(self.n_trials):
            y = torch.randperm(2 * self.seq_len)
            permutation[trial][y[:self.seq_len]] = 1.0 / self.seq_len
            permutation[trial][y[self.seq_len:]] = -1.0 / self.seq_len
        permutation = Variable(permutation).unsqueeze(0) # 1 x N x W
        prep = permutation.repeat(q1q2.size(0), 1, 1) # B x N X W
        trial_means = torch.bmm(prep, q1q2).abs() # B x N x D

        diffs = trial_means - mean_dist.repeat(1, self.n_trials, 1)
        if self.differentiable:
            return permutation, F.relu(diffs).pow(self.power).mean(dim=1).squeeze() + self.bias
        else:
            return permutation, (diffs > 0).float().mean(dim=1).squeeze()



def permute(dataset, embed_size=100, max_word_len=30, n_trials=2000, distance='squared'):
    '''Compute permutation test p-value features.

    Args:
        dataset: a list of triplets (q1, q2, y). q1/q2 are tensors backed
                 by the big q1_tensor and q2_tensors.

    Returns:
        features: a NxD tensor with the features.
    '''
    perm = torch.zeros((n_trials, 2 * max_word_len))
    invperm = torch.zeros((n_trials, 2 * max_word_len))
    hold = torch.zeros((2 * max_word_len, embed_size))
    out = torch.zeros((len(dataset), embed_size))

    def faster_distance(diff):
        if distance == 'squared':
            diff.pow_(2)
        else:
            diff.abs_()
   
    permute_q1_result = torch.zeros((n_trials, embed_size))
    base_dist_q1 = torch.zeros(embed_size)
    base_meaner_ = torch.zeros((1, 2 * max_word_len))

    for i, (q1, q2, y) in enumerate(dataset):
        if q1 is None or q2 is None:
            continue
        len_q1 = q1.size(0)
        len_q2 = q2.size(0)

        sum_len = len_q1 + len_q2
        perm.fill_(0)
        invperm.fill_(1)

        # Get views for the common matrices.
        base_meaner = base_meaner_[:,:sum_len]
        permutations = perm[:,:sum_len]
        words = hold[:sum_len,:]
        words[:len_q1] = q1
        base_meaner[0,:len_q1] = 1.0 / len_q1
        words[len_q1:] = q2
        base_meaner[0,len_q1:sum_len] = -1.0 / len_q2

        # Compare distances against this.
        torch.mm(base_meaner, words, out=base_dist_q1)
        faster_distance(base_dist_q1)

        for trial in xrange(n_trials):
            y = torch.randperm(sum_len)
            permutations[trial][y[:len_q1]] = 1.0 / len_q1
            permutations[trial][y[len_q1:]] = -1.0 / len_q2

        # Calc trials.
        torch.mm(permutations, words, out=permute_q1_result) # trials X embed
        faster_distance(permute_q1_result)
        out[i] = (permute_q1_result - base_dist_q1.repeat(n_trials, 1)).ceil().clamp(0, 1).mean(dim=0)

    return out


