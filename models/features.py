'''Feature extraction algorithms.'''

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging

logger = logging.getLogger('features')


class Permute(nn.Module):
    '''Computes permutation test p-values between two bags of words.'''

    def __init__(self, out_features=100, sentence_length=30,
            n_trials=2000, power=0.1, differentiable=False):
        super(Permute, self).__init__()
        self.n_trials = n_trials
        self.seq_len = sentence_length
        self.power = power
        self.differentiable = differentiable
        # Only used in differentiable mode. Correction term.
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def reset_parameters(self):
	self.bias.data.fill_(0)

    def forward(self, q1, q2):
        '''Args:
            q1: B x sentence_length x embed_size Tensor. Word embeddings.
            q2: B x sentence_length x embed_size Tensor. Word embeddings.'''
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
            approx = F.relu(diffs).pow(self.power).mean(dim=1).squeeze()
            biases = self.bias.unsqueeze(0).repeat(approx.size(0), 1)
            return permutation, approx + biases
        else:
            return permutation, (diffs > 0).float().mean(dim=1).squeeze()


def elementwise_mean_stdev(sentence, base_std=0.001):
    '''Gets the mean and variance of the sentence(s), viewed as a bag of words.
    Mean and variance is computed for each element independently.

    Args:
        sentence: Batch x sequence_length x Dim
        base_std: constant value added to stdev for numerical stability.

    Returns:
        mean: Variable (Batch x Dim)
        stdev: Variable (Batch x Dim)
    '''
    mean = sentence.mean(dim=1) # Batch x 1 x Dim
    diff = (sentence - mean.repeat(1, sentence.size(1), 1))
    stdev =diff.pow(2).mean(dim=1).sqrt() + base_std
    return mean.squeeze(), stdev.squeeze()


def elementwise_kl_div(p, q):
    '''Elementwise kl-div between the two sentences.
    
    Args:
        p: samples from the "true" distribution.
        q: samples from the approximation.'''

    mu1, sig1 = elementwise_mean_stdev(p)
    mu2, sig2 = elementwise_mean_stdev(q)

    return (sig2 / sig1).log() + \
            (sig1.pow(2) + (mu1 - mu2).pow(2)) / \
            (2 * sig2.pow(2)) - 0.5


def symmetric_kl_div(q1, q2):
    '''Calculates KLD between treating the sentences as different and as
    the same distributions.'''

    combine = torch.cat([q1, q2], 1)
    lq1, lq2 = q1.size(1), q2.size(1)
    len_all = 1.0 * (lq1 + lq2)
    klq1 = elementwise_kl_div(q1, combine)
    klq2 = elementwise_kl_div(q2, combine)
    return (lq1 / len_all) * klq1 + (lq2 / len_all) * klq2


def apply(q1, q2, module, batchsize=50, print_every=None):
    '''Compute permutation test p-value features in batches.

    Args:
        q1: Question 1 vectors
        q2: Question 2 vectors
        module: Module to apply
        bathsize: How many sentences to do each time

    Returns:
        diffs: a NxD tensor with the features.
    '''
    n_examples = q1.size(0)
    n_batches = n_examples / batchsize + 1

    n_return = q1.size(2)
    data = torch.zeros(n_examples, n_return)

    for batch in range(n_batches):
        if print_every and batch % print_every == 0:
            print(batch)
        start = batchsize * batch
        end = batchsize * (batch + 1)
        if start >= n_examples:
            break
        q1b = q1[start:end]
        q2b = q2[start:end]

        module_result = module(q1b, q2b)
        data[start:end] = module(q1b, q2b).data

    return Variable(data)


