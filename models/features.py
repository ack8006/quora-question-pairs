'''Feature extraction algorithms.'''

import numpy as np
import torch
import logging

logger = logging.getLogger('features')


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


