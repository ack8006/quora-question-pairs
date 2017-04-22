import json
import numpy as np
import torch
import itertools
import random

def iterate_epoch(clusters, args):
    '''Given a list of clusters, sample batches from them in a way that goes
    through all clusters, but still have plenty of same-cluster examples per
    batch.

    The way it works: for each batch, randomly pick a few (k) clusters, and
    take a few ids (m) from them at random. If the number of examples is still
    fewer than batch size (b), randomly take single examples from any cluster
    until the batch size is fulfilled.

    Typically k, m < b, but km > b. For example, k=m=10, and b=80.
    
    Args:
        clusters: a list of list of numbers that are clusters.
        args: parameters that can be tuned (see below)'''

    seed_max = args.seed_size # How many initial clusters
    take_max = args.take_clusters # How many duplicates from each cluster max
    batch_max = args.batchsize # Size of the batches
    id_max = args.max_sentences

    # Precompute which pairs are clusters.
    dups = []
    for c in clusters:
        for i in c:
            for j in c:
                dups.append((i, j))
                dups.append((j, i))
    dupset = set(dups)
    del dups

    np.random.shuffle(clusters)
    for idx in xrange(0, len(clusters), seed_max): # Seed size
        # Get a selection of duplicates as the seed.
        batch = []
        for cluster in clusters[:seed_max]:
            np.random.shuffle(cluster)
            batch.extend(cluster[:take_max])
            if len(batch) > batch_max:
                break

        # Shuffling prevents the first cluster from being unfairly
        # over-represented in case the batch needs to be cut.
        batch = [b for b in batch if b < id_max]
        np.random.shuffle(batch)
        batch = batch[:batch_max]

        # If we're short of examples, just take one from any cluster.
        check = {qid: i for i, qid in enumerate(batch)}
        while len(batch) < batch_max:
            cluster = random.choice(clusters)
            qid = random.choice(cluster)
            if qid not in check:
                check[qid] = len(batch)
                batch.append(qid)

        mtx = np.zeros((batch_max, batch_max), dtype=np.int32)
        for i, q1 in enumerate(batch):
            for j, q2 in enumerate(batch):
                mtx[i,j] = (q1, q2) in dupset

        # Print the proportion of duplicate pairs in batch.
        if args.debug and idx % 50 == 0:
            assert batch_max == len(batch)
            prop_dup = mtx.sum() / (batch_max * (batch_max - 1))
            print('batch duplicates: {0:.5f}'.format(prop_dup))

        # Yield question ids, duplicate matrix
        batch = torch.LongTensor(batch)
        mtx = torch.from_numpy(mtx)
        yield (batch, mtx)
