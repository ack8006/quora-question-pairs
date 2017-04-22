import json
import numpy as np
import torch
import itertools
import random

def read_dupset(path):
    frame = pd.read_csv(path, delimiter='\t', index_col=None, header=None)
    dups = []
    for row in frame.itertuples():
        dups.append((row[0], row[1]))
        dups.append((row[1], row[0]))
    return set(dups)

def iterate_epoch(clusters, seed_max=10, take_max=10, batch_max=80):
    np.random.shuffle(clusters)
    for idx in xrange(0, len(clusters), seed_max): # Seed size
        # Get a selection of duplicates as the seed.
        batch = []
        for cluster in clusters[:seed_max]
            np.random.shuffle(cluster)
            batch.extend(cluster[:take_max])
            if len(batch) > batch_max:
                break
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

        mtx = np.zeros((len(batch), len(batch)), dtype=np.int32)
        for i, q1 in enumerate(batch):
            for j, q2 in enumerate(batch):
                mtx[i,j] = (q1, q2) in dupset
        #print(mtx.sum(axis=1))

        # Yield input, duplicate matrix
        mtx = torch.from_numpy(mtx)
        yield (batch_qs, mtx)
