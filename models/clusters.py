import json
import numpy as np
import torch
import itertools

train_clusters = json.load(open('../data/train_clusters.json'))
valid_clusters = json.load(open('../data/valid_clusters.json'))
dupset, filler_base = read_dupset()

def iterate_filler():
    while True:
        np.random.shuffle(filler_base)
        for f in filler_base:
            yield f

filler_questions = iterate_filler()

def read_dupset():
    frame = pd.read_csv('../data/all_questions.tsv', delimiter='\t',
            index_col=None, header=None)
    dups = []
    qids = []
    for row in frame.itertuples():
        dups.append(row[0], row[1])
        dups.append(row[1], row[0])
        qids.append(row[0])
        qids.append(row[1])
    return set(dups), set(qids)

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

        # Consult filler.
        check = {qid: i for i, qid in enumerate(batch)}
        while len(batch) < batch_max:
            qid = next(filler_questions)
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
