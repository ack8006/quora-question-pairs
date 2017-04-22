'''Data loading/cleaning/extraction.'''

import numpy as np
import torch
import logging
import csv

from torch import nn
from torch.autograd import Variable

logger = logging.getLogger('data')

def load_embeddings(filepath, max_words=50000, embed_dim=100, learnable=True):
    '''Load a GloVE embedding into a nn.Embedding.
    The 2nd-to-last word will be changed into a 'padding' word with embedding 0.
    The last word will be changed into an "unknown" embedding by averaging.

    Args:
        filepath: path to a GloVE embeding file (i.e. 6B.100d.txt)
        max_words: maximum # of words to load.
        embed_size: dimension embedding size.
    
    Returns:
        dictionary: a Python dict from word string to row number
        lookup: a Python array from row number to word string
        embed: the Embeddings module.'''
    dictionary = {}
    lookup = []
    rows = []
   
    logger.info('loading from %s', filepath)
    with open(filepath, "rb") as f:
        for current_word, line in enumerate(f):
            line = line.decode("utf-8")
            elements = line.strip().split(' ', 1)
            word = elements[0]
            vector = [float(x) for x in elements[1].split(' ')]
            vector = np.array(vector)
            
            word_id = current_word
            dictionary[word] = word_id
            lookup.append(word)
            rows.append(vector)

            if current_word % 10000 == 0:
                logger.info('loaded %d words', current_word)
            if current_word == max_words - 1:
                break
    embeddings = torch.from_numpy(np.stack(rows, axis=0)).float()
    avg = embeddings.mean(dim=0)
    embeddings[-1,:] = avg
    embeddings[-2,:] = 0
    del dictionary[lookup[-1]]
    del dictionary[lookup[-2]]
    dictionary['<unk>'] = max_words-1
    dictionary['<pad>'] = max_words-2
    lookup[-1] = '<unk>'
    lookup[-2] = '<pad>'

    embed = nn.Embedding(embeddings.size(0), embeddings.size(1))
    # Word embeddings are frozen.
    embed.weight = nn.Parameter(embeddings, requires_grad=learnable)
    return dictionary, lookup, embed


def tensorize(dataset, dictionary, length=30):
    '''Turn a dataset into tensors.

    Length of dataset might change (this function may drop rows),
    but it will keep the rows between the outputs consistent.

    Args:
        dataset: iterable of tuples of (q1, q2, y)
        dictionary: mapping from string to int (word embedding index)

    Returns:
        q1_tensor: torch.LongTensor with word ids.
        q2_tensor: torch.LongTensor with word ids.
        y_tensor: torch.ByteTensor of is_duplicate.
    '''
    def to_indices(words):
        ql = [dictionary.get(w, dictionary['<unk>']) for w in words]
        qv = np.ones(length, dtype=int) * dictionary['<pad>'] # all padding
        qv[:len(ql)] = ql[:length] # set values
        return qv

    q1s = []
    q2s = []
    ys = []
    for q1, q2, y in dataset:
        q1 = to_indices(q1)
        q2 = to_indices(q2)
        if sum(q1) <= 0 or sum(q2) <= 0:
            continue # Completely invalid sentence; reject
        q1s.append(q1)
        q2s.append(q2)
        ys.append(y)
    q1s = torch.LongTensor(np.stack(q1s, axis=0))
    q2s = torch.LongTensor(np.stack(q2s, axis=0))
    ys = torch.ByteTensor(ys)

    return q1s, q2s, ys


def convert_tfidf_vectorizer(tfidf, lookup, missing_weight=0.1):
    '''Converts a TfIdfVectorizer into a torch.nn.Embedding.
    
    Args:
        tfidf: TfIdfVectorizer. Must be unnormalized (made with norm=None)
        lookup: list of words in embedding index.
        missing_weight: what weight ot put in when word is not in the
            tfidf vocabulary.
        
    Returns:
        tfidf_embed: nn.Embedding form word embedding size to 1.'''
    values = []
    for w in lookup:
        if w in tfidf.vocabulary_:
            sentence_sparse = tfidf.transform([w])
            values.append(max(sentence_sparse.max(), missing_weight))
        else:
            values.append(missing_weight)
    embeddings = np.stack(values, axis=0).reshape(-1, 1)
    embed = nn.Embedding(len(lookup), 1)
    # tfidf embeddings are frozen.
    embed.weight = nn.Parameter(
            torch.from_numpy(embeddings).float(), requires_grad=False)
    return embed


def get_reweighted_embeddings(tfidf, embeddings, sentence):
    '''Gets word vectors for the sentence reweighted using tf-idf.

    Args:
        tfidf: nn.Embedding (1) - TF-IDF weights for each word.
        embeddings: nn.Embedding (D) - Word embeddings.
        sentence: Batch x Len LongTensor, with word ids.
    '''
    emb_words = embeddings(sentence)
    emb_weights = tfidf(sentence)
    return emb_words * emb_weights.repeat(1, 1, emb_words.size(2))


def load_tokenized(path):
    '''Loads a tokenized corpus. Each line should be a sentence delimited by |.
    
    Args:
        path: path to the corpus PSV file (pipe separated).
        
    Returns:
        sentences: list of tuples of words.'''
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        sentences = []
        for row in reader:
            sentences.append(tuple(row))
        return sentences

# Reduce to 100d, to speed things up
# dictionary, embed = load_embeddings("../data/glove.6B.100d.txt")
# print(embed.shape)
