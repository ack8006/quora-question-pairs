'''Data loading/cleaning/extraction.'''

import numpy as np
import torch
import logging
import csv

logger = logging.getLogger('data')

def load_embeddings(filepath):
    '''Load a GloVE embedding into a Tensor.

    Args:
        filepath: path to a GloVE embeding file (i.e. 6B.100d.txt)
    
    Returns:
        dictionary: a Python dict from word string to row number
        tensor: a torch.Tensor containing the embeddings (no CUDA)'''
    dictionary = {}
    current_word = 0
    rows = []
   
    logger.info('loading from %s', filepath)
    read_lines = 0
    with open(filepath, "rb") as f:
        for line in f:
            line = line.decode("utf-8")
            elements = line.strip().split(' ', 1)
            word = elements[0]
            vector = [float(x) for x in elements[1].split(' ')]
            vector = np.array(vector)
            
            word_id = current_word
            current_word += 1
            dictionary[word] = word_id
            rows.append(vector)

            read_lines += 1
            if read_lines % 10000 == 0:
                logger.info('loaded %d words', read_lines)
    return dictionary, torch.Tensor(np.stack(rows, axis=0))


def embed_words(dictionary, embeddings, words, out, unk_embedding=0, check=True):
    '''Turn a tokenized sentence into a Tensor in-place.
    
    Args:
        dictionary: Python dict mapping word to row id.
        embeddings: A ?xD Tensor containing the word embeddings.
        words: List of strings of length L.
        out: An LxD tensor. Values will be written into this Tensor.
        unk_embedding: Value to enter if word is unknown.
        check: Do extra correctness checks in the input.
        
    Returns:
        words: int value, number of known words.'''
    if check:
        assert embeddings.size(1) == out.size(1)
        assert out.size(0) == len(words)
        assert len(embeddings.size()) == 2  # 2D Tensor ?xD
        assert len(out.size()) == 2  # 2D Tensor LxD
        assert len(words) > 0

    known_words = 0
    for i, w in enumerate(words):
        if w in dictionary:
            out[i,:] = embeddings[dictionary[w],:]
            known_words += 1
        else:
            out[i,:] = unk_embedding
    return known_words


def get_reweighted_embeddings(
        tfidf, dictionary, embeddings, sentence, out, check=True):
    '''Gets word vectors for the sentence reweighted using tf-idf.

    The sentences are vectorized, so the length of the embedding is probably
    less than 

    the 'out' matrix gets filled with one word each time, and must be able to
    fit the unique words of the sentence. Not all rows of the matrix will be
    filled out.

    Args:
        tfidf: pretrained TfIdfVectorizer
        dictionary: a Python dict from word string to row number
        embeddings: a ?xD torch.Tensor containing the embeddings
        sentence: list of words to vectorize of length L.
        out: Tensor size unique(L)xD, where to store the sentence vectors.
             The size of dim2 must be exactly D, but dim1 just needs to be big
             enough to contain the row.
        check: if True, do argument checking.

    Returns:
        words: How many rows in `out` are filled out.'''
    if check:
        assert embeddings.size(1) == out.size(1)
        assert out.size(0) >= 1
        assert len(embeddings.size()) == 2  # 2D Tensor ?xD
        assert len(out.size()) == 2  # 2D Tensor LxD
        assert len(sentence) > 0

    unique_words = set(filter(lambda w: w in dictionary, sentence))
    if len(unique_words) == 0:
        return 0
    weights = tfidf.transform([' '.join(sentence)])
    nonzero_weights = set(weights.indices)
    words = 0
    for w in unique_words:
        dict_w = tfidf.vocabulary_.get(w)
        if dict_w is None or dict_w not in nonzero_weights:
            continue
        tfidf_weight = weights[0, dict_w]
        out[words,:] = tfidf_weight * embeddings[dictionary[w],:]
        words += 1
    return words


def vectorize(
        dictionary, embeddings, dataset,
        q1_tensor, q2_tensor, tf=None, max_word_len=30):
    '''Compute tf-idf vectors for each word for each sentence in the dataset.

    Args:
        dataset: a list of triplets of (q1, q2, y). q1/q2 are lists of words.
        dictionary: a Python dict from word string to row number
        embeddings: a ?xD torch.Tensor containing the word embeddings
        q1_tensor: big tensor to hold values for q1.
        q2_tensor: big tensor to hold values for q2.
        tf: TfIdfVectorizer (optional)
        max_word_len: maximum length of sentences (default 30).

    Returns:
        vectorized: a list of triplets (q1, q2, y). q1/q2 are tensors backed
                    by the big q1_tensor and q2_tensors.
    '''
    assert len(dataset) <= q1_tensor.size(0)
    assert len(dataset) <= q2_tensor.size(0)
    assert max_word_len <= q1_tensor.size(1)
    assert max_word_len <= q2_tensor.size(1)
    assert embeddings.size(1) == q1_tensor.size(2)
    assert embeddings.size(1) == q2_tensor.size(2)
    assert len(embeddings.size()) == 2  # 2D Tensor ?xD
    assert len(q1_tensor.size()) == 3
    assert len(q2_tensor.size()) == 3

    embed_command = lambda sentence, tensor: embed_words(
            dictionary, embeddings, sentence, tensor, check=False)
    if tf is not None:
        embed_command = lambda sentence, tensor: get_reweighted_embeddings(
                tf, dictionary, embeddings, sentence, tensor)

    vectorized = []
    for i, (q1, q2, y) in enumerate(dataset):
        if i % 2000 == 0 and i > 0:
            logger.info('vectorizing: %d examples processed', i)
        q1_words = embed_command(q1[:max_word_len], q1_tensor[i])
        q2_words = embed_command(q2[:max_word_len], q2_tensor[i])
        if q1_words == 0 or q2_words == 0:
            vectorized.append((None, None, y)) # To make the indices match.
        else:
            vectorized.append((
                q1_tensor[i,:q1_words,:],
                q2_tensor[i,:q2_words,:],
                y))
    return vectorized


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
