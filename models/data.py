'''Data loading/cleaning/extraction.'''

import numpy as np
import torch
import logging

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
        unknown_words: int value, number of unknown words.'''
    if check:
        assert embeddings.size(1) == out.size(1)
        assert out.size(0) == len(words)
        assert len(embeddings.size()) == 2  # 2D Tensor ?xD
        assert len(out.size()) == 2  # 2D Tensor LxD
        assert len(words) > 0

    unknown_words = 0
    for i, w in enumerate(words):
        if w in dictionary:
            out[i,:] = embeddings[dictionary[w],:]
        else:
            out[i,:] = unk_embedding
            unknown_words += 1
    return unknown_words



# Reduce to 100d, to speed things up
# dictionary, embed = load_embeddings("../data/glove.6B.100d.txt")
# print(embed.shape)
