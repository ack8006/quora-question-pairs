import data
import features
import torch
from torch.autograd import Variable
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class QuoraSlice:
    '''A slice of Quora labeled data.'''
    def __init__(
            self,
            word_embeddings_path,
            tfidf_corpus,
            dataset,
            glove_max_words=20000,
            glove_dim=100,
            sentence_length=30,
            permute_batchsize=25,
            parent_model=None):

        if parent_model is None:
            self.sentence_length = sentence_length

            # ==================================================================
            # Word embeddings.
            # ==================================================================
            print('Loading GloVE.')

            self.glove_dictionary, self.glove_lookup, self.glove_embed = \
                data.load_embeddings(
                    '../data/glove.6B.{0}d.txt'.format(glove_dim),
                    max_words=glove_max_words)
            self.word_dim = self.glove_embed.weight.size(1)
            self.vocab_size = len(self.glove_dictionary)

            print('Loaded {0} words (D={1}) from GloVE.'.format(
                self.vocab_size, self.word_dim))

            # ==================================================================
            # Count TF-IDF.
            # ==================================================================
            print('Analyzing tf-idf.')

            tfidf = TfidfVectorizer(norm=None) # Norm none is important.
            tfidf.fit(itertools.imap(lambda x: ' '.join(x), tfidf_corpus))
            pickle.dump(tfidf, open('tfidf.pickle', 'wb'))
            self.tfidf_embed = data.convert_tfidf_vectorizer(
                    tfidf, self.glove_lookup)
            print('tf-idf done.')

            # Other modules/features.
            self.perm = features.Permute(differentiable=False)

        else:  # Model inherits from a parent.
            print('Taking embeddings from parent.')
            attributes = [
                'glove_dictionary',
                'glove_lookup',
                'glove_embed',
                'word_dim',
                'vocab_size',
                'sentence_length',
                'tfidf_embed',
                'perm'
            ]
            for atr in attributes:
                self.__dict__[atr] = getattr(parent_model, atr)

        # ==================================================================
        # Tensorize: Turn triplets into 3 Tensors.
        # ==================================================================
        print('Tensorizing sentences to LongTensors')
        self.q1_words, self.q2_words, self.y = \
            data.tensorize(dataset, dictionary, length=sentence_length)
        self data_size = self.q1_words.size(0)
        print(self.q1_words[3:10])
        print('Loaded {0} pairs', self.data_size)

        # ==================================================================
        # Embed: Get TF-IDF embeddings.
        # ==================================================================
        print('Embedding dataset using tf-idf and glove...')
        self.X_q1 = data.get_reweighted_embeddings(
                self.tfidf_embed, self.glove_embed, Variable(self.q1_words))
        self.X_q2 = data.get_reweighted_embeddings(
                self.tfidf_embed, self.glove_embed, Variable(self.q2_words))
        print(self.X_q1.size())

        # ==================================================================
        # Feature 1: Word vector means.
        # ==================================================================
        print('Computing means...')
        self.X_q1_mean = self.X_q1.mean(dim=1).squeeze()
        self.X_q2_mean = self.X_q2.mean(dim=1).squeeze()

        # ==================================================================
        # Feature 1: Word vector means.
        # ==================================================================
        print('Computing means...')
        self.X_q1_mean = self.X_q1.mean(dim=1).squeeze()
        self.X_q2_mean = self.X_q2.mean(dim=1).squeeze()

        # ==================================================================
        # Feature 2: Permutation p-values.
        # ==================================================================
        print('Permuting words...')
        do_perm = lambda q1, q2: self.perm(q1, q2)[1]
        self.X_permuted = features.apply(\
                self.X_q1, self.X_q2,
                do_perm, batchsize=permute_batchsize, print_every=100)
        print('Result: ' + str(X_train_permute.size()))

        # ==================================================================
        # Feature 3: KL-divergence based.
        # ==================================================================
        print('KL divergence')
        self.X_kl12 = features.apply(\
                self.X_q1, self.X_q2,
                features.elementwise_kl_div, print_every=100)
        self.X_kl21 = features.apply(\
                self.X_q2, self.X_q1,
                features.elementwise_kl_div, print_every=100)
        self.X_kls = features.apply(\
                self.X_q2, self.X_q1,
                features.symmetric_kl_div, print_every=100)





