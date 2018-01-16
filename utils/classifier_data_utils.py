import torch
import random
import torch.utils.data
import subprocess
import math
import data_processor as dp
import lm_data_utils
import nltk
from constants import *
from functools import reduce

# Data management for all classifiers: real/fake discriminator (i.e. encoder), and acceptability

VOCAB_PATH = "../data/data/bnc/bnc-tokenized-crop30-vocab-20000.txt"

class CorpusEpoch:
    """
    new instance for each epoch, contains refs to full data set, counts progress through current epoch
    """

    #TODO as stack
    def __init__(self, data_pairs, data_manager, batch_size=36):
        self.batch_size = batch_size
        self.data_manager = data_manager
        random.shuffle(data_pairs)
        self.data_pairs = data_pairs
        self.n_lines = len(data_pairs)
        self.n_batches = self.n_lines / self.batch_size
        self.curr_line = 0
        self.still_going = True

    def get_new_batch(self):
        pairs = [self.data_pairs.pop() for _ in range(min(self.batch_size, len(self.data_pairs)))]
        batch = Batch(pairs, self.data_manager)
        has_next = len(self.data_pairs) > 0
        self.still_going = has_next
        return batch, has_next


class DataManagerInMemory(lm_data_utils.DataManager):
    """
    stores all data for test/dev/training sets as lists of triples: (text, score, source)
    """
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, crop_pad_length=30, unked=False):
        """
        :param corpus_path: directory containing train.txt, valid.txt, test.txt
        :param embedding_path: file containing pretrained glove embeddings
        :param vocab_path: file containing list of word types in vocabulary
        :param embedding_size: dimension of word embeddings
        :param crop_pad_length: length to which all sentences are cropped/padded
        :param unked: whether or not the data contains unk tokens
        """
        super(DataManagerInMemory, self).__init__(corpus_path, embedding_path, vocab_path, embedding_size,
                                                  crop_pad_length=30, unked=unked)
        def read_pairs(path):
            pairs = []
            for line in open(path):
                vals = line.split("\t")
                try:
                    pairs.append((vals[3].strip(), vals[1], vals[0]))   # text, score, source
                    # pairs.append((vals[3].strip(), vals[1]))
                except IndexError:
                    pass
            return pairs

        self.training_pairs = read_pairs(self.training)
        self.valid_pairs = read_pairs(self.valid)
        self.test_pairs = read_pairs(self.test)


class InteractiveDataManager(lm_data_utils.DataManager):
    """
    Data manager for interactive classifier
    """
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, crop_pad_length=30, unked=False):
        super(InteractiveDataManager, self).__init__(corpus_path, embedding_path, vocab_path, embedding_size,
                                                  crop_pad_length=30, unked=unked)


class DataManagerEval(lm_data_utils.DataManager):
    """
    Data manager for evaluation mode
    """
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, crop_pad_length=30, unked=False):
        super(DataManagerEval, self).__init__(corpus_path, embedding_path, vocab_path, embedding_size,
                                                  crop_pad_length=30, unked=unked)
        self.test_path = corpus_path
        def read_pairs(path):
            pairs = []
            for line in open(path):
                vals = line.split("\t")
                try:
                    pairs.append((vals[3].strip(), vals[1], vals[0]))   # text, score, source
                    # pairs.append((vals[3].strip(), vals[1]))
                except IndexError:
                    pass
            return pairs
        self.test_pairs = read_pairs(self.test_path)



class Batch:
    """
    all data for a single batch
    separate lists for text, scores, sources
    can represent text as a list of sentences, list of lists of words, or a list of lists of tensors
    """
    def __init__(self, data_pairs, data_manager):
        self.sentences_view = [pair[0] for pair in data_pairs]
        self.targets_view = [float(pair[1]) for pair in data_pairs]
        self.source_view = [pair[2] for pair in data_pairs]
        self.data_manager = data_manager
        self.words_view = list(map(lambda x: x.split(" "), self.sentences_view))
        if len(self.words_view) == 0:
            x = 0
        self.batch_size = len(data_pairs)
        try:
            self.sentence_length = len(self.words_view[0])
        except IndexError:
            print(self.sentences_view)
        self.indices_view = self.indices_view()
        self.tensor_view = self.tensor_view()


    def nth_words(self, n):
        words = []
        for sentence in self.words_view:
            words.append(sentence[n])
        return words

    def indices_view(self):
        indices = []
        for sentence in self.words_view:
            sentence_indices = []
            for word in sentence:
                try:
                    sentence_indices.append(self.data_manager.vocab.index(word))
                except ValueError:
                    sentence_indices.append(self.data_manager.vocab.index(UNK))
            indices.append(sentence_indices)
        return indices

    def tensor_view(self):
        """makes a list of sentence length of dim_batch x embedding_size tensors"""
        tensors = []
        for _ in range(self.sentence_length):
            tensors.append(torch.Tensor(self.batch_size, self.data_manager.embedding_size))
        for i_s, s in enumerate(self.words_view):
            for i_w, w in enumerate(s):
                tensors[i_w][i_s] = self.data_manager.word_to_tensor(w.strip())
        return tensors

    def true_batch_n_words(self):
        n_words = 0
        for sentence in self.words_view:
            for word in sentence:
                if word != self.data_manager.STOP:
                    n_words += 1
        return n_words