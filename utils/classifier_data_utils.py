import torch
import random
import torch.utils.data
import subprocess
import math
import data_processor as dp
import data_utils
from constants import *
from functools import reduce



VOCAB_PATH = "../data/data/bnc/bnc-tokenized-crop30-vocab-20000.txt"

class CorpusEpoch:

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
        batch = NewBatch(pairs, self.data_manager)
        has_next = len(self.data_pairs) > 0
        return batch, has_next

    # def get_next_batch(self):
    #     batch_lines = []
    #     batch_targets = []
    #     for _ in range(self.batch_size):
    #         if self.curr_line < self.n_lines:
    #             pair = self.data_pairs[self.curr_line]
    #             batch_lines.append(pair[0].strip())
    #             # batch_targets.append(float(pair[1]))
    #             if float(pair[1]) < .5:
    #                 batch_targets.append(0)
    #             else:
    #                 batch_targets.append(1)
    #             self.curr_line += 1
    #         else:
    #             self.still_going = False
    #     if self.curr_line == self.n_lines:
    #         self.still_going = False
    #     return Batch(batch_lines, self.data_manager), batch_targets, self.still_going


class DataManagerInMemory(data_utils.DataManager):
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, crop_pad_length=30, unked=False):
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


# class Batch:
#     def __init__(self, sentences, data_manager):
#         self.sentences_view = sentences
#         self.data_manager = data_manager
#         self.words_view = list(map(lambda x: x.split(" "), sentences))
#         if len(self.words_view) == 0:
#             x = 0
#         self.batch_size = len(sentences)
#         self.sentence_length = len(self.words_view[0])
#         self.indices_view = self.indices_view()
#         self.tensor_view = self.tensor_view()
#
#
#     def nth_words(self, n):
#         words = []
#         for sentence in self.words_view:
#             words.append(sentence[n])
#         return words
#
#     def indices_view(self):
#         indices = []
#         for sentence in self.words_view:
#             sentence_indices = []
#             for word in sentence:
#                 try:
#                     sentence_indices.append(self.data_manager.vocab.index(word))
#                 except ValueError:
#                     sentence_indices.append(self.data_manager.vocab.index(UNK))
#             indices.append(sentence_indices)
#         return indices
#
#     def tensor_view(self):
#         """makes a list of sentence length of dim_batch x 50 tensors"""
#         tensors = []
#         for _ in range(self.sentence_length):
#             tensors.append(torch.Tensor(self.batch_size, self.data_manager.embedding_size))
#         for i_s, s in enumerate(self.words_view):
#             for i_w, w in enumerate(s):
#                 tensors[i_w][i_s] = self.data_manager.word_to_tensor(w)
#         return tensors
#
#     def true_batch_n_words(self):
#         n_words = 0
#         for sentence in self.words_view:
#             for word in sentence:
#                 if word != self.data_manager.STOP:
#                     n_words += 1
#         return n_words



class NewBatch:
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
        """makes a list of sentence length of dim_batch x 50 tensors"""
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








#
# class CorpusInMemory:
#     """a corpus is a data set on the order of training/test/validation"""
#     def __init__(self, path, crop_pad_length, batch_size=36):
#         self.path = path
#         self.crop_pad_length = crop_pad_length
#         self.stop_pad = " "
#         for _ in range(self.crop_pad_length):
#             self.stop_pad = self.stop_pad + STOP + " "
#         self.lines = self.crop_data_lines(path)
#         self.n_lines = len(self.lines)
#
#     def prepare(self, sentence):
#         sentence = START + " " + sentence.strip() + self.stop_pad
#         words = sentence.split(" ")
#         words = words[:self.crop_pad_length]
#         return reduce(lambda s, t: s + " " + t, words)
#
#     def crop_data_lines(self):
#         full_text_file = open(self.path)
#         lines = []
#         for line in full_text_file:
#             lines.append(self.prepare(line))
#         return lines
#
#     def shuffle(self):
#         random.shuffle(self.lines)
