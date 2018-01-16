import torch
import random
import torch.utils.data
import subprocess
import data_processor as dp
from constants import *
from functools import reduce

# Data management for Language Model only

class CorpusEpoch:
    def __init__(self, path, data_manager, batch_size=36):
        self.batch_size = batch_size
        self.data_manager = data_manager
        for i, _ in enumerate(open(path)):
            pass
        self.n_lines = i
        shuffled_path = path[:-4] + "-shuff.txt"
        out = open(shuffled_path, "w")
        # command = "gshuf %s > %s" % (self.processed_path, self.shuffled_path)
        subprocess.call(["gshuf", path], stdout=out)
        out.close()
        self.lines = open(shuffled_path)

    def get_next_batch(self):
        batch_lines = []
        still_going = True
        for i in range(self.batch_size):
            line = self.lines.readline()
            if line is not "":
                batch_lines.append(self.lines.readline().strip())
            else:
                still_going = False
                break
        return Batch(batch_lines, self.data_manager), still_going


class DataManager(object):
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, crop_pad_length=30, unked=False):
        # self.corpus_path = corpus_path
        self.embedding_size = embedding_size
        self.vocab = [x.strip() for x in open(vocab_path)]
        self.crop_pad_length = crop_pad_length
        self.n_vocab = len(self.vocab)
        self.training, self.valid, self.test = corpus_path + "/train.txt", corpus_path + "/valid.txt", corpus_path + "/test.txt"
        self.embeddings = dp.read_embeddings(embedding_path)

    def word_to_tensor(self, word):
        """makes 50 dim vector out of word"""
        tensor = torch.Tensor(self.embedding_size)
        if word in self.embeddings.keys():
            tensor = self.embeddings[word]
        else:
            tensor = self.embeddings[UNK]
        return tensor


class Batch:
    def __init__(self, sentences, data_manager):
        self.sentences_view = sentences
        self.data_manager = data_manager
        self.words_view = list(map(lambda x: x.split(" "), sentences))
        self.batch_size = len(sentences)
        self.sentence_length = len(self.words_view[0])
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
                tensors[i_w][i_s] = self.data_manager.word_to_tensor(w)
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
