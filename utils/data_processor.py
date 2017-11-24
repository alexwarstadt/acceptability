import glob
import lxml.etree as et
import nltk
import random
import operator
import math
import os
import torch
import errno
# import matplotlib.pyplot as plt
from constants import *
from functools import reduce


def tokenize(start_corpus_path):
    out_path = start_corpus_path[:-4] + "-tokenized.txt"
    if os.path.isfile(out_path):
        print("tokenized file %s already exists" % out_path)
    else:
        tokenized_corpus = open(out_path, "x")
        print("tokenizing ", start_corpus_path)
        for line in open(start_corpus_path):
            tokens = nltk.word_tokenize(line)
            token_line = reduce(lambda s, t: s + " " + t, tokens, "").strip().lower()
            tokenized_corpus.write(token_line + "\n")
        tokenized_corpus.close()


def filter_short_lines(start_corpus_path, n_words):
    temp_file = open(start_corpus_path + "temp", "w")
    for line in open(start_corpus_path):
        if len(line.split()) == n_words:
            temp_file.write(line)
    os.rename(start_corpus_path+"temp", start_corpus_path)



def crop(start_corpus_path, crop_pad_length):
    full_text_file = open(start_corpus_path)
    out_path = start_corpus_path[:-4] + "-crop%d.txt" % crop_pad_length
    if os.path.isfile(out_path):
        print("cropped file %s already exists" % out_path)
    else:
        crop_text_file = open(out_path, "x")
        print("cropping %s to %d words" % (start_corpus_path, crop_pad_length))
        stop_pad = " "
        for _ in range(crop_pad_length):
            stop_pad = stop_pad + STOP + " "
        for line in full_text_file:
            if line is not "\n":
                line = START + " " + line.strip() + stop_pad
                words = line.split(" ")
                words = words[:crop_pad_length]
                line = reduce(lambda s, t: s + " " + t, words) + " " + STOP
                crop_text_file.write(line + "\n")
            else:
                continue


def get_vocab(start_corpus_path, n_vocab=float("inf")):
    counts = {}
    out_path = start_corpus_path[:-4] + "-vocab-" + str(n_vocab) + ".txt"
    if n_vocab is math.inf:
        out_path = start_corpus_path[:-4] + "-vocab-all.txt"
    if os.path.isfile(out_path):
        print("vocab file %s already exists" % out_path)
    else:
        out_file = open(out_path, "x")
        print("getting vocab of size %d for %s" % (n_vocab, start_corpus_path))
        for line in open(start_corpus_path):
            for word in line.split():
                if word in counts:
                    counts[word] = counts[word] + 1
                else:
                    counts[word] = 1
                    if len(counts) % 10000 is 0:
                        print("n_words =", len(counts))
        counts_sorted = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)    # sort vocab by counts
        if n_vocab < len(counts_sorted):
            counts_sorted = counts_sorted[:n_vocab]
        vocab = [x[0] for x in counts_sorted]
        for word in vocab:
            out_file.write(word + "\n")


def unkify(start_corpus_path, vocab):
    out_path = start_corpus_path[:-4] + "-unked-%dwords.txt" % len(vocab)
    if os.path.isfile(out_path):
        print("unked file %s already exists" % out_path)
    else:
        out_file = open(out_path, "x")
        print("unking", start_corpus_path)
        for line in open(start_corpus_path):
            words = line.split()
            for i, word in enumerate(words):
                if word not in vocab:
                    words[i] = UNK
            out_file.write(reduce(lambda s, t: s + " " + t, words).strip() + "\n")
        out_file.close()


def init_embeddings(embeddings_path, vocab, corpus_name, embedding_size):
    #TODO why doesn't generation work when a new embeddings dict is built?
    out_path = embeddings_path[:-4] + "-%s-%dwords.txt" % (corpus_name.split("/")[-3], len(vocab))
    embeddings_dict = dict.fromkeys(list(vocab))
    if os.path.isfile(out_path):
        print("embeddings file %s already exists" % out_path)
    else:
        out_file = open(out_path, "x")
        embeddings_file = open(embeddings_path)
        for line in embeddings_file:
            words = line.split(" ")
            if words[0] in embeddings_dict:
                vec_list = []
                for word in words[1:]:
                    vec_list.append(float(word))
                embeddings_dict[words[0]] = torch.FloatTensor(vec_list)
        for w in vocab:
            if embeddings_dict[w] is None:
                vector = torch.FloatTensor(embedding_size)
                for i in range(embedding_size):
                    vector[i] = random.uniform(-1, 1)
                embeddings_dict[w] = vector
        for k, v in embeddings_dict.items():
            tensor_string = reduce(lambda s1, s2: str(s1) + " " + str(s2), v).strip()
            out_file.write(k + " " + tensor_string + "\n")
        out_file.close()


def read_embeddings(embeddings_path):
    embeddings_dict = {}
    for line in open(embeddings_path):
        words = line.split(" ")
        vec_list = []
        for word in words[1:]:
            vec_list.append(float(word))
        embeddings_dict[words[0]] = torch.FloatTensor(vec_list)
    return embeddings_dict


def apply_xslt(text_paths, xslt_path, data_dir):
    xslt = et.parse(xslt_path)
    transform = et.XSLT(xslt)
    output = open(data_dir + "raw_text", "w")
    for path in text_paths:
        dom = et.parse(path)
        newdom = transform(dom)
        output.write(str(newdom) + "\n")
    output.close()


def permute_sentences(input_path, out_path, min_percent, max_percent):
    if os.path.isfile(out_path):
        print("permuted file %s already exists" % out_path)
    else:
        output = open(out_path, "w")
        for line in open(input_path):
            words = line.split()
            original_length = len(words)
            words = remove_pad(words)
            words, punc_map = remove_punc(words)
            words = shuffle(words, random.uniform(.05, .3))
            words = replace_punc(words, punc_map)
            words = add_pad(words, original_length)
            new_line = reduce(lambda x, y: x + " " + y, words)
            output.write(new_line + "\n")
    output.close()

    # from utils.data_processor import permute_sentences; permute_sentences("")


def swap_permute(input_path):
    out_path = input_path[:-4] + "-permuted.txt"
    output = open(out_path, "w")
    for line in open(input_path):
        words = line.split()
        original_length = len(words)
        words = remove_pad(words)
        words, punc_map = remove_punc(words)
        words = swap(words, random.randint(1,1))
        words = replace_punc(words, punc_map)
        words = add_pad(words, original_length)
        new_line = reduce(lambda x, y: x + " " + y, words)
        output.write(new_line + "\n")
    output.close()

def swap(words, n):
    for _ in range(n):
        start = random.randint(0, max(len(words)-2, 1))
        end = random.randint(start, max(len(words)-1, 1))
        split_point = random.randint(start, end)
        before = words[0:start]
        first_half = words[0:split_point]
        after = words[split_point:]
        after.extend(before)
        words = after
    return words



def remove_punc(words):
    punc = [".", ",", ";", ":", "?", "!", "\"", "\'", "`", "``", "\'\'", "(", ")", "[", "]"]
    punc_map = {}
    true_words = []
    for i in range(len(words)):
        if words[i] in punc:
            punc_map[i] = words[i]
        else:
            true_words.append(words[i])
    return true_words, punc_map

def replace_punc(words, punc_map):
    for i in range(len(words) + len(punc_map)):
        if i in punc_map:
            words.insert(i, punc_map[i])
    return words


def shuffle(a_list, percent):
    """shuffle n elements of a list"""
    n = int(math.floor(percent * len(a_list)))
    if n < 2 and len(a_list) >= 2:
        n = 2
    idx = range(len(a_list))
    random.shuffle(idx)
    idx = idx[:n]
    mapping = dict((idx[i], idx[i - 1]) for i in range(n))
    return [a_list[mapping.get(x, x)] for x in range(len(a_list))]



def remove_pad(words):
    return [x for x in words if x != START and x != STOP]


def add_pad(words, n):
    words.insert(0, START)
    while len(words) < n:
        words.append(STOP)
    return words









#=============================== MAIN ===============================


# swap_permute("/Users/alexwarstadt/Workspace/data/bnc-30/test.txt")


raw_corpus = "data/bnc/bnc.txt"
crop_pad_length = 30
n_vocab = 50000





# e_v = embeddings_vocab('embeddings/glove.6B.300d.txt')
# print()

# frequency_counts = get_word_count_counts(raw_corpus)
# line_lengths = get_line_length_counts(raw_corpus)
#
# print(line_lengths)
# plt.plot(list(frequency_counts.keys()),
#          list(frequency_counts.values()))
# plt.show()
# plt.plot([1,2,3], [7, 8, 9])
# plt.show()

# tokenized = tokenize(raw_corpus)
# cropped = crop(tokenized, crop_pad_length)
# vocab = get_vocab(cropped, 20000)
# n_vocab = len(vocab)
# unked = unkify(cropped, vocab)
# filter_short_lines(unked, crop_pad_length+1)
# self.training, self.valid, self.test = dp.split(unked, .85, .05, .10)
# self.embeddings = self.init_embeddings(open(embedding_path))
# bnc =











