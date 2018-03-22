import glob
import lxml.etree as et
import nltk
import random
import operator
import math
import os
import torch
import errno
import matplotlib.pyplot as plt
from constants import *
from functools import reduce

# Various useful functions for processing datasets

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

def tokenize_with_header(start_corpus_path):
    out_path = start_corpus_path[:-4] + "-tokenized.txt"
    tokenized_corpus = open(out_path, "w")
    print("tokenizing ", start_corpus_path)
    for line in open(start_corpus_path):
        vals = line.split("\t")
        tokens = [x.strip().lower() for x in nltk.word_tokenize(vals[3])]
        token_line = " ".join(tokens)
        vals[3] = token_line
        tokenized_corpus.write("\t".join(vals) + "\n")
    tokenized_corpus.close()


def tokenize_sentence(s):
    tokens = nltk.word_tokenize(s)
    s = " ".join(tokens).lower()
    return s

def sanitize_sentence(s, crop_pad_length=30):
    s = s.strip().lower()
    s = tokenize_sentence(s)
    s = crop_line(s, crop_pad_length)
    return s

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
    out_file = open(out_path, "w")
    print("getting vocab of size %d for %s" % (n_vocab, start_corpus_path))
    for line in open(start_corpus_path):
        for word in line.split():
            if word in counts:
                counts[word] = counts[word] + 1
            else:
                counts[word] = 1
                if len(counts) % 10000 is 0:
                    print("n_words =", len(counts))
    counts_sorted = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    if n_vocab < len(counts_sorted):
        counts_sorted = counts_sorted[:n_vocab]
    vocab = [x[0] for x in counts_sorted]
    for word in vocab:
        out_file.write(word + "\n")


def unkify(start_corpus_path, vocab_file):
    vocab = set()
    for line in open(vocab_file):
        vocab.add(line.strip())
    out_path = start_corpus_path[:-4] + "-unked-%dwords.txt" % len(vocab)
    out_file = open(out_path, "w")
    for line in open(start_corpus_path):
        vals = line.split("\t")
        line = vals[3]
        tokens = nltk.word_tokenize(line)
        for i, word in enumerate(tokens):
            if word.lower() not in vocab:
                tokens[i] = UNK
        line = " ".join(tokens)
        vals[3] = line
        out_file.write("\t".join(vals) + "\n")
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



def get_word_type_counts(path):
    corpus = open(path)
    type_counts = {}
    for line in corpus:
        for word in line.split():
            count = type_counts[word] if word in type_counts else 0
            type_counts[word] = count + 1
    return type_counts


def get_line_length_counts(path):
    corpus = open(path)
    line_lengths = {}
    for line in corpus:
        length = len(line.split())
        count = line_lengths[length] if length in line_lengths else 0
        line_lengths[length] = count + 1
    return line_lengths


def get_word_count_counts(path):
    """for each word frequency count, the number of word types in corpus"""
    type_counts = get_word_type_counts(path)
    frequency_counts = {}
    for type in type_counts.keys():
        frequency = type_counts[type]
        count = frequency_counts[frequency] if frequency in frequency_counts else 0
        frequency_counts[frequency] = count + 1
    return frequency_counts


def embeddings_vocab(path, size, out):
    embeddings = open(path)
    n = 0
    out = open(out, "w")
    for line in embeddings:
        out.write(line.split()[0] + "\n")
        # vocab.append(line.split()[0])
        n += 1
        if n == size:
            break
    # return vocab




def tokenize_corpus_with_prefix():
    out = open("../acceptability_corpus/test_tokenized", "w")
    for line in open('../acceptability_corpus/raw/test.tsv'):
        vals = line.split("\t")
        line = vals[3]
        line = line.lower()
        tokens = nltk.word_tokenize(line)
        token_line = reduce(lambda s, t: s + " " + t, tokens, "").strip().lower()
        out.write("%s\t%s\t%s\t%s\n" % (vals[0], vals[1], vals[2], token_line))


def any_unks(vocab, out=""):
    voc = {}
    unks = set()
    # for w in embeddings_vocab('embeddings/glove.6B.300d.txt'):
    for w in open(vocab):
        voc[w.strip()] = 1
    for line in open('../acceptability_corpus/corpus_table.tsv'):
        vals = line.split("\t")
        line = vals[3]
        tokens = nltk.word_tokenize(line)
        for w in tokens:
            try:
                voc[w.lower()]
            except KeyError:
                unks.add(w)
    if out is "":
        for u in unks:
            print(u)
    else:
        out = open(out, "w")
        for u in unks:
            out.write(u + "\n")

def crop_corpus_with_prefix():
    file = open("lm_generated/all_lm_and_bnc-long_lines")
    out = open("lm_generated/all_lm_and_bnc-long_lines-cropped", "w")
    stop_pad = " "
    for _ in range(crop_pad_length):
        stop_pad = stop_pad + STOP + " "
    for line in file:
        if line is not "\n":
            vals = line.split("\t")
            line = vals[3].strip() + stop_pad
            words = line.split(" ")
            words = words[:crop_pad_length]
            line = reduce(lambda s, t: s + " " + t, words) + " " + STOP
            out.write("%s\t%s\t%s\t%s\n" % (vals[0], vals[1], vals[2], line))
        else:
            continue

def un_crop_corpus_with_prefix(in_file, out_file):
    in_file = open(in_file)
    out_file = open(out_file, "w")
    for line in in_file:
        if line is not "\n":
            vals = line.split("\t")
            line = vals[3].strip()
            words = line.split(" ")
            words = filter(lambda x: x != STOP and x != START, words)
            line = " ".join(words)
            out_file.write("%s\t%s\t%s\t%s\n" % (vals[0], vals[1], vals[2], line))
        else:
            continue

# def crop_sentences(sentences):
#     stop_pad = " "
#     for _ in range(crop_pad_length):
#         stop_pad = stop_pad + STOP + " "
#     cropped_sentences = []
#     for s in sentences:
#         s = START + " " + s + stop_pad
#         words = s.split()
#         words = words[:crop_pad_length]
#         s = reduce(lambda w, v: w + " " + v, words) + " " + STOP
#         cropped_sentences.append(s)
#     return cropped_sentences

def prefix(input, output, src, score, mark):
    file = open(input)
    out = open(output, "w")
    for line in file:
        out.write("%s\t%s\t%s\t%s\n" % (src, score, mark, line.strip()))
    out.close()



def corpus_bias():
    file = open("acceptability_corpus/corpus_table_tokenized_crop30")
    n_lines = 0
    total_acceptability = 0
    for line in file:
        n_lines += 1
        total_acceptability += float(line.split("\t")[1])
    return total_acceptability/n_lines


def remove_short_lines_with_prefix():
    file = open('lm_generated/all_lm_and_bnc')
    out = open("lm_generated/all_lm_and_bnc-long_lines", "w")
    non_words = [STOP, START, ",", ".", "\"", "?", "!", "(", ")", "[", "]", "``", "''"]
    for line in file:
        vals = line.split("\t")
        s = vals[3].strip()
        words = s.split(" ")
        words = [x for x in words if x not in non_words]
        if len(words) > 3:
            out.write(line)

def split(in_path, out_dir, train, test, valid):
    if train + test + valid != 1:
        print("proportions should sum to 1")
    else:
        train_out = open(out_dir + "train.tsv", "w+")
        test_out = open(out_dir + "test.tsv", "w+")
        valid_out = open(out_dir + "valid.tsv", "w+")
        for line in open(in_path):
            n = random.uniform(0, 1)
            if n <= train:
                train_out.write(line)
            elif n <= train + valid:
                valid_out.write(line)
            else:
                test_out.write(line)
        train_out.close(), valid_out.close(), test_out.close()


def verify_corpus_table(in_path, out_path):
    out = open(out_path, "w")
    for line in open(in_path):
        vals = line.split("\t")
        if len(vals) == 4:
            words = vals[3].split()
            if len(words) == 31:
                out.write(line)
            else:
                print(line)
                print(len(words))
        else:
            print(line)
    out.close()


def crop_line(line, crop_pad_length):
    stop_pad = ""
    for _ in range(crop_pad_length):
        stop_pad = stop_pad + STOP + " "
    line = START + " " + line.strip() + " " + stop_pad
    words = line.split(" ")
    words = words[:crop_pad_length]
    return " ".join(words) + " " + STOP



def write_percent(input, output, percent):
    if os.path.exists(output):
        output = open(output, "a")
    else:
        output = open(output, "w")
    for line in open(input):
        r = random.random()
        if r < percent:
            output.write(line)
    output.close()


def source_stats(file):
    counts = {}
    for line in open(file):
        vals = line.split("\t")
        if vals[0] not in counts:
            counts[vals[0]] = (0,0,0)
        counts[vals[0]] = (counts[vals[0]][0], counts[vals[0]][1], counts[vals[0]][2] + 1)
        if vals[1] == '0':
            counts[vals[0]] = (counts[vals[0]][0] + 1, counts[vals[0]][1], counts[vals[0]][2])
        else:
            counts[vals[0]] = (counts[vals[0]][0], counts[vals[0]][1] + 1, counts[vals[0]][2])
    print counts.__str__()





#=============================== MAIN ===============================

workspace = "/Users/alexwarstadt/Workspace/"

tokenize_with_header(workspace + "acceptability/acceptability_corpus/raw/fake_test.tsv")










