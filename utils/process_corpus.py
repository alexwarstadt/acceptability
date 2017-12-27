import nltk
from constants import *
from functools import reduce
import random

raw_corpus = "data/bnc/bnc.txt"
crop_pad_length = 30
n_vocab = 50000

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


def embeddings_vocab(path):
    embeddings = open(path)
    vocab = []
    for line in embeddings:
        vocab.append(line.split()[0])
    return vocab


def tokenize_corpus_with_prefix():
    out = open("acceptability_corpus/corpus_table_tokenized", "w")
    for line in open('acceptability_corpus/corpus_table'):
        vals = line.split("\t")
        line = vals[3]
        line = line.lower()
        tokens = nltk.word_tokenize(line)
        token_line = reduce(lambda s, t: s + " " + t, tokens, "").strip().lower()
        out.write("%s\t%s\t%s\t%s\n" % (vals[0], vals[1], vals[2], token_line))


def any_unks():
    voc = {}
    for w in embeddings_vocab('embeddings/glove.6B.300d.txt'):
        voc[w] = 1
    for line in open('acceptability_corpus/corpus_table_tokenized'):
        vals = line.split("\t")
        line = vals[3]
        words = line.split()
        for w in words:
            try:
                voc[w]
            except KeyError:
                print(w)


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

def prefix(input, output):
    file = open(input)
    out = open(output, "w")
    for line in file:
        out.write("per	0	*	" + line)



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
        train_out = open(out_dir + "train.txt", "w+")
        test_out = open(out_dir + "test.txt", "w+")
        valid_out = open(out_dir + "valid.txt", "w+")
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





split("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/corpus_table_tokenized_crop30", "/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/all/", .8, .1, .1)
split("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/corpus_table_tokenized_crop30_balanced", "/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/all_balanced/", .8, .1, .1)