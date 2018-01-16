# This Python file uses the following encoding: utf-8

import random
import operator
import math
import os
from constants import *
from functools import reduce
import re

# Various functions for generating permuted data for real/fake classifier

# SHUFFLE PERMUTING

def shuffle_permute_file(input_path, out_path, min_percent, max_percent):
    if os.path.isfile(out_path):
        print("permuted file %s already exists" % out_path)
    else:
        output = open(out_path, "w")
        for line in open(input_path):
            words = line.split()
            original_length = len(words)
            words = remove_pad(words)
            words, punc_map = remove_punc(words)
            words = shuffle_line(words, random.uniform(min_percent, max_percent))
            words = replace_punc(words, punc_map)
            words = add_pad(words, original_length)
            new_line = reduce(lambda x, y: x + " " + y, words)
            output.write(new_line + "\n")
    output.close()

def shuffle_line(a_list, percent):
    """shuffle n elements of a list"""
    n = int(math.floor(percent * len(a_list)))
    if n < 2 and len(a_list) >= 2:
        n = 2
    idx = range(len(a_list))
    random.shuffle(idx)
    idx = idx[:n]
    mapping = dict((idx[i], idx[i - 1]) for i in range(n))
    return [a_list[mapping.get(x, x)] for x in range(len(a_list))]








# SWAP PERMUTING

def swap_permute(input_path):
    out_path = input_path[:-4] + "-permuted.txt"
    output = open(out_path, "w")
    for line in open(input_path):
        words = line.split()
        original_length = len(words)
        words = remove_pad(words)
        words, punc_map = remove_punc(words)
        words = swap(words, random.randint(1, 1))
        words = replace_punc(words, punc_map)
        words = add_pad(words, original_length)
        new_line = reduce(lambda x, y: x + " " + y, words)
        output.write(new_line + "\n")
    output.close()


def swap(words, n):
    for _ in range(n):
        start = random.randint(0, max(len(words) - 2, 1))
        end = random.randint(start, max(len(words) - 1, 1))
        split_point = random.randint(start, end)
        before = words[0:start]
        first_half = words[0:split_point]
        after = words[split_point:]
        after.extend(before)
        words = after
    return words




# SWAP PERMUTING BY PUNC

def permute_file_by_punc(input_path, out_path, rlb, rub):
    output = open(out_path, "w")
    for line in open(input_path):
        new_line = permute_by_punc(line, rlb, rub)
        output.write(new_line + "\n")
    output.close()


def permute_by_punc(line, rlb, rub):
    words = line.split()
    original_length = len(words)
    words = remove_pad(words)
    words = apostrophe_s(words)
    chunks, punc_map = chunk_at_punc(words)
    r = random.randint(rlb, rub)
    try:
        r1 = random.randint(0, min(r, len(chunks)-1))
    except ValueError:
        r1 = 0
    r2 = r - r1
    chunks = swap_chunks(chunks, r1 + 1)
    chunks = swap_split_chunks(chunks, r2)
    chunks = replace_punc(chunks, punc_map)
    words = [item for sublist in chunks for item in sublist]
    words = add_pad(words, original_length)
    new_line = reduce(lambda x, y: x + " " + y, words)
    return new_line


def chunk_at_punc(words):
    punc = [".", ",", ";", ":", "?", "!", "\"", "\'", "`", "``", "\'\'", "(", ")", "[", "]", "‘", "’"]
    chunks = []
    curr_chunk = []
    punc_map = {}
    n = 0
    for i in range(len(words)):
        if words[i] in punc:
            if curr_chunk:
                chunks.append(curr_chunk[:])
                curr_chunk = []
                n += 1
            punc_map[n] = [words[i]]
            n += 1
        else:
            curr_chunk.append(words[i])
    if curr_chunk:
        chunks.append(curr_chunk[:])
    return chunks, punc_map

def swap_chunks(chunks, r):
    idx = range(len(chunks))
    random.shuffle(idx)
    idx = idx[:r]
    try:
        mapping = dict((idx[i], idx[i - 1]) for i in range(r))
    except IndexError:
        return chunks
    return [chunks[mapping.get(x, x)] for x in range(len(chunks))]

def swap_split_chunks(chunks, r2):
    for _ in range(r2):
        try:
            i = random.randint(0, len(chunks)-1)
        except ValueError:
            return chunks
        c = chunks[i]
        j = random.randint(0, len(c)-1)
        chunks[i] = c[j:]
        chunks[i].extend(c[0:j])
    return chunks


def apostrophe_s(words):
    prev = ""
    new_list = []
    for w in words:
        if re.match(".*\'.*", w) and new_list:
            new_list[-1] = new_list[-1] + " " + w
        else:
            new_list.append(w)
    return new_list







def remove_punc(words):
    punc = [".", ",", ";", ":", "?", "!", "\"", "\'", "`", "``", "\'\'", "(", ")", "[", "]", "‘"]
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


def remove_pad(words):
    return [x for x in words if x != START and x != STOP]


def add_pad(words, n):
    words.insert(0, START)
    while len(words) < n:
        words.append(STOP)
    return words







################# MAIN ####################

