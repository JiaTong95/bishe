# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
import json
import re
import csv
import tqdm
import random
from torch.utils.data import Dataset, dataset
from transformers import BertTokenizer

DATASET_DIR = '/extend/bishe/0-dataset'
PRETRAINED_MODEL_DIR = '/extend/bishe/pretrained_models/glove.42B.300d.txt'

# build_tokenizer 创建分词器
def build_tokenizer(dataset_name, max_seq_len, dat_fname, opt, rebuild=True):
    if not rebuild and os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        with open(f"{DATASET_DIR}/{dataset_name}/{opt.target}_train.csv") as f:
            reader = csv.DictReader(f)
            for line in reader:
                Tweet = line['Tweet']#.replace('#', '')
                text += Tweet + ' '
        with open(f"{DATASET_DIR}/{dataset_name}/{opt.target}_test.csv") as f:
            reader = csv.DictReader(f)
            for line in reader:
                Tweet = line['Tweet']#.replace('#', '')
                text += Tweet + ' '

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            if tokens[0] == '.':
                print(tokens[1:])
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, rebuild=True):
    if not rebuild and os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        word_vec = _load_word_vec(PRETRAINED_MODEL_DIR, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class MyDataset(Dataset):
    def __init__(self, dataset_name, target, train_or_test, tokenizer, opt):
        polarity_dict = {'AGAINST':0, 'NONE':1, 'FAVOR':2,
                         '-1':0, '0':1, '1':2}
        with open(f"{DATASET_DIR}/{dataset_name}/{opt.target}_{train_or_test}.csv") as f:
            reader = csv.DictReader(f)
            
            all_data = []
            for line in reader:
                ID, Target, Tweet, Stance = line['ID'], line['Target'], line['Tweet'], line['Stance']
                polarity = polarity_dict[Stance]
                text = Tweet

                text_indices = tokenizer.text_to_sequence(text)
                context_indices = tokenizer.text_to_sequence(text)
                aspect_indices = tokenizer.text_to_sequence(target)

                text_len = np.sum(text_indices != 0)
                aspect_len = np.sum(aspect_indices != 0)
                concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + target + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
                text_bert_indices = tokenizer.text_to_sequence(
                    "[CLS] " + text + " [SEP]")
                aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + target + " [SEP]")

                data = {'concat_bert_indices': concat_bert_indices,
                        'concat_segments_indices': concat_segments_indices,
                        'text_bert_indices': text_bert_indices,
                        'aspect_bert_indices': aspect_bert_indices,
                        'text_indices': text_indices,
                        'context_indices': context_indices,
                        'aspect_indices': aspect_indices,
                        'polarity': polarity,
                        }
                all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)