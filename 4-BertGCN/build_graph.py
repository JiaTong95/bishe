import os
import random
import numpy as np
import pickle as pkl
import csv
import json
import scipy.sparse as sp
from collections import Counter
from math import log
from sklearn import svm
from sklearn.utils import shuffle
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import argparse


CORPUS_PATH = "data/corpus"
IND_PATH = "data/ind"
BTM_PATH = "/extend/bishe/3-TextGCN/data/btm"
VAE_PATH = "/extend/bishe/3-TextGCN/data/vae"


class Build_Graph:
    def __init__(self, opt):
        self.opt = opt
        self.prefix = self.opt.dataset + '_' + self.opt.target  # 'SDwH_trump'

        self.Get_Word_Vectors()
        self.Get_Train_and_Test_Lines(shuffle=True)
        self.Get_Labels()
        self.Get_Indexes()

        self.Build_Vocab()

        self.Get_Train_Feature()
        self.Get_Test_Feature()
        self.Get_ALL_Feature()

        # 最后存成两个矩阵
        # A 存为 adj
        # A + A' 存为 adj_topic
        # 然后train的时候代码逻辑是一样的，只不过就是A不同。
        self.Get_Doc_Heterogeneous_Graph()
        self.Get_Graph()
        if self.opt.topic_by != "":
            self.Get_Topic_Graph()

    def Get_Word_Vectors(self):
        # word_vector_file = '/extend/bishe/pretrained_models/glove.42B.300d.txt'
        # # word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
        # 未写loadWord2Vec函数
        # _, embd, self.word_vector_map = loadWord2Vec(word_vector_file)
        # self.word_embeddings_dim = len(embd[0])
        self.word_embeddings_dim = 300
        self.word_vector_map = {}

    def Get_Train_and_Test_Lines(self, shuffle=False):
        self.train_lines, self.test_lines = [], []
        self.train_ids, self.test_ids = [], []
        self.doc_content_list, self.label_lines = [], []

        with open(f"{CORPUS_PATH}/{opt.dataset}_{opt.target}.clean.txt", 'r', encoding='utf-8') as f:
            clean_lines = f.readlines()
        with open(f"{CORPUS_PATH}/{opt.dataset}_{opt.target}.labels.txt", 'r', encoding='utf-8') as f:
            label_lines = f.readlines()

        text_and_label = []
        for text, label in zip(clean_lines, label_lines):
            text_and_label.append((text, label))
        if shuffle == True:
            random.shuffle(text_and_label)

        for text, label in text_and_label:
            self.doc_content_list.append(text)
            self.label_lines.append(label.split('\t')[2])
            if "train" in label:
                self.train_lines.append(text)
                self.train_ids.append(label.split('\t')[0])
            if "test" in label:
                self.test_lines.append(text)
                self.test_ids.append(label.split('\t')[0])

        self.train_size = len(self.train_lines)
        self.test_size = len(self.test_lines)
        self.val_size = int(self.train_size*0.1)
        print(f"train_size={self.train_size}, test_size={self.test_size}")

    def Get_Labels(self):
        # 获取标签列表
        # self.label_lines 长度为train_size + test_size
        # self.label_list 长度为3(polarities_dim)

        # 获取标签集合
        label_set = set()
        for line in self.label_lines:
            label_set.add(line)
        self.label_list = list(label_set)

        label_list_str = '\n'.join(self.label_list)
        with open(f"{IND_PATH}/{self.prefix}_labels.txt", 'w') as f:
            f.write(label_list_str)

    def Get_Indexes(self):
        with open(f"{IND_PATH}/{self.prefix}.train.index", 'w') as f:
            for line in self.train_ids:
                f.write(line + '\n')
        with open(f"{IND_PATH}/{self.prefix}.test.index", 'w') as f:
            for line in self.test_ids:
                f.write(line + '\n')

    def Build_Vocab(self):
        # word_freq 词频
        word_freq = {}
        # word_set 词集合
        word_set = set()
        # word_list 词列表(未去重)
        word_list = []
        for line in self.doc_content_list:
            words = line.split()
            for word in words:
                word_list.append(word)

        word_set = set(word_list)
        word_freq = dict(Counter(word_list))
        # vocab 词表(去重过的)
        self.vocab = list(word_set)
        self.vocab_size = len(self.vocab)

        vocab_str = '\n'.join(self.vocab)
        with open(f"{IND_PATH}/{self.prefix}_vocab.txt", 'w') as f:
            f.write(vocab_str)

        # word_doc_list 词在哪一行出现过
        word_doc_list = {}
        for i in range(len(self.doc_content_list)):
            doc_words = self.doc_content_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    word_doc_list[word].append(i)
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        # word_doc_freq 词出现过几次
        self.word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            self.word_doc_freq[word] = len(doc_list)

        # word_id_map 词映射id, word2id
        self.word_id_map = {}
        for i in range(self.vocab_size):
            self.word_id_map[self.vocab[i]] = i

    def Get_Train_Feature(self):
        row_x = []
        col_x = []
        data_x = []

        for i in range(self.train_size - self.val_size):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = self.doc_content_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    # print(doc_vec)
                    # print(np.array(word_vector))
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

        # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            self.train_size - self.val_size, self.word_embeddings_dim))

        y = []
        for i in range(self.train_size - self.val_size):
            label = self.label_lines[i]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        y = np.array(y)
        print(f"x.shape={x.shape}, y.shape={y.shape}")

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.x", 'wb') as f:
            pkl.dump(x, f)

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.y", 'wb') as f:
            pkl.dump(y, f)

    def Get_Test_Feature(self):
        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(self.test_size):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = self.doc_content_list[i + self.train_size]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                           shape=(self.test_size, self.word_embeddings_dim))

        ty = []
        for i in range(self.test_size):
            label = self.label_lines[i]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ty.append(one_hot)
        ty = np.array(ty)
        print(f"tx.shape={tx.shape}, ty.shape={ty.shape}")

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.tx", 'wb') as f:
            pkl.dump(tx, f)

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.ty", 'wb') as f:
            pkl.dump(ty, f)

    def Get_ALL_Feature(self):
        # allx: the the feature vectors of both labeled and unlabeled training instances
        # (a superset of x)
        # unlabeled training instances -> words

        word_vectors = np.random.uniform(-0.01, 0.01,
                                         (self.vocab_size, self.word_embeddings_dim))

        for i in range(self.vocab_size):
            word = self.vocab[i]
            if word in self.word_vector_map:
                vector = self.word_vector_map[word]
                word_vectors[i] = vector

        row_allx = []
        col_allx = []
        data_allx = []

        for i in range(self.train_size):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])
            doc_words = self.doc_content_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        for i in range(self.vocab_size):
            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i + self.train_size))
                col_allx.append(j)
                data_allx.append(word_vectors.item((i, j)))

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix(
            (data_allx, (row_allx, col_allx)), shape=(self.train_size + self.vocab_size, self.word_embeddings_dim))

        ally = []
        for i in range(self.train_size):
            label = self.label_lines[i]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(self.vocab_size):
            one_hot = [0 for l in range(len(self.label_list))]
            ally.append(one_hot)
        ally = np.array(ally)

        print(f"allx.shape={allx.shape}, ally.shape={ally.shape}")
        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.allx", 'wb') as f:
            pkl.dump(allx, f)

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.ally", 'wb') as f:
            pkl.dump(ally, f)

    def Get_Doc_Heterogeneous_Graph(self):
        '''
        Doc word heterogeneous graph
        '''
        # word co-occurence with context windows
        window_size = 20
        # windows 划分窗口,比如窗口大小是3,[1,2,3,4,5]划分为[1,2,3],[2,3,4],[3,4,5]三个窗口
        self.windows = []

        for doc_words in self.doc_content_list:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                self.windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    self.windows.append(window)
                    # print(window)

        # self.word_window_freq 词在窗口中出现的频率
        self.word_window_freq = {}
        for window in self.windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in self.word_window_freq:
                    self.word_window_freq[window[i]] += 1
                else:
                    self.word_window_freq[window[i]] = 1
                appeared.add(window[i])

        # self.word_pair_count 两个单词在同一窗口中出现的次数
        self.word_pair_count = {}
        for window in self.windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = self.word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in self.word_pair_count:
                        self.word_pair_count[word_pair_str] += 1
                    else:
                        self.word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in self.word_pair_count:
                        self.word_pair_count[word_pair_str] += 1
                    else:
                        self.word_pair_count[word_pair_str] = 1

    def Get_Graph(self):
        self.row = []
        self.col = []
        self.weight = []

        # pmi as weights
        num_window = len(self.windows)

        for key in self.word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = self.word_pair_count[key]
            word_freq_i = self.word_window_freq[self.vocab[i]]
            word_freq_j = self.word_window_freq[self.vocab[j]]
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue
            self.row.append(self.train_size + i)
            self.col.append(self.train_size + j)
            self.weight.append(pmi)
        
        # word vector cosine similarity as weights

        '''
        for i in range(vocab_size):
            for j in range(vocab_size):
                if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                    vector_i = np.array(word_vector_map[vocab[i]])
                    vector_j = np.array(word_vector_map[vocab[j]])
                    similarity = 1.0 - cosine(vector_i, vector_j)
                    if similarity > 0.9:
                        print(vocab[i], vocab[j], similarity)
                        row.append(train_size + i)
                        col.append(train_size + j)
                        weight.append(similarity)
        '''

        node_size = self.train_size + self.vocab_size + self.test_size
        adj = sp.csr_matrix((self.weight, (self.row, self.col)), shape=(node_size, node_size))

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.adj", 'wb') as f:
            pkl.dump(adj, f)

    def Get_Topic_Graph(self):
        topic_row = []
        topic_col = []
        topic_weight = []

        # doc word frequency
        doc_word_freq = {}

        for doc_id in tqdm(range(len(self.doc_content_list))):
            doc_words = self.doc_content_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = self.word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1
        if self.opt.topic_by == "btm":
            filename = f"{BTM_PATH}/{self.opt.dataset}_{self.opt.target}.json"
        if self.opt.topic_by == "vae":
            filename = f"{VAE_PATH}/{self.opt.dataset}_{self.opt.target}.json"
        with open(filename, 'r', encoding='utf-8') as f:
            topic_file = json.load(f)

        for i in tqdm(range(len(self.doc_content_list))):
            doc_words = self.doc_content_list[i]
            words = doc_words.split()
            doc_word_set = set()
            temp_j = {}
            for word in words:
                # graph 矩阵A，是用PMI和tfidf做的权重的矩阵
                if word in doc_word_set:
                    continue
                j = self.word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < self.train_size:
                    self.row.append(i)
                else:
                    self.row.append(i + self.vocab_size)
                self.col.append(self.train_size + j)
                idf = log(1.0 * len(self.doc_content_list) /
                          self.word_doc_freq[self.vocab[j]])
                self.weight.append(freq * idf)
                doc_word_set.add(word)

                # topic_graph 矩阵A'，跟前两天我用的那个代码是一样的
                if word in topic_file:
                    item = topic_file[word]
                    for k, word_list in enumerate(item['top_words']):
                        topic_num = len(item['top_words'])
                        distribution = item['topic_distribution'][k] * topic_num
                        # distribution = 1
                        for topic_word in word_list:
                            if topic_word not in self.word_id_map:
                                continue
                            tj = self.word_id_map[topic_word] + self.train_size
                            if tj not in temp_j:
                                temp_j[tj] = 0
                            temp_j[tj] += distribution
            if i < self.train_size:
                ti = i
            else:
                ti = i + self.vocab_size
            for tj, val in temp_j.items():
                topic_row.append(ti)
                topic_col.append(tj)
                topic_weight.append(val)

        g = {}
        for i in range(len(self.row)):
            g[(self.row[i], self.col[i])] = self.weight[i]
        for i in range(len(topic_row)):
            if (topic_row[i], topic_col[i]) not in g:
                g[(topic_row[i], topic_col[i])] = 0
            g[(topic_row[i], topic_col[i])] += topic_weight[i]

        # d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        # for tw in weight:
        #     if tw >= 6:
        #         d[6]+=1
        #     else:
        #         d[int(tw)] += 1
        # for key in d.keys():
        #     d[key] = d[key] / len(weight)
        #     d[key] = round(d[key], 2)
        # print("graph",d)
        #
        # d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        # for tw in topic_weight:
        #     if tw >= 6:
        #         d[6]+=1
        #     else:
        #         d[int(tw)] += 1
        # for key in d.keys():
        #     d[key] = d[key] / len(topic_weight)
        #     d[key] = round(d[key], 2)
        # print("topic_graph",d)
        topic_row = []
        topic_col = []
        topic_weight = []
        for (tr, tc), tv in g.items():
            topic_row.append(tr)
            topic_col.append(tc)
            topic_weight.append(tv)

        node_size = self.train_size + self.vocab_size + self.test_size
        adj_topic = sp.csr_matrix(
            (topic_weight, (topic_row, topic_col)), shape=(node_size, node_size))

        with open(f"data/ind/{self.opt.dataset}_{self.opt.target}.adj_topic", 'wb') as f:
            pkl.dump(adj_topic, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', type=str, required=True, help="eg: SDwH")
    parser.add_argument('--target', type=str, required=True, help="eg: trump")
    parser.add_argument('--topic_by', type=str,
                        required=True, help="eg: btm,vae")

    opt = parser.parse_args()

    Build_Graph(opt)
