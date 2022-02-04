"""
step 2 建图
graph V*V V为sentence数量+word数量，V = s + w，0~s是句子节点，s+1~s+w是单词节点
"""
import os
from collections import Counter

import networkx as nx
import json
import itertools
import math
from collections import defaultdict
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import print_graph_detail
from settings import GRAPH_PATH, WORD2ID_PATH, CLEAN_CORPUS_PATH

def get_window(content_lst, window_size):
    """
    找出窗口
    """
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0.):
    if isinstance(content_lst, str):
        content_lst = list(open(content_lst, "r"))
    print("pmi read file len:", len(content_lst))

    pmi_start = time()
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,
                                                                window_size=window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    print("Total number of edges between word:", len(pmi_edge_lst))
    pmi_time = time() - pmi_start
    return pmi_edge_lst, pmi_time


class BuildGraph:
    def __init__(self, dataset, target):
        if not os.path.exists(GRAPH_PATH):
            os.makedirs(GRAPH_PATH)
        if not os.path.exists(WORD2ID_PATH):
            os.makedirs(WORD2ID_PATH)

        self.word2id = dict()  # 单词映射
        self.dataset = dataset
        self.target = target
        print(f"\n==> 现在的数据集是:{dataset}_{target}<==")

        self.g = nx.Graph()

        self.content = f"{CLEAN_CORPUS_PATH}{dataset}_{target}.txt"

        self.get_tfidf_edge()
        self.get_pmi_edge()
        self.save()

    def get_pmi_edge(self):
        pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, window_size=20, threshold=0.0)
        print("pmi time:", self.pmi_time)

        for edge_item in pmi_edge_lst:
            word_indx1 = self.node_num + self.word2id[edge_item[0]]
            word_indx2 = self.node_num + self.word2id[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.g.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        print_graph_detail(self.g)

    def get_tfidf_edge(self):
        # 获得tfidf权重矩阵（sparse）和单词列表
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num + col_ind
                self.g.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst.append(count)

        print_graph_detail(self.g)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        """
        start = time()
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        tfidf_vec = text_tfidf.fit_transform(open(self.content, "r"))

        self.tfidf_time = time() - start
        print("tfidf time:", self.tfidf_time)
        print("tfidf_vec shape:", tfidf_vec.shape)
        print("tfidf_vec type:", type(tfidf_vec))

        self.node_num = tfidf_vec.shape[0]

        # 映射单词
        vocab_lst = text_tfidf["vect"].get_feature_names()
        print("vocab_lst len:", len(vocab_lst))
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind

        self.vocab_lst = vocab_lst

        return tfidf_vec

    def save(self):
        print("total time:", self.pmi_time + self.tfidf_time)
        nx.write_weighted_edgelist(self.g, f"{GRAPH_PATH}{self.dataset}_{self.target}.txt")
        with open(f"{WORD2ID_PATH}{self.dataset}_{self.target}.json", 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False)
        print("\n")


def main():
    params_list = [("SDwH", "trump"), 
                   ("SDwH", "biden"), 
                   ("PStance", "trump"), 
                   ("PStance", "biden"), 
                   ("PStance", "bernie"),
                   ("semeval16", "a")]
    for dataset, target in params_list:
        BuildGraph(dataset=dataset, target=target)

if __name__ == '__main__':
    main()
