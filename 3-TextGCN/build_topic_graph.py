import json
from xml.etree.ElementInclude import default_loader
import networkx as nx
import re
import tqdm
import argparse
from settings import BTM_PATH, VAE_PATH, WORD2ID_PATH, CLEAN_CORPUS_PATH

# 建立一个V*V的矩阵，其中V为语料库大小

class TOPIC_GRAPH:
    def __init__(self, dataset, target, topic_by):
        self.dataset = dataset
        self.target = target
        self.topic_by = topic_by
        if topic_by == "btm":
            self.TOPIC_PATH = BTM_PATH
        if topic_by == "vae":
            self.TOPIC_PATH = VAE_PATH
        self.g = nx.Graph()
        self.g_mask = nx.Graph()
        with open(f"{self.TOPIC_PATH}{self.dataset}_{self.target}.json", 'r', encoding='utf-8') as f:
            self.topic_file = json.load(f)
        with open(f"{WORD2ID_PATH}{self.dataset}_{self.target}.json", 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        with open(f"{CLEAN_CORPUS_PATH}{self.dataset}_{self.target}.txt", 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        v = len(self.texts) + len(self.word2id)
        for i in range(v):
            # 这里是凑维度用，把矩阵的对角线置为0，不然nx.Graph会自动压缩矩阵，到时候维度对不上
            self.g.add_edge(i, i, weight=0)
            self.g_mask.add_edge(i, i, weight=0)
        self.main()

    def main(self):
        print(f"==>{self.dataset}_{self.target}_{self.topic_by}<==")
        for i, line in tqdm.tqdm(enumerate(self.texts)):
            hashtags = re.findall('#\w+', line)
            for hashtag in hashtags:
                if hashtag in self.topic_file:
                    # 计算A'
                    item = self.topic_file[hashtag]
                    for k, word_list in enumerate(item['top_words']):
                        distribution = item['topic_distribution'][k]
                        # distribution = 1
                        for word in word_list:
                            if word not in self.word2id:
                                continue
                            j = self.word2id[word] + len(self.texts)

                            if self.g.has_edge(i, j):
                                value = self.g.get_edge_data(i, j)["weight"] + distribution
                            else:
                                value = distribution
                            self.g.add_edge(i, j, weight=value)

                    # 计算mask
                    # 加句子和hashtag的边
                    j = self.word2id[hashtag] + len(self.texts)
                    if not self.g_mask.has_edge(i, j):
                        self.g_mask.add_edge(i, j, weight=1)

                    # 加句子和hashtag对应的topic_word的边
                    item = self.topic_file[hashtag]
                    for k, word_list in enumerate(item['top_words']):
                        for word in word_list:
                            if word not in self.word2id:
                                continue
                            j = self.word2id[word] + len(self.texts)
                            if not self.g_mask.has_edge(i, j):
                                self.g_mask.add_edge(i, j, weight=1)

        nx.write_weighted_edgelist(self.g, f"{self.TOPIC_PATH}{self.dataset}_{self.target}_graph.txt")
        nx.write_weighted_edgelist(self.g_mask, f"{self.TOPIC_PATH}{self.dataset}_{self.target}_mask.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', type=str, default="SDwH")
    parser.add_argument('--target', type=str, default="trump")
    parser.add_argument('--topic_by', type=str, default="btm")
    opt = parser.parse_args()

    TOPIC_GRAPH(dataset=opt.dataset, target=opt.target, topic_by=opt.topic_by)
    # TOPIC_GRAPH(dataset="SDwH", target="trump", topic_by="btm")
    # TOPIC_GRAPH(dataset="SDwH", target="trump", topic_by="vae")

    # TOPIC_GRAPH(dataset="SDwH", target="biden", topic_by="btm")
    # TOPIC_GRAPH(dataset="SDwH", target="biden", topic_by="vae")

    # TOPIC_GRAPH(dataset="PStance", target="trump", topic_by="btm")
    # TOPIC_GRAPH(dataset="PStance", target="trump", topic_by="vae")
    # TOPIC_GRAPH(dataset="PStance", target="bernie", topic_by="btm")
    # TOPIC_GRAPH(dataset="PStance", target="bernie", topic_by="vae")
    # TOPIC_GRAPH(dataset="PStance", target="biden", topic_by="btm")
    # TOPIC_GRAPH(dataset="PStance", target="biden", topic_by="vae")