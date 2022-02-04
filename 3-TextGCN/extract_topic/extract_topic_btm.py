"""
step1.1 使用btm建图
"""
import csv
import json
import re
import numpy as np
import math
import tqdm
import threading
import argparse

from collections import Counter
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
import os
import sys
sys.path.append('..')
from settings import DATASET_PATH, CLEAN_CORPUS_PATH, BTM_PATH

useless_hashtags = ['#Election2020', '#election2020', '#Elections2020', '#elections2020',
                    '#2020Election', '#2020election', '#2020Elections', '#2020elections',
                    '#Election', '#election']


class BTM:
    def __init__(self, dataset, target, topic_num=5, step=100):
        self.dataset = dataset
        self.target = target
        self.topic_num = topic_num
        self.step = step
        self.unlabeled_texts = []

        # self.result 所有得出的结果，以字典的形式存储，{key: [topic_word1, topic_word2...]}
        self.result = {}

        self.semaphore = threading.BoundedSemaphore(24)  # 最多允许8个线程同时运行
        self.main()

    # get_hashtags 获取数据集中的hashtag
    def get_hashtags(self):
        print("获取数据集中的所有hashtag")
        # self.hashtags 所有标签
        self.hashtags = []
        with open(f"{CLEAN_CORPUS_PATH}{self.dataset}_{self.target}.txt", 'r', encoding='utf-8')as f:
            lines = f.readlines()
            for line in lines:
                text = line
                hashtag = re.findall('#\w+', text)

                self.hashtags.extend(hashtag)

        # 这里不计算频次过低（出现次数小于等于1次）的hashtag
        # counter = Counter(self.hashtags).items()
        # self.hashtags = [_[0] for _ in counter if _[1] > 1]
        self.hashtags = list(set(self.hashtags))

    # get_unlabeled_texts 获取unlabeled的原始文本
    def get_unlabeled_texts(self):
        print("读取unlabeled原始文本")
        # self.unlabeled_texts 所有的unlabeled的文本
        self.unlabeled_texts = []

        re_url = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)

        with open(f"{DATASET_PATH}original/unlabeled/mongo_all.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in tqdm.tqdm(reader):
                _text = re.sub(re_url, '', line['full_text'])     
                self.unlabeled_texts.append(_text)

    # get_texts 获取对应hashtag的文本
    def get_texts(self, hashtag):
        texts = []
        for text in self.unlabeled_texts:
            if hashtag in re.findall('#\w+', text):
                # 去掉空格和回车
                _text = re.sub('#\w+', '', text)
                _text = ' '.join(_text.split())
                texts.append(_text)

        return texts

    def train(self, hashtag):
        # vectorize texts
        vec = CountVectorizer(stop_words='english')
        texts = self.get_texts(hashtag)
        try:
            X = vec.fit_transform(texts).toarray()
        except:
            self.semaphore.release()     #释放
            return

        # get vocabulary
        vocab = np.array(vec.get_feature_names())
        # get biterms
        biterms = vec_to_biterms(X)
        btm = oBTM(num_topics=self.topic_num, V=vocab)

        print("Training Online BTM ..")

        for i in range(0, len(biterms), self.step):  # process chunk of 200 texts
            print(f"第[{i // self.step}]波，共[{math.ceil(len(biterms) / self.step)}]波")
            biterms_chunk = biterms[i:i + self.step]
            btm.fit(biterms_chunk, iterations=10)
        topics = btm.fit_transform(biterms, iterations=10)
        print("Topic coherence ..")
        _topic_summuary = topic_summuary(btm.phi_wz.T, X, vocab, 10)

        _topic_summuary["top_words"] = [list(_) for _ in _topic_summuary["top_words"]]

        _topic_distribution = [0] * self.topic_num
        for _ in topics:
            _topic_distribution[_.argmax()] += 1
        _topic_distribution = [round(float(_) / sum(_topic_distribution), 6) for _ in _topic_distribution]
        _topic_summuary['topic_distribution'] = _topic_distribution

        data = {hashtag: _topic_summuary}
        self.save(data)

        self.semaphore.release()     #释放

    def save(self, data):
        if os.path.exists(f"{BTM_PATH}{self.dataset}_{self.target}.json"):
            with open(f"{BTM_PATH}{self.dataset}_{self.target}.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_data.update(data)
            with open(f"{BTM_PATH}{self.dataset}_{self.target}.json", "w", encoding="utf-8") as f:
                json.dump(old_data, f)
        else:
            with open(f"{BTM_PATH}{self.dataset}_{self.target}.json", "w", encoding="utf-8") as f:
                json.dump(data, f)


    def main(self):
        self.get_unlabeled_texts()
        self.get_hashtags()
        
        if not os.path.exists(f"{BTM_PATH}{self.dataset}_{self.target}.json"):
            with open(f"{BTM_PATH}{self.dataset}_{self.target}.json", 'w') as f:
                json.dump({}, f)
        for hashtag in tqdm.tqdm(self.hashtags):
            print(f"当前hashtag={hashtag}")
            if hashtag in useless_hashtags:
                continue
            # if hashtag exists
            with open(f"{BTM_PATH}{self.dataset}_{self.target}.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if hashtag in old_data:
                    continue
            try:
                # 多线程处理
                self.semaphore.acquire()   #加锁
                t = threading.Thread(target=self.train, args=(hashtag,)) # 这个逗号不能省略
                t.start()
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', type=str, required=True, help="eg: SDwH")
    parser.add_argument('--target', type=str, required=True, help="eg: trump")
    opt = parser.parse_args()
    
    # btm = BTM("SDwH", "trump")
    BTM(opt.dataset, opt.target)
