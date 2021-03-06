import os
import sys
sys.path.append('..')
from data_processor import StringProcess
from settings import DATASET_PATH, CLEAN_CORPUS_PATH, VAE_PATH
from curses import raw
import pickle
import re
import argparse
from GSM import GSM
from vae_utils import DocDataset
import csv
import json
import torch
import tqdm
import threading


class Extract_topic_words:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.target = args.target
        self.no_below = args.no_below
        self.no_above = args.no_above
        self.num_epochs = args.num_epochs
        self.n_topic = args.n_topic
        self.batch_size = args.batch_size
        self.criterion = args.criterion
        self.auto_adj = args.auto_adj
        self.device = args.device
        self._init_get_hashtags()
        self._init_get_unlabeled_texts()
        self.semaphore = threading.BoundedSemaphore(1)  # 最多允许n个线程同时运行
        self.main()

    # get_hashtags 获取数据集中的hashtag

    def _init_get_hashtags(self):
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
        print(f"共有{len(self.hashtags)}个hashtag")

    # get_unlabeled_texts 获取unlabeled的原始文本
    def _init_get_unlabeled_texts(self):
        print("读取unlabeled原始文本")
        # self.unlabeled_texts 所有的unlabeled的文本
        self.unlabeled_texts = []
        sp = StringProcess()
        re_url = re.compile(
            r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)

        with open(f"{DATASET_PATH}original/unlabeled/mongo_all.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in tqdm.tqdm(reader):
                # 去掉网址
                _text = re.sub(re_url, '', line['full_text'])
                # 去掉@用户
                _text = re.sub('@\w+', '', _text)
                # 清理字符串
                _text = sp.clean_str(_text).strip()
                # 去掉空格和回车
                _text = ' '.join(_text.split())
                self.unlabeled_texts.append(_text)

    def get_txtLines_from_unlabeled_corpus(self, hashtag):
        texts = []
        for text in self.unlabeled_texts:
            if hashtag in re.findall('#\w+', text):
                # 去掉#hashtag
                _text = re.sub('#\w+', '', text)
                # 去掉空格和回车
                _text = ' '.join(_text.split())
                texts.append(_text)
        print(f"话题标签{hashtag}对应的unlabeled文本有{len(texts)}条")
        return texts

    def save_json(self, data):
        if os.path.exists(f"{VAE_PATH}{self.dataset}_{self.target}.json"):
            with open(f"{VAE_PATH}{self.dataset}_{self.target}.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_data.update(data)
            with open(f"{VAE_PATH}{self.dataset}_{self.target}.json", "w", encoding="utf-8") as f:
                json.dump(old_data, f)
        else:
            with open(f"{VAE_PATH}{self.dataset}_{self.target}.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

    def run(self, hashtag):
        try:
            raw_hashtag = hashtag.lstrip("#")
            txtLines = self.get_txtLines_from_unlabeled_corpus(hashtag)
                
            docSet = DocDataset(txtLines=txtLines, no_below=self.no_below, no_above=self.no_above, use_tfidf=False)
            if self.auto_adj:
                no_above = docSet.topk_dfs(topk=20)
                docSet = DocDataset(no_below=self.no_below, no_above=no_above, use_tfidf=False)

            voc_size = docSet.vocabsize
            print('voc size:', voc_size)

            model = GSM(bow_dim=voc_size, n_topic=self.n_topic,
                        taskname=f"{self.dataset}_{self.target}", device=self.device)
            model.train(train_data=docSet, batch_size=self.batch_size, test_data=docSet,
                        num_epochs=self.num_epochs, log_every=10, beta=1.0, criterion=self.criterion)
            model.evaluate(test_data=docSet)

            # 存储topic词和词分布
            topic_words = model.show_topic_words(topK=10)
            topic_distribution = model.show_topic_distribution()
            self.save_json({hashtag: {"top_words": topic_words,
                                    "topic_distribution": topic_distribution}
                            })

            # # 用训练好的模型将文档中的每一句话对应的topic输出
            # txt_lst, embeds = model.get_embed(
            #     train_data=docSet, num=len(txtLines)+1)
            # with open(f'temp/theta/{raw_hashtag}.jsonl', 'w', encoding='utf-8') as wfp:
            #     for t, e in zip(txt_lst, embeds):
            #         _s = json.dumps(list(t)) + '\n' + \
            #             json.dumps([float(_) for _ in e])
            #         wfp.write(_s + '\n')
            # # 存储模型，使用的时候再用load state_dict
            # save_name = f'temp/state_dict/{raw_hashtag}.model'
            # torch.save(model.vae.state_dict(), save_name)
        except:
            pass
        self.semaphore.release()  # 释放
        print("释放。。")

    def main(self):
        if not os.path.exists(f"{VAE_PATH}{self.dataset}_{self.target}.json"):
            with open(f"{VAE_PATH}{self.dataset}_{self.target}.json", 'w') as f:
                json.dump({}, f)

        for hashtag in tqdm.tqdm(self.hashtags):
            # if hashtag exists
            with open(f"{VAE_PATH}{self.dataset}_{self.target}.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if hashtag in old_data:
                    continue
            # 多线程处理
            print("加锁。。")
            self.semaphore.acquire()  # 加锁
            t = threading.Thread(target=self.run, args=(hashtag,))  # 这个逗号不能省略
            t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('GSM topic model')
    parser.add_argument('--dataset', type=str, default='SDwH')
    parser.add_argument('--target', type=str,
                        default='trump', help='Taskname e.g trump')
    parser.add_argument('--no_below', type=int, default=1,
                        help='The lower bound of count for words to keep, e.g 10')
    parser.add_argument('--no_above', type=float, default=0.005,
                        help='The ratio of upper bound of count for words to keep, e.g 0.3')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
    parser.add_argument('--n_topic', type=int, default=5, help='Num of topics')
    parser.add_argument('--bkpt_continue', type=bool, default=False,
                        help='Whether to load a trained model as initialization and continue training.')
    parser.add_argument('--use_tfidf', type=bool, default=False,
                        help='Whether to use the tfidf feature for the BOW input')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default=512)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
    parser.add_argument('--auto_adj', action='store_true',
                        help='To adjust the no_above ratio automatically (default:rm top 20)')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    etw = Extract_topic_words(args)
