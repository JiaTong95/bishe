"""
    Step 1 数据预处理，删掉数据文本中的特殊符号、网址等，全部置为小写等
"""
import os
import re
from collections import Counter
from collections import defaultdict
import numpy as np
import csv
import tqdm
from settings import CLEAN_CORPUS_PATH, LABEL_PATH, DATASET_PATH

# class StringProcess 处理字符串类
class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?#\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        # self.nlp 语法模型
        self.nlp = None

    def clean_str(self, string):
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result


def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])

        
class CorpusProcess:
    def __init__(self, dataset, fname, text_col="Tweet"):
        self.text_col = text_col
        self.fread = open(f"{DATASET_PATH}{dataset}/{fname}", 'r', encoding='utf-8')
        self.fwrite = open(f"{CLEAN_CORPUS_PATH}{dataset}_{fname}", 'w', encoding='utf-8')

        self.clean_text()

        self.fread.close()
        self.fwrite.close()

    # 对源文件中的数据进行处理，输出为对应的csv
    def clean_text(self):
        reader = csv.DictReader(self.fread)
        writer = csv.DictWriter(self.fwrite, fieldnames=reader.fieldnames)
        writer.writeheader()

        sp = StringProcess()
        # word_list 所有单词列表
        word_list = []
        # word_set 所有单词集合
        word_set = set()
        # doc_len_list 文档长度
        doc_len_list = []

        # 词频小于5的做丢弃处理
        for line in tqdm.tqdm(reader):
            text = line[self.text_col]
            text = sp.clean_str(text)
            text = sp.remove_stopword(text)
            word_list.extend(text.split())
        #!TODO 这里的问题在于，原代码是将train和test合并到一起处理的，
        # 所以词频小于5是train+test都小于5才删掉，如果分开处理的话，删掉的词会更多些
        for word, value in Counter(word_list).items():
            if value < 3:
                continue
            word_set.add(word)

        # 重置读文件
        self.fread.seek(0)
        reader = csv.DictReader(self.fread)
        # 数据过滤+写文件
        for line in tqdm.tqdm(reader):
            d = line.copy()
            text = line[self.text_col]
            text = sp.clean_str(text)
            text = sp.remove_stopword(text)
            text = remove_less_word(text, word_set)
            d[self.text_col] = text
            writer.writerow(d)
            doc_len_list.append(len(text.split()))
        
        print("文本平均长度：", np.mean(doc_len_list))
        print("文本数量：", len(doc_len_list))
        print("单词个数：", len(word_set))

# combine_csv_to_txt 把源数据整理成不含有标签的txt文件
# 这个txt文件是为了以后的建图做准备，所以它是不需要标签的
def combine_csv_to_txt(train_file, test_file, target):
    corpus = []
    for file_name in [train_file, test_file]:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                corpus.append(line['Tweet'] + '\n')
    with open(f"{CLEAN_CORPUS_PATH}{dataset}_{target}.txt", 'w', encoding='utf-8') as f:
        f.writelines(corpus)

    labels = []
    with open(train_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            ID = line['ID']
            Stance = line['Stance']
            s = f"{ID}\ttrain\t{Stance}\t{target}\n"
            labels.append(s)
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            ID = line['ID']
            Stance = line['Stance']
            s = f"{ID}\ttest\t{Stance}\t{target}\n"
            labels.append(s)
    with open(f"{LABEL_PATH}{dataset}_{target}.txt", 'w', encoding='utf-8') as f:
        f.writelines(labels)
    print(f"合并{target}的train,test文件，总行数为{len(corpus)}")


    
if __name__ == '__main__':
    params_list = [("SDwH", "trump"), 
                   ("SDwH", "biden"), 
                   ("PStance", "trump"), 
                   ("PStance", "biden"), 
                   ("PStance", "bernie"),
                   ("semeval16", "a")]
    for dataset, target in params_list:
        CorpusProcess(dataset=dataset, fname=f"{target}_train.csv")
        CorpusProcess(dataset=dataset, fname=f"{target}_test.csv")
        combine_csv_to_txt(train_file=f"{CLEAN_CORPUS_PATH}{dataset}_{target}_train.csv",
                           test_file=f"{CLEAN_CORPUS_PATH}{dataset}_{target}_test.csv",
                           target=target)
    
    # CorpusProcess("/extend/jt_2/0-dataset/original/unlabeled", "mongo_all.csv", 'full_text')