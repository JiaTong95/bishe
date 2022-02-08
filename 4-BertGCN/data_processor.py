from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str
import sys
import os
import argparse
import csv
from tqdm import tqdm

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

parser = argparse.ArgumentParser(description="Run .")
parser.add_argument('--dataset', type=str, required=True, help="eg: SDwH")
parser.add_argument('--target', type=str, required=True, help="eg: trump")
opt = parser.parse_args()

DATASET_PATH = f"/extend/bishe/0-dataset/"
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/corpus"):
    os.mkdir("data/corpus")
if not os.path.exists("data/dat"):
    os.mkdir("data/dat")
if not os.path.exists("data/ind"):
    os.mkdir("data/ind")
if not os.path.exists("data/pkl"):
    os.mkdir("data/pkl")

doc_content_list = []
label_list = []
with open(f"{DATASET_PATH}/{opt.dataset}/{opt.target}_train.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for line in reader:
        ID = line['ID']
        Stance = line['Stance']
        Tweet = line['Tweet']
        target = line['Target']

        doc_content_list.append(Tweet)
        label_list.append(f"{ID}\ttrain\t{Stance}\t{target}")

with open(f"{DATASET_PATH}/{opt.dataset}/{opt.target}_test.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for line in reader:
        ID = line['ID']
        Stance = line['Stance']
        Tweet = line['Tweet']
        target = line['Target']

        doc_content_list.append(Tweet)
        label_list.append(f"{ID}\ttest\t{Stance}\t{target}")

with open(f"data/corpus/{opt.dataset}_{opt.target}.txt", 'w', encoding='utf-8') as f:
    for line in doc_content_list:
        f.write(line + '\n')
with open(f"data/corpus/{opt.dataset}_{opt.target}.labels.txt", 'w', encoding='utf-8') as f:
    for line in label_list:
        f.write(line + '\n')

word_freq = {}  # to remove rare words

for doc_content in tqdm(doc_content_list):
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in tqdm(doc_content_list):
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    # if doc_str == '':
    # doc_str = temp
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)

with open(f'data/corpus/{opt.dataset}_{opt.target}.clean.txt', 'w') as f:
    f.write(clean_corpus_str)

min_len = 10000
aver_len = 0
max_len = 0

with open(f'data/corpus/{opt.dataset}_{opt.target}.clean.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)

aver_len = 1.0 * aver_len / len(lines)
print('Min_len : ' + str(min_len))
print('Max_len : ' + str(max_len))
print('Average_len : ' + str(aver_len))
