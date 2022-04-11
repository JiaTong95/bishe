# 用learn to compare 的思路把三分类变成二分类

import csv
import re
import random
import copy

DATASET_PATH = "/extend/bishe/0-dataset"

def Transfer_data_from_3_to_2(dataset, target):
    print(f"转换为NLI数据格式...")
    # 读取数据
    train_lines, test_lines = [], []
    with open(f"{DATASET_PATH}/{dataset}/{target}_train.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            train_lines.append(line)
    with open(f"{DATASET_PATH}/{dataset}/{target}_test.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            test_lines.append(line)
    
    # 为每一个label随机选出N个句子
    N = 1
    random_train_lines = train_lines.copy()
    random.shuffle(random_train_lines)
    AGAINST_sentences, NONE_sentences, FAVOR_sentences = [], [], []
    for i, line in enumerate(random_train_lines):
        if (line['Stance'] == 'AGAINST' or line['Stance'] == '-1') and len(AGAINST_sentences) < N:
            AGAINST_sentences.append(line['Tweet'])
        if (line['Stance'] == 'NONE' or line['Stance'] == '0') and len(NONE_sentences) < N:
            NONE_sentences.append(line['Tweet'])
        if (line['Stance'] == 'FAVOR' or line['Stance'] == '1') and len(FAVOR_sentences) < N:
            FAVOR_sentences.append(line['Tweet'])
    
    # 在训练集和测试集中，根据每句话的hashtag，拼一条相同标签的训练数据在后面
    new_train_lines, new_test_lines = [], []
    idx = 0
    for i, line in enumerate(train_lines):
        _, Target, Tweet, Stance = line['ID'], line['Target'], line['Tweet'], line['Stance']
        new_line0 = copy.deepcopy(line)
        new_line1 = copy.deepcopy(line)
        new_line2 = copy.deepcopy(line)
        new_line0['Tweet'] += ' [SEP] ' + random.choice(AGAINST_sentences)
        new_line1['Tweet'] += ' [SEP] ' + random.choice(NONE_sentences)
        new_line2['Tweet'] += ' [SEP] ' + random.choice(FAVOR_sentences)
        new_line0['ID'], new_line1['ID'], new_line2['ID'] = i, i+1, i+2
        i += 3
        if Stance == 'AGAINST' or Stance == '-1':
            new_line0['Stance'] = 'SAME'
            new_line1['Stance'] = 'DIFFERENT'
            new_line2['Stance'] = 'DIFFERENT'
        elif Stance == 'NONE' or Stance == '0':
            new_line0['Stance'] = 'DIFFERENT'
            new_line1['Stance'] = 'SAME'
            new_line2['Stance'] = 'DIFFERENT'
        elif Stance == 'FAVOR' or Stance == '1':
            new_line0['Stance'] = 'DIFFERENT'
            new_line1['Stance'] = 'DIFFERENT'
            new_line2['Stance'] = 'SAME'

        new_train_lines.append(new_line0)
        new_train_lines.append(new_line1)
        new_train_lines.append(new_line2)
    
    for line in test_lines:
        _, Target, Tweet, Stance = line['ID'], line['Target'], line['Tweet'], line['Stance']
        new_line0 = copy.deepcopy(line)
        new_line1 = copy.deepcopy(line)
        new_line2 = copy.deepcopy(line)
        new_line0['Tweet'] += ' [SEP] ' + random.choice(AGAINST_sentences)
        new_line1['Tweet'] += ' [SEP] ' + random.choice(NONE_sentences)
        new_line2['Tweet'] += ' [SEP] ' + random.choice(FAVOR_sentences)
        new_line0['ID'], new_line1['ID'], new_line2['ID'] = i, i+1, i+2
        i += 3
        if Stance == 'AGAINST' or Stance == '-1':
            new_line0['Stance'] = 'SAME'
            new_line1['Stance'] = 'DIFFERENT'
            new_line2['Stance'] = 'DIFFERENT'
        elif Stance == 'NONE' or Stance == '0':
            new_line0['Stance'] = 'DIFFERENT'
            new_line1['Stance'] = 'SAME'
            new_line2['Stance'] = 'DIFFERENT'
        elif Stance == 'FAVOR' or Stance == '1':
            new_line0['Stance'] = 'DIFFERENT'
            new_line1['Stance'] = 'DIFFERENT'
            new_line2['Stance'] = 'SAME'

        new_test_lines.append(new_line0)
        new_test_lines.append(new_line1)
        new_test_lines.append(new_line2)
    
    # 写新数据
    with open(f"{DATASET_PATH}/{dataset}/{target}_nli_train.csv", 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Target', 'Tweet', 'Stance'])
        writer.writeheader()
        for line in new_train_lines:
            writer.writerow(line)
    with open(f"{DATASET_PATH}/{dataset}/{target}_nli_test.csv", 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Target', 'Tweet', 'Stance'])
        writer.writeheader()
        for line in new_test_lines:
            writer.writerow(line)


if __name__ == "__main__":
    Transfer_data_from_3_to_2(dataset="acl-14-short-data", target="")