import os
import pickle
import numpy as np    
import json
from sklearn import metrics
from beautifultable import BeautifulTable, BTRowCollection
from senticnet.senticnet import SenticNet
from openprompt.prompts import ManualTemplate, MixedTemplate

GLOVE_PATH = '/extend/bishe/pretrained_models/glove.42B.300d.txt'

SN = SenticNet()


def timer(function):
    import time
    """
    装饰器函数timer
    """
    def wrapper(*args, **kwargs):
        time_start = time.time()
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("【%s】运行时间：【%s】秒" % (function.__name__, cost_time))
        return res
    return wrapper

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

def build_embedding_matrix(word2idx, dat_fname, embed_dim=300, rebuild=False):
    if not rebuild and os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        word_vec = _load_word_vec(GLOVE_PATH, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

def get_words_from_microsoft(target, topK=8):
    import requests
    instance = target
    smooth = 0.0001
    response = requests.get(url=f'https://concept.research.microsoft.com/api/Concept/ScoreByCross?instance={instance}&topK={topK}&smooth={smooth}',
                            verify=False)
    print(response.json())
    return response.json()

def get_senticnet_n_hop(all_seed_words, n):
    assert n > 0
    # 先定义几个种子词，然后用senticnet做 N-hop，拓展之后作为label_words

    expanded_seed_words = all_seed_words.copy()

    for label, seed_words in enumerate(all_seed_words):
        for _h in range(0, n):
            new_words = seed_words.copy()
            for word in seed_words:
                new_words.extend(SN.semantics(word))
            # 去重
            seed_words = list(set(new_words))
        expanded_seed_words[label] = seed_words
    return expanded_seed_words


# 计算a到b是几跳
def cal_hop(a, b, max_hop=3):
    if a == b:
        return 0
    seed_words = [a]

    for hop in range(0, max_hop):
        new_seed_words = []
        for word in seed_words:
            try:
                new_words = SN.semantics(word)
            except KeyError:
                continue

            new_seed_words.extend(new_words)
            if b in new_seed_words:
                return hop + 1
        seed_words = list(set(new_seed_words))
    return -1


def update_result(self, result, opt):
    mi_ma_2 = result["(mi+ma)/2"]

    tid = f"T{opt.template_id}"
    lid = f"L{opt.label_words_id}"
    for metrics in ["micro_f", "macro_f", "(mi+ma)/2"]:
        with open(f"f_score/{opt.dataset}_{opt.target}_prompt_f.json", "r") as file:
            all_f_score = json.load(file)
        if tid not in all_f_score[metrics]:
            all_f_score[metrics][tid] = {}
        if lid not in all_f_score[metrics][tid]:
            all_f_score[metrics][tid][lid] = {metrics: 0}

        previous_score = all_f_score[metrics][tid][lid][metrics]
        if mi_ma_2 > previous_score:
            all_f_score[metrics][tid][lid] = result
            with open(f"f_score/{opt.dataset}_{opt.target}_prompt_f.json", "w") as file:
                json.dump(all_f_score, file, indent=2)


def cal_f1(y_pred, y_true, two_class=True):
    if two_class:
        labels = [0, 2]
    else:
        labels = [0, 1, 2]
    micro_f = metrics.f1_score(y_true, y_pred, labels=labels, average='micro')
    f_avg = metrics.f1_score(y_true, y_pred, labels=labels, average='macro')
    f_favor = metrics.f1_score(y_true, y_pred, labels=[2], average='macro')
    f_against = metrics.f1_score(y_true, y_pred, labels=[0], average='macro')
    f_none = metrics.f1_score(y_true, y_pred, labels=[1], average='macro')

    table = BeautifulTable()
    rows = BTRowCollection(table)
    rows.append(["micro_f", round(micro_f*100, 2)])
    rows.append(["macro_f", round(f_avg*100, 2)])
    rows.append(["f_favor", round(f_favor*100, 2)])
    rows.append(["f_against", round(f_against*100, 2)])
    rows.append(["f_none", round(f_none*100, 2)])
    # print(table)
    return {"micro_f": round(micro_f*100, 2),
            "macro_f": round(f_avg*100, 2),
            "f_favor": round(f_favor*100, 2),
            "f_against": round(f_against*100, 2),
            "f_none": round(f_none*100, 2),
            "(mi+ma)/2": round((micro_f + f_avg)/2*100, 2)}


def get_label_words(lid=1, tokenizer=None, remove_UNK=False):
    if lid == 1:
        label_words = [["bad"],
                       ["nothing"],
                       ["good"]]
    elif lid == 2:
        label_words = [["against"],
                       ["nothing"],
                       ["favor"]]
    elif lid == 3:
        label_words = [["against"],
                       ["neutral"],
                       ["favor"]]
    elif lid == 4:
        all_seed_words = [["against"],
                          ["neutral"],
                          ["favor"]]
        label_words = get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 5:
        label_words = [["negative"],
                       ["neutral"],
                       ["positive"]]
    elif lid == 6:
        all_seed_words = [["negative"],
                          ["neutral"],
                          ["positive"]]
        label_words = get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 7:
        label_words = [["wrong", "bad", "different"],
                       ["neutral", "unique", "cool"],
                       ["right", "good", "great"]]
    elif lid == 8:
        all_seed_words = [["wrong", "bad", "stupid"],
                          ["neutral", "unique", "cool"],
                          ["beautiful", "good", "great"]]
        label_words = get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 9:
        label_words = [["agree"],
                       ["okay"],
                       ["disagree"]]
    elif lid == 10:
        label_words = [["likes"],
                       ["none"],
                       ["hates"]]
    elif lid == 11:
        label_words = [["same"],
                       ["different"]]
    elif lid == 12:
        label_words = [["same", "alike", "equivalent", "identical", "equal", "compatible", "consistent", "coherent"],
                       ["different", "disparate", "unlike", "diverse", "various", "inconsistent", "conflicting", "incompatible", "discrepant"]]
    else:
        raise Exception("label_words_id 错误，请检查是否超出了定义范围")

    if remove_UNK:
        _label_words = [[] for _ in range(len(label_words))]
        for i, words in enumerate(label_words):
            for word in words:
                if tokenizer._convert_token_to_id(word) != 100:
                    _label_words[i].append(word)
        label_words = _label_words
    return label_words


def get_prompt_template(tokenizer, tid=1):

    if tid == 1:
        # [X] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 2:
        # [X]. It was [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. It was {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 3:
        # [X] is [MASK] of [Target].
        return ManualTemplate(
            text='{"placeholder":"text_a"} is {"mask"} of {"placeholder":"text_b"}.',
            tokenizer=tokenizer,
        )
    elif tid == 4:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 5:
        # [X].I felt the [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. I felt the {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 6:
        # [X]. The [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. The {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 7:
        # [X]. The [Target] made me feel [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. The {"placeholder":"text_b"} made me feel {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 8:
        # [X]. I think [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. I think {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 11:
        # [X1 [SEP] X2]. The two sentence's attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. The two sentence\'s attitude to {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
