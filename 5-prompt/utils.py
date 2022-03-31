import argparse
import time
import torch
import re
import json
from beautifultable import BeautifulTable, BTRowCollection
from sklearn import metrics


def timer(function):
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


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed yelp_dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default="semeval16")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--mask', action="store_true")  # 默认不输入就是False
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=80)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--target', type=str, default="fm")
    parser.add_argument('--topic_by', type=str, default="",
                        help="btm, vae or (empty)")
    parser.add_argument('--label_words_id', type=int, default=4)
    parser.add_argument('--template_id', type=int, default=4)

    return parser.parse_args()

def cal_nil_f1(output_logits, targets):
    pred = []
    y = []
    outputs = [max(t[0], t[1]) for t in output_logits]

    for i in range(0, len(outputs), 3):
        max_score = max(outputs[i], outputs[i+1], outputs[i+2])
        if outputs[i] == max_score:
            pred.append(0)
        elif outputs[i+1] == max_score:
            pred.append(1)
        elif outputs[i+2] == max_score:
            pred.append(2)
        
        if targets[i] == 1:
            y.append(0)
        elif targets[i+1] == 1:
            y.append(1)
        elif targets[i+2] == 1:
            y.append(2)
        
    assert len(pred) == len(y) == len(targets) // 3
    return cal_f1(pred, y)

def cal_f1(outputs, targets, two_class=True):
    if two_class:
        labels = [0, 2]
    else:
        labels = [0, 1, 2]
    micro_f = metrics.f1_score(targets, outputs, labels=labels, average='micro')
    f_avg = metrics.f1_score(targets, outputs, labels=labels, average='macro')
    f_favor = metrics.f1_score(targets, outputs, labels=[2], average='macro')
    f_against = metrics.f1_score(targets, outputs, labels=[0], average='macro')
    f_none = metrics.f1_score(targets, outputs, labels=[1], average='macro')

    table = BeautifulTable()
    rows = BTRowCollection(table)
    rows.append(["micro_f", round(micro_f*100, 2)])
    rows.append(["macro_f", round(f_avg*100, 2)])
    rows.append(["f_favor", round(f_favor*100, 2)])
    rows.append(["f_against", round(f_against*100, 2)])
    rows.append(["f_none", round(f_none*100, 2)])
    # print(table)
    return {"micro_f": micro_f,
            "macro_f": f_avg,
            "f_favor": f_favor,
            "f_against": f_against,
            "f_none": f_none,
            "(mi+ma)/2": (micro_f + f_avg)/2}


class get_topic_words:
    def __init__(self) -> None:
        self.no = 0
        self.yes = 0

    def main(self, text, opt):
        if opt.topic_by == "":
            return text
        elif opt.topic_by == "btm":
            with open(f"/extend/bishe/3-TextGCN/data/btm/{opt.dataset}_{opt.target}.json", 'r', encoding='utf-8') as f:
                d = json.load(f)
        elif opt.topic_by == "vae":
            with open(f"/extend/bishe/3-TextGCN/data/vae/{opt.dataset}_{opt.target}.json", 'r', encoding='utf-8') as f:
                d = json.load(f)
        else:
            raise Exception(
                "topic_by error, please choose from [btm, vae and (empty)]")

        hashtags = re.findall('#\w+', text)
        for hashtag in hashtags:
            if hashtag not in d:
                self.no += 1
                continue
            self.yes += 1
            topic_distribution = d[hashtag]["topic_distribution"]
            topic_words = d[hashtag]["top_words"]
            words = topic_words[topic_distribution.index(
                max(topic_distribution))]
            text += " " + " ".join(words)
        return text


def get_senticnet_n_hop(all_seed_words, n):
    assert n > 0
    # 先定义几个种子词，然后用senticnet做 N-hop，拓展之后作为label_words
    from senticnet.senticnet import SenticNet
    sn = SenticNet()

    expanded_seed_words = all_seed_words.copy()

    for label, seed_words in enumerate(all_seed_words):
        for _h in range(0, n):
            new_words = seed_words.copy()
            for word in seed_words:
                new_words.extend(sn.semantics(word))
            # 去重
            seed_words = list(set(new_words))
        expanded_seed_words[label] = seed_words
    return expanded_seed_words


def get_label_words(lid=1):
    if lid == 1:
        return [["bad"],
                ["nothing"],
                ["good"]]
    elif lid == 2:
        return [["against"],
                ["nothing"],
                ["favor"]]
    elif lid == 3:
        return [["against"],
                ["neutral"],
                ["favor"]]
    elif lid == 4:
        all_seed_words = [["against"],
                          ["neutral"],
                          ["favor"]]
        return get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 5:
        return [["negative"],
                ["neutral"],
                ["positive"]]
    elif lid == 6:
        all_seed_words = [["negative"],
                          ["neutral"],
                          ["positive"]]
        return get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 7:
        return [["wrong", "bad", "different"],
                ["neutral", "unique", "cool"],
                ["right", "good", "great"]]
    elif lid == 8:
        all_seed_words = [["wrong", "bad", "stupid"],
                          ["neutral", "unique", "cool"],
                          ["beautiful", "good", "great"]]
        return get_senticnet_n_hop(all_seed_words=all_seed_words, n=2)
    elif lid == 9:
        return [["agree"],
                ["okay"],
                ["disagree"]]
    elif lid == 10:
        return [["likes"],
                ["none"],
                ["hates"]]
    elif lid == 11:
        return [["same"],
                ["different"]]
    elif lid == 12:
        return [["same", "alike", "equivalent", "identical", "equal", "compatible" ,"consistent", "coherent"],
                ["different", "disparate", "unlike", "diverse", "various", "inconsistent", "conflicting", "incompatible", "discrepant"]]
    else:
        raise Exception("label_words_id error, please choose id between 1~3")


def get_prompt_template(tokenizer, tid=1):
    from openprompt.prompts import ManualTemplate, MixedTemplate
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
    elif tid == 9:
        # 
        return ManualTemplate(
            text='{"placeholder":"text_a"}. It {"mask"} {"placeholder":"text_b"}',
            tokenizer=tokenizer,
        )
    elif tid == 10:
        # 
        return ManualTemplate(
            text='{"placeholder":"text_a"}. It is {"mask"} with {"placeholder":"text_b"}',
            tokenizer=tokenizer,
        )
    elif tid == 11:
        # [X1 [SEP] X2]. The two sentence's attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. The two sentence\'s attitude to {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 12:
        # [X1] [X2]. The two sentence's attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. The two sentence\'s attitude to {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 101:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to trump is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 102:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to republican candidate is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 103:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to republican presidential candidate is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 104:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to candidate is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 105:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to high building is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 106:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to radical candidate is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 107:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to liberal republican is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 108:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to dreamer is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 109:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to charismatic individual is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 110:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to political outsider is {"mask"}.',
            tokenizer=tokenizer,
        )
    elif tid == 111:
        # [X]. Its attitude to [Target] is [MASK].
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to businessman is {"mask"}.',
            tokenizer=tokenizer,
        )
    else:
        raise Exception("template_id error, please choose id between 1~4")
