import argparse
import time
import torch
import re
import json
from beautifultable import BeautifulTable
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
    parser.add_argument('--dataset', type=str, default="SDwH")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--hid_dim', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--mask', action="store_true")  # 默认不输入就是False
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=80)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--target', type=str, default="trump")
    parser.add_argument('--topic_by', type=str, default="",
                        help="btm, vae or (empty)")
    parser.add_argument('--label_words_id', type=int, default=1)
    parser.add_argument('--template_id', type=int, default=1)

    return parser.parse_args()


def cal_f1(outputs, targets):
    micro_f = metrics.f1_score(targets, outputs, labels=[
                               0, 1, 2], average='micro')
    f_avg = metrics.f1_score(targets, outputs, labels=[
                             0, 1, 2], average='macro')
    f_favor = metrics.f1_score(targets, outputs, labels=[2], average='macro')
    f_against = metrics.f1_score(targets, outputs, labels=[0], average='macro')
    f_none = metrics.f1_score(targets, outputs, labels=[1], average='macro')

    table = BeautifulTable()
    table.append_row(["micro_f", micro_f])
    table.append_row(["macro_f", f_avg])
    table.append_row(["f_favor", f_favor])
    table.append_row(["f_against", f_against])
    table.append_row(["f_none", f_none])
    return {"micro_f": micro_f,
            "macro_f": f_avg,
            "f_favor": f_favor,
            "f_against": f_against,
            "f_none": f_none}


class get_topic_words:
    def __init__(self, ) -> None:
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
            raise Exception("topic_by error, please choose from [btm, vae and (empty)]")

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
        return [["against", "hate"],
                ["nothing", "none", "no feelings"],
                ["in favor of", "support"]]
    else:
        raise Exception("label_words_id error, please choose id between 1~3")


def get_prompt_template(tokenizer, tid=1):
    from openprompt.prompts import ManualTemplate, MixedTemplate
    if tid == 1:
        return ManualTemplate(
            text='{"placeholder":"text_a"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    if tid == 2:
        return ManualTemplate(
            text='{"placeholder":"text_a"}. It was {"mask"}.',
            tokenizer=tokenizer,
        )
    if tid == 3:
        return ManualTemplate(
            text='{"placeholder":"text_a"} is {"mask"} of {"placeholder":"text_b"}.',
            tokenizer=tokenizer,
        )
    if tid == 4:
        return ManualTemplate(
            text='{"placeholder":"text_a"}. Its attitude to {"placeholder":"text_b"} is {"mask"}.',
            tokenizer=tokenizer,
        )
    else:
        raise Exception("template_id error, please choose id between 1~4")
