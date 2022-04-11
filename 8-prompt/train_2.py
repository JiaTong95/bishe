# 思路：拼接主题词
# 输入文本后面加上用topic_model提取出来的topic_words

import os
import pickle
import torch
import tqdm
import json
import re
import argparse
from openprompt.plms import load_plm, MLMTokenizerWrapper
from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, AdamW

from util import timer, cal_f1, get_label_words, get_prompt_template, update_result

DATASET_PATH = "/extend/bishe/0-dataset"

BERT_BASE_PATH = "/extend/bishe/pretrained_models/bert-base-uncased"
BERT_LARGE_PATH = "/extend/bishe/pretrained_models/bert-large-uncased"
ROBERTA_BASE_PATH = "/extend/bishe/pretrained_models/roberta-base-uncased"
ROBERTA_LARGE_PATH = "/extend/bishe/pretrained_models/roberta-large-uncased"
# 如果没有手动下载预训练模型，就用下面几行
# BERT_BASE_PATH = "bert-base-uncased"
# BERT_LARGE_PATH = "bert-large-uncased"
# ROBERTA_BASE_PATH = "roberta-base-uncased"
# ROBERTA_LARGE_PATH = "roberta-large-uncased"

class get_topic_words:
    def __init__(self, opt) -> None:
        self.no = 0
        self.yes = 0
        self.opt = opt

    def main(self, text):
        if opt.topic_by == "":
            return text
        elif opt.topic_by == "btm":
            with open(f"/extend/bishe/3-TextGCN/data/btm/{self.opt.dataset}_{self.opt.target}.json", 'r', encoding='utf-8') as f:
                d = json.load(f)
        elif opt.topic_by == "vae":
            with open(f"/extend/bishe/3-TextGCN/data/vae/{self.opt.dataset}_{self.opt.target}.json", 'r', encoding='utf-8') as f:
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

class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt
        if not os.path.exists('pretrained_model'):
            os.mkdir('pretrained_model')
        # torch.manual_seed(self.opt.seed)
        # torch.cuda.manual_seed(self.opt.seed)

    # 获取数据集
    def Get_Dataset(self):
        import csv
        G = get_topic_words(self.opt)
        d = {"AGAINST": 0, "NONE": 1, "FAVOR": 2, '-1': 0, '0': 1, '1': 2}
        self.dataset = {}
        for split in ["train", "test"]:  # ["train", "validation", "test"]:
            self.dataset[split] = []
            with open(f"{DATASET_PATH}/{self.opt.dataset}/{self.opt.target}_{split}.csv", 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in tqdm.tqdm(reader):
                    _input_example = InputExample(guid=line["ID"],
                                                  text_a=G.main(line["Tweet"].lower()),
                                                  text_b=line["Target"],
                                                  label=d[line["Stance"]],
                                                  meta={'text': line["Tweet"].lower()})
                    self.dataset[split].append(_input_example)

    # 加载预训练语言模型
    @timer
    def Load_Pretrained_Language_Model(self):
        # 本函数中用pickle存起来的目的是，将加载预训练模型的时间从缩短为1秒左右，否则要几分钟

        plm_name, plm_path = 'bert-base-uncased', BERT_BASE_PATH
        # plm_name, BERTPATH = 'bert-large-uncased', BERT_LARGE_PATH
        # plm_name, ROBERTAPATH = 'roberta-base-uncased', ROBERTA_BASE_PATH
        # plm_name, ROBERTAPATH = 'roberta-large-uncased', ROBERTA_LARGE_PATH

        if plm_name.startswith('bert'):
            fname_bert_model = f'pretrained_model/{plm_name}.bert_model'
            fname_bert_for_mask = f'pretrained_model/{plm_name}.bert_for_mask'
            fname_tokenizer = f'pretrained_model/{plm_name}.tokenizer'

            # bert_config
            self.bert_config = BertConfig.from_pretrained(plm_path)

            # bert_model
            if os.path.exists(fname_bert_model):
                self.bert_model = pickle.load(open(fname_bert_model, 'rb'))
            else:
                self.bert_model = BertModel.from_pretrained(
                    plm_path, config=self.bert_config)
                pickle.dump(self.bert_model, open(fname_bert_model, 'wb'))

            # bert_for_mask
            if os.path.exists(fname_bert_for_mask):
                self.bert_for_mask = pickle.load(
                    open(fname_bert_for_mask, 'rb'))
            else:
                self.bert_for_mask = BertForMaskedLM.from_pretrained(
                    plm_path, config=self.bert_config)
                pickle.dump(self.bert_for_mask, open(
                    fname_bert_for_mask, 'wb'))

            # bert_tokenizer
            if os.path.exists(fname_tokenizer):
                self.tokenizer = pickle.load(open(fname_tokenizer, 'rb'))
            else:
                self.tokenizer = self.tokenizer = BertTokenizer.from_pretrained(
                    plm_path)
                pickle.dump(self.tokenizer, open(fname_tokenizer, 'wb'))

            # Wrapper
            self.WrapperClass = MLMTokenizerWrapper
            self.wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=self.opt.max_seq_length,
                                                         decoder_max_length=3,
                                                         tokenizer=self.tokenizer,
                                                         truncate_method="head")
            self.plm = self.bert_for_mask
        elif plm_name.startswith('roberta'):
            pass

    # 定义Verbalizer
    def Define_Verbalizer(self):
        # 获取label_words
        self.label_words = get_label_words(
            lid=self.opt.label_words_id, tokenizer=self.tokenizer, remove_UNK=self.opt.remove_UNK)
        print("self.label_words", self.label_words)

        # 构造verbalizer
        self.promptVerbalizer = ManualVerbalizer(num_classes=3,
                                                 label_words=self.label_words,
                                                 tokenizer=self.tokenizer,
                                                 )

        # 所有的label_words
        self.all_label_words_list = []
        for i in self.label_words:
            for j in i:
                self.all_label_words_list.append(j)
        print("all_label_words_list:", self.all_label_words_list)

        # 所有的label_words对应的id
        self.all_label_words_ids = self.tokenizer.convert_tokens_to_ids(
            self.all_label_words_list)
        print("self.all_label_words_id", self.all_label_words_ids)

    # 定义prompt模板
    def Define_PromptTemplate(self):
        self.promptTemplate = get_prompt_template(
            tokenizer=self.tokenizer, tid=self.opt.template_id)

    # 组合构建为PromptModel类
    def Define_PromptModel(self):
        self.prompt_model = PromptForClassification(plm=self.plm,
                                                    template=self.promptTemplate,
                                                    verbalizer=self.promptVerbalizer,
                                                    freeze_plm=False).to(self.opt.device)

    # 获取dataloader
    def Get_Dataloader(self, mode):
        if mode == "train":
            shuffle = True
            dataset = self.dataset["train"]
        elif mode == "test":
            shuffle = False
            dataset = self.dataset["test"]

        dataloader = PromptDataLoader(dataset=dataset,
                                      template=self.promptTemplate,
                                      tokenizer=self.tokenizer,
                                      tokenizer_wrapper_class=self.WrapperClass,
                                      max_seq_length=self.opt.max_seq_length,
                                      decoder_max_length=3,
                                      batch_size=self.opt.batch_size,
                                      shuffle=shuffle,
                                      teacher_forcing=False,
                                      predict_eos_token=False,
                                      truncate_method="head")
        return dataloader

    # 训练模型
    @ timer
    def train(self):
        self.train_dataloader = self.Get_Dataloader(mode="train")
        self.test_dataloader = self.Get_Dataloader(mode="test")

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.prompt_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.prompt_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.opt.learning_rate)

        for epoch in range(self.opt.max_epoch):
            # self.prompt_model.train()
            tot_loss = 0
            for step, inputs in enumerate(self.train_dataloader):
                inputs = inputs.to(self.opt.device)
                labels = inputs['label']

                logits = self.prompt_model(inputs)

                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(
                        epoch, tot_loss/(step+1)), flush=True)
            self.test()

    def test(self):
        # self.prompt_model.eval()
        allpreds, alllabels = [], []
        for step, inputs in enumerate(self.test_dataloader):
            inputs = inputs.to(self.opt.device)
            labels = inputs['label']

            logits = self.prompt_model(inputs)

            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        result = cal_f1(y_pred=allpreds, y_true=alllabels)
        print(result)
        # update_result(result=result, opt=self.opt)

    def main(self):
        self.Get_Dataset()
        self.Load_Pretrained_Language_Model()
        self.Define_Verbalizer()
        self.Define_PromptTemplate()
        self.Define_PromptModel()

        self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run .")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default="semeval16")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--hid_dim', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--max_seq_length', type=int, default=80)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--target', type=str, default="fm")

    parser.add_argument('--label_words_id', type=int, default=4)
    parser.add_argument('--template_id', type=int, default=4)
    parser.add_argument('--remove_UNK', type=int, default=0,
                        help="移除不存在于tokenizer中的单词。\
                              因为在手动定义的label_word中，或者在senticnet中，有可能有不存在于tokenizer中的单词。\
                              remove_UNK为0时，表示不从label_words中删除掉这些单词，\
                              remove_UNK为1时，表示从label_words中删除掉这些单词，默认为0")

    parser.add_argument('--topic_by', type=str, default='btm', choices=['btm', 'vae'])
    opt = parser.parse_args()

    print(opt)

    trainer = Trainer(opt)
    trainer.main()
