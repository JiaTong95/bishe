# 思路：ipet
# ipet联合训练思路，使用M个模板，分N_stage个阶段单独训练，然后把TestSet的一部分当做unlabel数据，分阶段拼接到TrainSet上
# N_split = 10 代表把unlabel数据随机平均分成10份，每阶段在TrainSet后面拼接1/10的unlabel，使用投票决定这部分的标签(y_true)
# N_stage = 3 代表分3阶段训练
# 相当于总共拼接了3/10的数据在TrainSet后面
# 联合训练完成之后，再分别用自己的模型进行test
#【注】本代码在开始第二阶段以后训练模型的时候，可能还有参数共享的bug(模型重置参数失败)，导致loss下降不正常，目前还没解决。。

import os
import pickle
import torch
import tqdm
import argparse
import random
import copy
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

# 打包模型
class Model_wrapper:
    def __init__(self, model,
                       tid,
                       lid,
                       template,
                       verbalizer,
                       trainset,
                       testset,
                       loss_func,
                       optimizer) -> None:
        self.model = model
        self.tid = tid
        self.lid = lid
        self.template = template
        self.verbalizer = verbalizer
        self.trainset = trainset
        self.testset = testset
        self.loss_func = loss_func
        self.optimizer = optimizer
    
    # 重置模型，使参数不共享
    def reset_model(self, plm, opt):
        print("重置模型参数")
        self.model = PromptForClassification(plm=plm,
                                             template=self.template,
                                             verbalizer=self.verbalizer,
                                             freeze_plm=False).to(opt.device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)



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

        d = {"AGAINST": 0, "NONE": 1, "FAVOR": 2, '-1': 0, '0': 1, '1': 2}
        self.dataset = {}
        for split in ["train", "test"]:  # ["train", "validation", "test"]:
            self.dataset[split] = []
            with open(f"{DATASET_PATH}/{self.opt.dataset}/{self.opt.target}_{split}.csv", 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in tqdm.tqdm(reader):
                    _input_example = InputExample(guid=line["ID"],
                                                  text_a=line["Tweet"].lower(),
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
            self.fname_bert_model = f'pretrained_model/{plm_name}.bert_model'
            self.fname_bert_for_mask = f'pretrained_model/{plm_name}.bert_for_mask'
            self.fname_tokenizer = f'pretrained_model/{plm_name}.tokenizer'

            # bert_config
            self.bert_config = BertConfig.from_pretrained(plm_path)

            # bert_model
            if os.path.exists(self.fname_bert_model):
                self.bert_model = pickle.load(open(self.fname_bert_model, 'rb'))
            else:
                self.bert_model = BertModel.from_pretrained(
                    plm_path, config=self.bert_config)
                pickle.dump(self.bert_model, open(self.fname_bert_model, 'wb'))

            # bert_for_mask
            if os.path.exists(self.fname_bert_for_mask):
                self.bert_for_mask = pickle.load(
                    open(self.fname_bert_for_mask, 'rb'))
            else:
                self.bert_for_mask = BertForMaskedLM.from_pretrained(
                    plm_path, config=self.bert_config)
                pickle.dump(self.bert_for_mask, open(
                    self.fname_bert_for_mask, 'wb'))

            # bert_tokenizer
            if os.path.exists(self.fname_tokenizer):
                self.tokenizer = pickle.load(open(self.fname_tokenizer, 'rb'))
            else:
                self.tokenizer = self.tokenizer = BertTokenizer.from_pretrained(
                    plm_path)
                pickle.dump(self.tokenizer, open(self.fname_tokenizer, 'wb'))

            # Wrapper
            self.WrapperClass = MLMTokenizerWrapper
            self.wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=self.opt.max_seq_length,
                                                         decoder_max_length=3,
                                                         tokenizer=self.tokenizer,
                                                         truncate_method="head")
            self.plm = self.bert_for_mask
        elif plm_name.startswith('roberta'):
            pass

    def new_plm(self):
        return pickle.load(open(self.fname_bert_for_mask, 'rb'))

    # 定义Verbalizer
    def Define_Verbalizer_List(self):
        self.promptVerbalizer_list = {}
        for lid in [1, 2, 3, 4, 5, 6, 7, 8]:
            self.promptVerbalizer_list[lid] = ManualVerbalizer(num_classes=3,
                                                               label_words=get_label_words(lid=lid),
                                                               tokenizer=self.tokenizer)

    # 定义prompt模板
    def Define_PromptTemplate_List(self):
        self.promptTemplate_list = {}
        for tid in [1, 2, 3, 4, 5, 6, 7, 8]:
            self.promptTemplate_list[tid] = get_prompt_template(tokenizer=self.tokenizer, tid=tid)

    # 定义所有模型
    @ timer
    def Define_all_models(self, tids_lids):
        # 准备好所有模型, 打包起来
        self.model_list = []
        for model_index, (tid, lid) in enumerate(tids_lids):
            # prompt相关
            template = self.promptTemplate_list[tid]
            verbalizer = self.promptVerbalizer_list[lid]

            # 模型设置
            model = PromptForClassification(plm=self.new_plm(),
                                            template=template,
                                            verbalizer=verbalizer,
                                            freeze_plm=False).to(self.opt.device)
            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)

            # 打包
            mymodel = Model_wrapper(model=model,
                                    tid=tid,
                                    lid=lid,
                                    template=template,
                                    verbalizer=verbalizer,
                                    trainset=self.dataset["train"].copy(),
                                    testset=self.dataset["test"].copy(),
                                    loss_func=loss_func,
                                    optimizer=optimizer)
            self.model_list.append(mymodel)
        print("所有模型准备完毕")

    # 获取dataloader
    def get_dataloader(self, model_index, mode):
        trainset = self.model_list[model_index].trainset
        testset = self.model_list[model_index].testset
        
        train_size = int(len(trainset) * 0.9)
        valset = trainset[train_size:]
        trainset = trainset[:train_size]
        
        if mode == 'train':
            dataset = trainset
            shuffle = True
        if mode == 'val':
            dataset = valset
            shuffle = True
        if mode == 'test':
            dataset = testset
            shuffle = False
        
        dataloader = PromptDataLoader(dataset=dataset,
                                      template=self.model_list[model_index].template,
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

    # 联合训练
    def train_ensemble(self, reset_model=False):        
        # 每次训练都加一部分testset数据拼在trainset上去, 分N个阶段
        self.N_stage = 3
        # 每次加unlabeled，如果n_split=10,那每次就是加10分之一
        self.N_split = 10

        test_len = len(self.dataset["test"])
        each_split_len = test_len // self.N_split
        
        # 开始分阶段训练
        for stage in range(self.N_stage):
            # 分别独立训练
            all_preds = []
            for model_index in range(len(self.model_list)):
                # 模型参数重置，使参数不共享
                if reset_model:
                    self.model_list[model_index].reset_model(plm=self.plm, opt=self.opt)
                # 独立训练，保存预测结果，稍后进行投票
                preds = self.train_individual(model_index, stage_training=True)
                all_preds.append(preds)
              
            # 投票，决定出新加的unlabel数据的y_true
            preds = []
            for j in range(len(all_preds[0])):
                vote = {0: 0, 1: 0, 2: 0}
                for i in range(len(all_preds)):
                    pred = all_preds[i][j]
                    vote[pred] += 1
                max_num = max(vote.values())
                pred = []
                for key in vote.keys():
                    if vote[key] == max_num:
                        pred.append(key)
                preds.append(random.choice(pred))

            # 拼接unlabelset
            for model_index in range(len(self.model_list)):
                # 然后在trainset后面加上随机模板预测的unlabelset, 从unlabelset的第start_i行到第end_i行
                start_i, end_i = each_split_len * stage, each_split_len * stage + each_split_len
                for i in range(start_i, end_i):
                    pred = preds[i]
                    new_line = copy.deepcopy(self.dataset["test"][i])
                    new_line.label = pred
                    self.model_list[model_index].trainset.append(new_line)
            print(f"==========完成stage{stage+1}训练==========")
        
        # 最终测试
        for model_index in range(len(self.model_list)):
            # 模型参数重置，使参数不共享
            if reset_model:
                self.model_list[model_index].reset_model(plm=self.new_plm(), opt=self.opt)
            self.train_individual(model_index, stage_training=False)


    # 训练模型
    @ timer
    def train_individual(self, model_index, stage_training=True):
        print(f"当前模型tid={self.model_list[model_index].tid}, lid={self.model_list[model_index].lid}")          
        train_dataloader = self.get_dataloader(model_index, 'train')
        val_dataloader = self.get_dataloader(model_index, 'val')
        test_dataloader = self.get_dataloader(model_index, 'test')

        best_macro_score = 0
        for epoch in range(self.opt.max_epoch):
            # self.model_list[model_index].model.train()
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(self.opt.device)
                labels = inputs['label']

                logits = self.model_list[model_index].model(inputs)

                loss = self.model_list[model_index].loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                self.model_list[model_index].optimizer.step()
                self.model_list[model_index].optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(
                        epoch, tot_loss/(step+1)), flush=True)
            
            if stage_training:
                # 根据验证集最好的结果，选出一个最好的epoch，确定unlabeled数据的标签
                val_result, _ = self.eval(model_index, val_dataloader)
                if val_result['macro_f'] > best_macro_score:
                    best_macro_score = val_result['macro_f']
                    test_result, best_test_labels = self.eval(model_index, test_dataloader)
            else:
                test_result, best_test_labels = self.eval(model_index, test_dataloader)
                print(test_result)
        return best_test_labels

    def eval(self, model_index, dataloader):
        # self.model_list[model_index].model.eval()
        allpreds, alllabels = [], []
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.opt.device)
            labels = inputs['label']

            logits = self.model_list[model_index].model(inputs)

            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        result = cal_f1(y_pred=allpreds, y_true=alllabels)
        return result, allpreds

    def main(self):
        self.Get_Dataset()
        self.Load_Pretrained_Language_Model()
        self.Define_Verbalizer_List()
        self.Define_PromptTemplate_List()
        self.Define_all_models(tids_lids=[(1, 1), (4, 4), (6, 8), (7, 2), (8, 6)])

        self.train_ensemble(reset_model=True)


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
    opt = parser.parse_args()

    print(opt)

    trainer = Trainer(opt)
    trainer.main()
