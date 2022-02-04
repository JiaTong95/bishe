import logging
import argparse
import math
import os
import sys
import time
from torch.functional import _return_counts
import tqdm
import json
import random
import numpy as np

from pytorch_transformers import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, MyDataset
from models import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# class Instructor 模型训练构造器
class Instructor:
    def __init__(self, opt):
        # self.opt 模型参数设置
        self.opt = opt

        # bert模型需要单独设置
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            self.pretrained_bert_state_dict = bert.state_dict()
        else:
            # tokenizer 分词器
            tokenizer = build_tokenizer(
                dataset_name=opt.dataset,  # opt.dataset_file['test'][opt.target]],
                max_seq_len=opt.max_seq_len,
                dat_fname=f"dat/{opt.dataset}_{opt.target}_tokenizer.dat",
                opt=opt,
                rebuild=False)
            # embedding matrix 词嵌入矩阵
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname=f"dat/{opt.embed_dim}_{opt.dataset}_{opt.target}_embedding_matrix.dat",
                rebuild=False)
            print(f"embedding_matrix.shape={embedding_matrix.shape}")
            # self.model 模型
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        # self.trainset 训练集
        self.trainset = MyDataset(opt.dataset, opt.target, "train", tokenizer, opt)
        # self.testset 测试集
        self.testset = MyDataset(opt.dataset, opt.target, "test", tokenizer, opt)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def get_log_str(self, params, mode=''):
        acc, micro_f, f_avg, f_favor, f_against, f_none = params
        s = ""
        
        if mode=='0':
            # mode0 写入结果文件中
            s += f"model_name={self.opt.model_name} " + \
                 f"dataset={self.opt.dataset} " + \
                 f"target={self.opt.target} " + \
                 f"learning_rate={self.opt.learning_rate} " + \
                 f"dropout={self.opt.dropout} " + \
                 f"num_epoch={self.opt.num_epoch} " + \
                 f"batch_size={self.opt.batch_size} " + \
                 "\n"
            s += f"micro_f={micro_f:.4}, f_avg={f_avg:.4}, f_against={f_against:.4}, f_favor={f_favor:.4}, f_none={f_none:.4}\n"
            s += f"{micro_f:.4}\t{f_avg:.4}\t{f_against:.4}\t{f_favor:.4}\t{f_none:.4}\n"
        if mode=='1':
            # mode1 训练过程中每一步的logger
            s += f"micro_f={micro_f:.4}, f_avg={f_avg:.4}, f_against={f_against:.4}, f_favor={f_favor:.4}, f_none={f_none:.4}"
        return s

    # _evaluate 训练指标
    def _evaluate(self, data_loader, mode="val"):
        # correct_num 预测正确数量 total_num 数据总数量
        correct_num, total_num = 0, 0
        # t_targets_all 实际结果标签 t_outputs_all 模型输出标签
        t_targets_all, t_outputs_all = None, None
        
        # 将模型设置为评估模式，参数不更新，对应self.model.train()
        self.model.eval()

        # 模型计算输出
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                correct_num += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                total_num += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        # 计算指标
        acc = correct_num / total_num
        labels = [i for i in range(opt.polarities_dim)]

        #'micro':通过先计算总体的TP，FN和FP的数量，再计算F1
        #'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
        micro_f = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=labels,
                              average='micro')
        f_avg = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=labels,
                                 average='macro')
        f_favor = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[2],
                                   average='macro')
        f_against = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0],
                                     average='macro')
        f_none = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[1],
                                  average='macro')
        params = (acc, micro_f, f_avg, f_favor, f_against, f_none)
        if mode=="test":
            # 写训练日志
            with open(f"logs/_result.txt", "a+") as file:
                s0 = self.get_log_str(params=params, mode="0")
                file.write(s0)
            # =====更新最佳结果=====
            best_macro = {"macro_f1": float(f_avg), "micro_f1": float(micro_f), 
                          "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                          "learning_rate": self.opt.learning_rate, "num_epoch": self.opt.num_epoch,
                          "batch_size": self.opt.batch_size, "dropout": self.opt.dropout, "seed": self.opt.seed}
            best_micro = {"micro_f1": float(micro_f), "macro_f1": float(f_avg),
                          "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                          "learning_rate": self.opt.learning_rate, "num_epoch": self.opt.num_epoch,
                          "batch_size": self.opt.batch_size, "dropout": self.opt.dropout, "seed": self.opt.seed}
            if not os.path.exists('../result.json'):
                with open(f"../result.json", "w") as file:
                    json.dump({}, file)
            with open(f"../result.json", "r") as file:
                _result = json.load(file)
            print(_result)
            if self.opt.dataset not in _result:
                _result[self.opt.dataset] = {}
            if self.opt.target not in _result[self.opt.dataset]:
                _result[self.opt.dataset][self.opt.target] = {}
            if self.opt.model_name not in _result[self.opt.dataset][self.opt.target]:
                _result[self.opt.dataset][self.opt.target][self.opt.model_name] = {"macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
            # 按照macro更新
            if _result[self.opt.dataset][self.opt.target][self.opt.model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
                _result[self.opt.dataset][self.opt.target][self.opt.model_name]["macro"] = best_macro
            # 按照micro更新
            if _result[self.opt.dataset][self.opt.target][self.opt.model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
                _result[self.opt.dataset][self.opt.target][self.opt.model_name]["micro"] = best_micro
            with open(f"../result.json", "w") as file:
                json.dump(_result, file, indent=2)
            print(_result)
            # =====更新最佳结果=====end
        s1 = self.get_log_str(params=params, mode="1")
        logger.info(s1)
        # print(f"{round(micro_f*100, 2)}\t{round(f_avg*100, 2)}\t{round(f_against*100,2)}\t{round(f_favor*100,2)}\t{round(f_none*100,2)}")
        return acc, micro_f, (f_avg, f_favor, f_against, f_none)

    # _print_args 打印模型参数
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # _reset_params 重置模型参数
    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                self.model.bert.load_state_dict(self.pretrained_bert_state_dict)

    # _train 训练
    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, fid=-1):
        """
            criterion 损失函数
            optimizer 优化器
            train_data_loader 训练集data_loader
            val_data_loader 验证集data_loader
            fid K折交叉验证时，第x折
        """
        # max_val_acc 最大准确率 max_val_f1 最大f1
        max_val_acc, max_val_f1 = 0, 0
        # global_step 全局训练步数，打印日志用
        global_step = 0
        # path 最优模型路径
        path = None

        for epoch in range(self.opt.num_epoch):
            logger.info('epoch: {}'.format(epoch))
            # correct_num 预测正确数量, total_num 数据总数量, loss_total 总loss
            correct_num, total_num, loss_total = 0, 0, 0
            # 将模型设置为训练模式，参数更新，对应self.model.eval()
            self.model.train()

            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                correct_num += (torch.argmax(outputs, -1) == targets).sum().item()
                total_num += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = correct_num / total_num
                    train_loss = loss_total / total_num
                    logger.info(
                        'fold:{}/{} epoch:{}/{} |loss: {:.4f}| acc: {:.4f}'.format(fid + 1, opt.cross_val_fold,
                                                                                   epoch + 1, opt.num_epoch, train_loss,
                                                                                   train_acc))

            val_acc, val_f1, _= self._evaluate(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            # 按照最优的F1保存模型
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = f'state_dict/{self.opt.model_name}_{self.opt.target}_val_f1{round(val_f1, 4)}'
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

        return path

    def run(self):
        if opt.mode == "test":
            self.test()
            return
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        # 交叉验证
        # if (opt.cross_val_fold <= 1):
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True,
                                        drop_last=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False,
                                        drop_last=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False,
                                        drop_last=True)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()   
        test_acc, test_f1, _= self._evaluate(test_data_loader, "test")
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        # else:
        #     test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=True,
        #                                   drop_last=True)
        #     valset_len = len(self.trainset) // self.opt.cross_val_fold
        #     splitedsets = random_split(self.trainset, tuple([valset_len] * (self.opt.cross_val_fold - 1) + [
        #         len(self.trainset) - valset_len * (self.opt.cross_val_fold - 1)]))

        #     all_test_acc, all_test_f1 = [], []
        #     for fid in range(self.opt.cross_val_fold):
        #         logger.info('fold : {}'.format(fid + 1))
        #         logger.info('>' * 100)
        #         trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
        #         valset = splitedsets[fid]
        #         train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True,
        #                                        drop_last=True)
        #         val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False,
        #                                      drop_last=True)

        #         self._reset_params()
        #         best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, fid)

        #         self.model.load_state_dict(torch.load(best_model_path))
        #         test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader, "test")
        #         all_test_acc.append(test_acc)
        #         all_test_f1.append(test_f1)
        #         logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        #     mean_test_acc, mean_test_f1 = np.mean(all_test_acc), np.mean(all_test_f1)
        #     logger.info('>' * 100)
        #     logger.info('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}'.format(mean_test_acc, mean_test_f1))

    def test(self):
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True)

        file_list = os.listdir('state_dict')
        prefix = f"{self.opt.model_name}_{self.opt.target}_val_f1"
        max_val_f1 = 0
        # print(prefix)
        for file_name in file_list:
            # print(file_name)
            if file_name.startswith(prefix):
                val_f1 = float(file_name.split(prefix)[-1])
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    best_model_path = file_name
        self.model.load_state_dict(torch.load("state_dict/" + best_model_path))
        test_acc, test_f1, _ = self._evaluate(test_data_loader, "test")

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='tan', type=str)
    # parser.add_argument('--bidirectional', default=True, type=bool) #parser 传不了bool型的参数
    parser.add_argument('--dataset', default='trump', type=str)
    parser.add_argument('--target', default='trump', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)  # l2正则化 （权重衰减）
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.176, type=float,
                        help='set ratio between 0 and 1 for validation support (default 70-15-15, 15/85=0.176)')
    parser.add_argument('--cross_val_fold', default=0, type=int, help='k-fold cross validation')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    # The following parameters are only valid for the hashtag dataset
    parser.add_argument('--nohashtag', default=0, type=int, help='0:original 1:remove hashtags 2:remove #')
    parser.add_argument('--mode', default="train", type=str,)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # model_classes 模型种类
    model_classes = {'lstm': LSTM,
                     'text_cnn': TextCNN,
                     'atae_lstm': ATAE_LSTM,
                     'ian': IAN,
                     'memnet': MemNet,
                     'aoa': AOA,
                     'bert_spc': BERT_SPC,
                     'aen_bert': AEN_BERT,
                     'lcf_bert': LCF_BERT,
                     'tan':TAN,
                     }

    # input_colses 各个模型输入所需要的列
    input_colses = {
        'lstm': ['text_indices'],
        'text_cnn': ['text_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'aoa': ['text_indices', 'aspect_indices'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'tan': ['text_indices', 'aspect_indices']
    }

    # initializers 初始化算法 uniform 均匀分布 normal 正态分布 orthogonal 正交初始化
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    # optimizers 优化器
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = 'logs/{}-{}.log'.format(opt.dataset, opt.model_name)
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()
    
