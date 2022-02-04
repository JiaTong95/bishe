import os
import warnings
from time import time

import tqdm
import copy
import json
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support

from model import GCN
from utils import accuracy
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import parameter_parser
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed
import datetime

from settings import BTM_PATH, VAE_PATH, GRAPH_PATH, LABEL_PATH, WORD2ID_PATH, CLEAN_CORPUS_PATH

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def timer(function):
    """
    装饰器函数timer
    """
    def wrapper(*args, **kwargs):
        time_start = time()
        res = function(*args, **kwargs)
        cost_time = time() - time_start
        print("【%s】运行时间：【%s】秒" % (function.__name__, cost_time))
        return res

    return wrapper


class PrepareData:
    def __init__(self, opt):
        self.opt = opt
        self.Get_Graph()
        self.Get_Topic_Graph_And_Mask_Graph()
        self.Get_Features()
        self.Get_Trainset_Testset_Split()
        self.Get_Text_Indices()
        self.Get_NClass()

    # Get_Graph 获取图(邻接矩阵)
    def Get_Graph(self):
        self.graph = nx.read_weighted_edgelist(
            f"{GRAPH_PATH}{self.opt.dataset}_{self.opt.target}.txt", nodetype=int)
        print_graph_detail(self.graph)
        # 转换成networkx库可以处理的格式
        adj = nx.to_scipy_sparse_matrix(self.graph,
                                        nodelist=list(
                                            range(self.graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        # 非对称矩阵转换成对称矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

    # Get_Topic_Graph_And_Mask_Graph 获取主题矩阵和mask矩阵(mask是个N*N的矩阵，把除了hashtag以外的都挡住，)
    def Get_Topic_Graph_And_Mask_Graph(self):
        self.topic_graph = None
        self.mask_graph = None

        if self.opt.topic_by == "btm":
            TOPIC_PATH = BTM_PATH
        elif self.opt.topic_by == "vae":
            TOPIC_PATH = VAE_PATH
        else:
            return

        # 获取topic_graph
        _topic_graph = nx.read_weighted_edgelist(
            f"{TOPIC_PATH}{self.opt.dataset}_{self.opt.target}_graph.txt", nodetype=int)
        self.topic_graph = nx.to_scipy_sparse_matrix(_topic_graph,
                                                     nodelist=list(range(_topic_graph.number_of_nodes())),
                                                     weight='weight',
                                                     dtype=np.float32)
        self.topic_graph = preprocess_adj(
            self.topic_graph, is_sparse=True, plus_I=False)

        # 获取mask
        if self.opt.mask == True:
            _mask_graph = nx.read_weighted_edgelist(
                f"{TOPIC_PATH}{self.opt.dataset}_{self.opt.target}_mask.txt", nodetype=int)
            _mask_graph = nx.to_scipy_sparse_matrix(_mask_graph,
                                                    nodelist=list(
                                                        range(_mask_graph.number_of_nodes())),
                                                    weight='weight',
                                                    dtype=np.float32)
            # 非对称矩阵变为对称矩阵
            _mask_graph = _mask_graph + _mask_graph.T.multiply(
                _mask_graph.T > _mask_graph) - _mask_graph.multiply(_mask_graph.T > _mask_graph)
            self.mask_graph = preprocess_adj(
                _mask_graph, is_sparse=True, plus_I=False)

    # Get_Features 获取初始训练特征
    def Get_Features(self):
        """
            (应该是)一个NxN的单位矩阵，转成了稀疏方式存储
        """
        self.feat_dim = self.graph.number_of_nodes()
        row = list(range(self.feat_dim))
        col = list(range(self.feat_dim))
        value = [1.] * self.feat_dim
        shape = (self.feat_dim, self.feat_dim)
        indices = torch.from_numpy(np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        self.features = torch.sparse.FloatTensor(indices, values, shape)

    # Get_Trainset_Testset_Split 获取训练集、验证集和数据集
    def Get_Trainset_Testset_Split(self):
        from sklearn.model_selection import train_test_split

        self.train_list, self.test_list = [], []
        fname = f"{LABEL_PATH}{self.opt.dataset}_{self.opt.target}.txt"
        with read_file(fname, 'r', encoding='utf-8') as f:
            for indx, item in enumerate(f):
                if item.split("\t")[1] in ["train", "training", "20news-bydate-train"]:
                    self.train_list.append(indx)
                else:
                    self.test_list.append(indx)

        self.train_list, self.val_list = train_test_split(self.train_list,
                                                          test_size=self.opt.val_ratio,
                                                          shuffle=True)

    # Get_Text_Indices 获取词标记
    def Get_Text_Indices(self):
        with open(f"{WORD2ID_PATH}{self.opt.dataset}_{self.opt.target}.json", 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.text_indices = []
        with open(f"{CLEAN_CORPUS_PATH}{self.opt.dataset}_{self.opt.target}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_indices = []
                for word in line.split():
                    line_indices.append(self.word2id[word])
                self.text_indices.append(line_indices)

    # Get_NClass 获取标签种类数量
    def Get_NClass(self):
        y_fname = f"{LABEL_PATH}{self.opt.dataset}_{self.opt.target}.txt"
        y = np.array(pd.read_csv(y_fname,
                                      sep="\t",
                                      header=None)[2])
        y2id = {label: indx for indx, label in enumerate(set(y))}
        self.y = [y2id[label] for label in y]
        self.nclass = len(y2id)


class Instructor:
    def __init__(self, opt):
        self.Init_parameters(opt)
        self.data = PrepareData(opt)

    # 初始化参数
    def Init_parameters(self, opt):
        self.dataset = opt.dataset
        self.target = opt.target
        self.device = opt.device
        self.dropout = opt.dropout
        self.early_stopping = opt.early_stopping
        self.hid_dim = opt.hid_dim
        self.learning_rate = opt.learning_rate
        self.mask = opt.mask
        self.max_epoch = opt.max_epoch
        self.seed = opt.seed
        self.topic_by = opt.topic_by
        self.v = opt.v
        self.val_ratio = opt.val_ratio

        if self.topic_by == "":
            self.model_name = "gcn"
        else:
            self.model_name = "gcn_" + self.topic_by

    # 深拷贝数据，节省磁盘IO时间
    def Copy_data(self):
        copy_data = copy.deepcopy(self.data)
        self.feat_dim = copy_data.feat_dim
        self.nclass = copy_data.nclass
        self.adj = copy_data.adj
        self.features = copy_data.features
        self.y = copy_data.y
        self.train_list = copy_data.train_list
        self.val_list = copy_data.val_list
        self.test_list = copy_data.test_list
        self.topic_graph = copy_data.topic_graph
        self.mask_graph = copy_data.mask_graph

    # Convert_tensor 转换tensor
    def Convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.y = torch.tensor(self.y).long().to(self.device)
        self.train_list = torch.tensor(self.train_list).long().to(self.device)
        self.val_list = torch.tensor(self.val_list).long().to(self.device)
        self.test_list = torch.tensor(self.test_list).long().to(self.device)
        if self.topic_graph != None:
            self.topic_graph = self.topic_graph.to(self.device)
        if self.mask_graph != None:
            self.mask_graph = self.mask_graph.to(self.device)

    # Set_seed 设置种子
    def Set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Get_Model 获取模型
    def Get_Model(self):
        self.model = GCN(feat_dim=self.feat_dim,
                         hid_dim=self.hid_dim,
                         nclass=self.nclass,
                         dropout=self.dropout,
                         v=self.v)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    # Record 记录实验结果
    def Record(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        for key, value_list in self.result.items():
            min_v = f"min:{float(np.min(value_list)):.4}"
            max_v = f"max:{float(np.max(value_list)):.4}"
            mean = f"mean:{float(np.mean(value_list)):.4}"
            value_list = [f"{float(x):.4}" for x in value_list]
            value_list.extend([min_v, max_v, mean])
            table.add_column(key, value_list)
        print(table)
        suffix = ""
        if self.topic_by != "":
            suffix += "_" + self.topic_by
        if self.mask == True:
            suffix += "_mask"
        with open(f"logs/{self.dataset}_{self.target}{suffix}.log", 'w') as f:
            f.write(table.__str__())

    def train(self):
        self.earlystopping = EarlyStopping(self.early_stopping)
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(x=self.features, adj=self.adj,
                                topic_graph=self.topic_graph, mask_graph=self.mask_graph)
            loss = self.criterion(logits[self.train_list],
                                  self.y[self.train_list])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_list)

            desc = dict(**{"epoch": epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            if self.earlystopping(val_desc["val_loss"]):
                # print(f"epoch={epoch}, earlystopping...")
                break

    @torch.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.y[x])
            acc = accuracy(logits[x],
                           self.y[x])
            precision, recall, _, _ = precision_recall_fscore_support(y_true=self.y[x].cpu(),
                                                                      y_pred=torch.argmax(
                                                                          logits[x], -1).cpu(),
                                                                      labels=[
                                                                          0, 1, 2],
                                                                      average='macro')
            micro_f1 = f1_score(y_true=self.y[x].cpu(),
                                y_pred=torch.argmax(logits[x], -1).cpu(),
                                labels=[0, 1, 2],
                                average='micro')
            macro_f1 = f1_score(y_true=self.y[x].cpu(),
                                y_pred=torch.argmax(logits[x], -1).cpu(),
                                labels=[0, 1, 2],
                                average='macro')

            f_against = f1_score(y_true=self.y[x].cpu(),
                                 y_pred=torch.argmax(logits[x], -1).cpu(),
                                 labels=[0],
                                 average='macro')
            f_favor = f1_score(y_true=self.y[x].cpu(),
                               y_pred=torch.argmax(logits[x], -1).cpu(),
                               labels=[2],
                               average='macro')
            f_none = f1_score(y_true=self.y[x].cpu(),
                              y_pred=torch.argmax(logits[x], -1).cpu(),
                              labels=[1],
                              average='macro')
            f_avg = (f_against + f_favor + f_none) / 3.0

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc": acc,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                # "f_avg" : f_avg,
                "f_against": f_against,
                "f_favor": f_favor,
                "f_none": f_none,
                "precision": precision,
                "recall": recall,
            }
        return desc

    @torch.no_grad()
    def test(self):
        self.test_list = torch.tensor(self.test_list).long().to(self.device)
        t = self.val(self.test_list, prefix="test")
        # test_desc["seed"] = self.seed
        for key, value in t.items():
            if key not in self.result:
                self.result[key] = []
            self.result[key].append(value)
        self.result["learning_rate"].append(self.learning_rate)
        self.result["dropout"].append(self.dropout)
        self.result["v"].append(self.v)

        # =====更新最佳结果=====
        best_macro = {"macro_f1": float(t['macro_f1']), "micro_f1": float(t['micro_f1']), 
                      "f_favor": float(t['f_favor']), "f_against": float(t['f_against']), "f_none": float(t['f_none']),
                      "learning_rate": self.learning_rate, "num_epoch": self.max_epoch,
                      "batch_size": 1, "dropout": self.dropout, "seed": self.seed}
        best_micro = {"micro_f1": float(t['micro_f1']), "macro_f1": float(t['macro_f1']),
                       "f_favor": float(t['f_favor']), "f_against": float(t['f_against']), "f_none": float(t['f_none']),
                       "learning_rate": self.learning_rate, "num_epoch": self.max_epoch,
                       "batch_size": 1, "dropout": self.dropout, "seed": self.seed}
        if not os.path.exists('../result.json'):
            with open(f"../result.json", "w") as file:
                json.dump({}, file)
        with open(f"../result.json", "r") as file:
            _result = json.load(file)
        if self.dataset not in _result:
            _result[self.dataset] = {}
        if self.target not in _result[self.dataset]:
            _result[self.dataset][self.target] = {}
        if self.model_name not in _result[self.dataset][self.target]:
            _result[self.dataset][self.target][self.model_name] = {"macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
        # 按照macro更新
        if _result[self.dataset][self.target][self.model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
            _result[self.dataset][self.target][self.model_name]["macro"] = best_macro
        # 按照micro更新
        if _result[self.dataset][self.target][self.model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
            _result[self.dataset][self.target][self.model_name]["micro"] = best_micro
        with open(f"../result.json", "w") as file:
            json.dump(_result, file, indent=2)
        # =====更新最佳结果=====end

    @timer
    def main(self):
        self.result = {"learning_rate": [], "dropout": [], "v": []}
        learning_rate_list = [0.01, 0.02, 0.03, 0.05, 0.001,
                              0.002, 0.003, 0.005, 0.0001, 0.0002, 0.0003, 0.0005]
        dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        v_list = [0] if self.topic_by == "" else [x/10 for x in range(11)]
        for learning_rate in tqdm.tqdm(learning_rate_list):
            self.learning_rate = learning_rate
            for dropout in dropout_list:
                self.dropout = dropout
                for v in v_list:
                    self.v = v
                    self.Copy_data()
                    self.Set_seed(self.seed)
                    self.Get_Model()
                    self.Convert_tensor()
                    self.train()
                    self.test()
        self.Record()


if __name__ == "__main__":
    opt = parameter_parser()
    print(opt)
    ins = Instructor(opt)
    ins.main()
