import gc
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

from settings import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

class PrepareData:
    def __init__(self, opt):
        self.opt = opt
        self.Get_Graph()
        self.Get_Features()
        self.Get_Trainset_Testset_Split()
        self.Get_Text_Indices()
        self.Get_NClass()

    # Get_Graph 获取图(邻接矩阵)
    def Get_Graph(self):
        print("==Get_Graph 获取图(邻接矩阵)")
        self.graph = nx.read_weighted_edgelist(
            f"{GRAPH_PATH}/{self.opt.dataset}.txt", nodetype=int)
        print_graph_detail(self.graph)
        # 转换成networkx库可以处理的格式
        adj = nx.to_scipy_sparse_matrix(self.graph,
                                        nodelist=list(range(self.graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        # 非对称矩阵转换成对称矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

    # Get_Features 获取初始训练特征
    def Get_Features(self):
        print("==Get_Trainset_Testset_Split 获取训练集和数据集")
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
        print("==Get_Trainset_Testset_Split 获取训练集、验证集和数据集")
        from sklearn.model_selection import train_test_split
        
        self.train_list, self.test_list = [], []
        fname = f"{LABEL_PATH}/{self.opt.dataset}.txt"
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
        print("==Get_Text_Indices 获取词标记")
        with open(f"{WORD2ID_PATH}/{self.opt.dataset}.json", 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.text_indices = []
        with open(f"{CLEAN_CORPUS_PATH}/{self.opt.dataset}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_indices = []
                for word in line.split():
                    line_indices.append(self.word2id[word])
                self.text_indices.append(line_indices)

    # Get_NClass 获取标签种类数量
    def Get_NClass(self):
        print("==Get_NClass 获取标签种类数量")
        target_fn = f"{LABEL_PATH}/{self.opt.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.data = PrepareData(opt)

    def Copy_data(self):
        copy_data = copy.deepcopy(self.data)
        self.feat_dim = copy_data.feat_dim
        self.nclass = copy_data.nclass
        self.adj = copy_data.adj
        self.features = copy_data.features
        self.target = copy_data.target
        self.train_list = copy_data.train_list
        self.val_list = copy_data.val_list
        self.test_list = copy_data.test_list

    # Convert_tensor 转换tensor
    def Convert_tensor(self):
        print("==Convert_tensor 转换tensor")
        self.model = self.model.to(self.opt.device)
        self.adj = self.adj.to(self.opt.device)
        self.features = self.features.to(self.opt.device)
        self.target = torch.tensor(self.target).long().to(self.opt.device)
        self.train_list = torch.tensor(self.train_list).long().to(self.opt.device)
        self.val_list = torch.tensor(self.val_list).long().to(self.opt.device)    
        self.test_list = torch.tensor(self.test_list).long().to(self.opt.device)

    # Set_seed 设置种子
    def Set_seed(self, seed):
        print("==Set_seed 设置种子")
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Get_Model 获取模型
    def Get_Model(self):
        print("==Get_Model 获取模型")
        self.model = GCN(feat_dim=self.feat_dim,
                         hid_dim=self.opt.hid_dim,
                         nclass=self.nclass,
                         dropout=self.opt.dropout,
                         v=self.opt.v)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
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

    def train(self):
        print("开始训练")
        self.earlystopping = EarlyStopping(self.opt.early_stopping)
        for epoch in range(self.opt.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_list],
                                  self.target[self.train_list])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_list)

            desc = dict(**{"epoch": epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            if self.earlystopping(val_desc["val_loss"]):
                print(f"epoch={epoch}, earlystopping...")
                break

    @torch.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            precision, recall, _, _ = precision_recall_fscore_support(y_true=self.target[x].cpu(),
                                                                      y_pred=torch.argmax(logits[x], -1).cpu(),
                                                                      labels=[0, 1, 2],
                                                                      average='macro')
            micro_f1 = f1_score(y_true=self.target[x].cpu(),
                                y_pred=torch.argmax(logits[x], -1).cpu(),
                                labels=[0, 1, 2],
                                average='micro')
            macro_f1 = f1_score(y_true=self.target[x].cpu(),
                                y_pred=torch.argmax(logits[x], -1).cpu(),
                                labels=[0, 1, 2],
                                average='macro')

            f_against = f1_score(y_true=self.target[x].cpu(),
                                 y_pred=torch.argmax(logits[x], -1).cpu(),
                                 labels=[0],
                                 average='macro')
            f_favor = f1_score(y_true=self.target[x].cpu(),
                               y_pred=torch.argmax(logits[x], -1).cpu(),
                               labels=[2],
                               average='macro')
            f_none = f1_score(y_true=self.target[x].cpu(),
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
        self.test_list = torch.tensor(self.test_list).long().to(self.opt.device)
        test_desc = self.val(self.test_list, prefix="test")
        test_desc["seed"] = self.seed
        for key, value in test_desc.items():
            if key not in self.result:
                self.result[key] = []
            self.result[key].append(value)

    def main(self):
        self.result = {}
        for seed in range(1, self.opt.seed_num+1):
            self.Copy_data()
            self.Set_seed(seed)
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
