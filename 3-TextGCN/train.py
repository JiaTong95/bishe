import gc
import warnings
from time import time

import tqdm
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class PrepareData:
    def __init__(self, args):
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        self.features = torch.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target

        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split

        self.train_lst, self.test_lst = get_train_test(target_fn)

        with open(f"data/word2id/{self.args.dataset}.json", 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)

        self.text_indices = self.get_text_indices()

    def get_text_indices(self):
        text_indices = []
        with open(f"data/text_dataset/clean_corpus/{self.args.dataset}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_indices = []
                for word in line.split():
                    line_indices.append(self.word2id[word])
                text_indices.append(line_indices)
        return text_indices


class TextGCNTrainer:
    def __init__(self, args, model, pre_data: PrepareData):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        # print(self.model.parameters)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model_param = sum(param.numel() for param in self.model.parameters())
        # print('# model parameters:', self.model_param)
        self.convert_tensor()

        start = time()
        self.train()
        self.train_time = time() - start

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass
        self.text_indices = self.predata.text_indices

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = torch.tensor(self.target).long().to(self.device)
        self.train_lst = torch.tensor(self.train_lst).long().to(self.device)
        self.val_lst = torch.tensor(self.val_lst).long().to(self.device)

    def train(self):
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch": epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            # self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
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
        self.test_lst = torch.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main(times):
    args = parameter_parser()

    model = GCN

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    for ind, seed in tqdm.tqdm(enumerate(return_seed(args.times))):
        print(f"==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if torch.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    s = record.show_str()
    with open(f'logs/{args.dataset}{datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M")}.txt', 'w') as f:
        f.write(s)


if __name__ == '__main__':
    main()
