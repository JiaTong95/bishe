import argparse
import random
from collections import defaultdict
from prettytable import PrettyTable
import numpy as np
import torch as th
import scipy.sparse as sp


def macro_f1(pred, targ, num_classes: list):
    pred = th.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))

    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    if np.isnan(f1):
        f1 = 0
    return f1, precision, recall


def accuracy(pred, targ):
    pred = th.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc


def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)


def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    import networkx as nx
    dst = {"nodes": nx.number_of_nodes(graph),
           "edges": nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates": nx.number_of_isolates(graph),
           "覆盖度": 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)


def print_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)


def return_seed(nums=10):
    # seed = [47, 17, 1, 3, 87, 300, 77, 23, 13]
    seed = [i for i in range(nums)]
    # seed = random.sample(range(0, 100000), nums)
    return seed


def preprocess_adj(adj, is_sparse=False, plus_e=True):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    # 加上单位矩阵
    if plus_e:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()


# def preprocess_adj_topic(adj):
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "hdd_data/prepare_dataset/model/model.pt"

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        th.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return th.load(self.model_path)


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed yelp_dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hid_dim', type=int, default=200)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed_num', type=int, default=10)
    parser.add_argument('--v', type=float, default=0)
    parser.add_argument('--dataset', type=str, default="trump")
    return parser.parse_args()


class LogResult:
    def __init__(self):
        self.result = defaultdict(list)
        pass

    def log(self, result: dict):
        for key, value in result.items():
            self.result[key].append(value)

    def log_single(self, key, value):
        self.result[key].append(value)

    # def show_str(self):
    #     print()
    #     string = ""
    #     for key, value_lst in self.result.items():
    #         value = np.mean(value_lst)
    #         if isinstance(value, int):
    #             string += f"{key}:\n{value}\n{max(value_lst)}\n{min(value_lst)}\n"
    #             # string += f"{key}:\n{value}\n"
    #         else:
    #             string += f"{key}:\n{value:.4f}\n{max(value_lst):.4f}\n{min(value_lst):.4f} \n"
    #             # string += f"{key}:\n{value:.4f}\n"
    #     print(string)

    def show_str(self):
        table = PrettyTable()
        for key, value_list in self.result.items():
            if key == 'gpu_mem':
                continue
            min_v = f"min:{float(np.min(value_list)):.4}"
            max_v = f"max:{float(np.max(value_list)):.4}"
            mean = f"mean:{float(np.mean(value_list)):.4}"
            value_list = [f"{float(x):.4}" for x in value_list]
            value_list.extend([min_v, max_v, mean])
            table.add_column(key, value_list)
        print(table)
        return table.__str__()

class CudaUse(object):
    def __init__(self):
        self.cuda_available = th.cuda.is_available()
        if self.cuda_available:
            from fastai.utils.pynvml_gate import load_pynvml_env
            self.pynvml = load_pynvml_env()

    def get_cuda_id(self):
        if self.cuda_available:
            gpu_mem = sorted(self.gpu_mem_get_all(), key=lambda item: item.free, reverse=True)
            low_use_id = gpu_mem[0].id
            return th.device(f'cuda:{low_use_id}')
        else:
            return th.device('cpu')

    def gpu_mem_get_all(self):
        "get total, used and free memory (in MBs) for each available gpu"
        return list(map(self.gpu_mem_get, range(self.pynvml.nvmlDeviceGetCount())))

    def gpu_mem_get(self, _id=None):
        """get total, used and free memory (in MBs) for gpu `id`. if `id` is not passed,
        currently selected torch device is used"""
        from collections import namedtuple
        GPUMemory = namedtuple('GPUMemory', ['total', 'free', 'used', 'id'])

        if _id is None:
            _id = th.cuda.current_device()
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(_id)
            info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            # return GPUMemory(*(map(b2mb, [info.total, info.free, info.used])), id=_id)
            return b2mb(info.used)
        except:
            return GPUMemory(0, 0, 0, -1)