import argparse
import re
import torch
import numpy as np
import random
import scipy.sparse as sp
import sys
import pickle as pkl

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='SDwH')
    parser.add_argument('--target', type=str, required=True, help="trump")
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cuda:0", "cuda:1"])
    parser.add_argument('--gcn_lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    args = parser.parse_args()
    return args


# def get_file(choice):
#     file_dict = {"R8": 'R8.txt', 'ohsumed': 'ohsumed.txt', '20news': "20ng.txt", 'mr': 'mr.txt'}
#     if choice not in file_dict.keys():
#         raise FileNotFoundError
#     return file_dict[choice]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj', 'adj_topic']
    objects = []
    for i in range(len(names)):
        with open(f"data/ind/{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj, adj_topic = tuple(objects)
    print(f"x.shape={x.shape}, y.shape={y.shape}, tx.shape={tx.shape}, ty.shape={ty.shape}, allx.shape={allx.shape}, ally.shape={ally.shape}")
    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print("len(labels) =", len(labels))

    train_idx_orig = parse_index_file(f"data/ind/{dataset_str}.train.index")
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # 将非对称邻接矩阵转变为对称邻接矩阵(有向图转无向图), 可用数学证明。
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_topic = adj_topic + adj_topic.T.multiply(adj_topic.T > adj_topic) - adj_topic.multiply(adj_topic.T > adj_topic)
    return adj, adj_topic, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size