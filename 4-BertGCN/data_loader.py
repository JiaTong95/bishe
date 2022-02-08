import dgl
import torch
from scipy.sparse import eye
from utils import load_corpus, normalize_adj
from torch.utils.data import Dataset
from torch import tensor
from transformers import BertTokenizer
import pickle
import os


class DataSet_Bert_base(Dataset):
    def __init__(self, name, usage, label_dict=None):
        if not os.path.exists("data/pkl/BertTokenizer-bert-base-uncased.pkl"):
            tokenizers = BertTokenizer.from_pretrained('bert-base-uncased')
            pickle.dump(tokenizers, open(
                "data/pkl/BertTokenizer-bert-base-uncased.pkl", 'wb'))
        else:
            tokenizers = pickle.load(
                open("data/pkl/BertTokenizer-bert-base-uncased.pkl", 'rb'))
        label_path = 'data/pkl/' + name + '_labels.pkl'
        indexs = 'data/pkl/' + name + '_indexs.pkl'
        current_usage = 'data/pkl/' + name + '_' + usage + '_index.pkl'
        current_usage = pickle.load(open(current_usage, 'rb'))
        orig_data_path = 'data/corpus/' + name + '.clean.txt'
        orig_data = open(orig_data_path).readlines()
        indexs = pickle.load(open(indexs, 'rb'))
        labels = pickle.load(open(label_path, 'rb'))
        orig_data = [orig_data[i] for i in indexs]
        orig_data = [tensor(tokenizers.encode(each, max_length=512))
                     for each in orig_data]
        self.data = [orig_data[i] for i in current_usage]
        self.label = [labels[i] for i in current_usage]
        self.class_num = len(pickle.load(
            open('data/pkl/' + name + '_label_dict.pkl', 'rb')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return tensor(self.data[item]), tensor(self.label[item])


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=512, truncation=True,
                      padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


class DataSet_Bert_GCN(Dataset):
    def __init__(self, name):
        adj, adj_topic, feature, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
            name)
        # doc_mask = train_mask+test_mask+val_mask
        print("adj.shape =", adj.shape)
        adj = normalize_adj(adj+eye(adj.shape[0]))
        print("adj.shape =", adj.shape)
        print("adj_topic.shape =", adj_topic.shape)
        adj_topic = normalize_adj(adj_topic)

        train_num = train_mask.sum().item()
        val_num = val_mask.sum().item()
        test_num = test_mask.sum().item()
        node_size = adj.shape[0]
        y = torch.tensor(y_train+y_val+y_test)
        self.y = torch.argmax(y, -1)

        self.train_index = [i for i in range(
            train_num+val_num)] + [i for i in range(node_size-test_num, node_size)]
        corpse_file = open('data/corpus/' + name + '.clean.txt').readlines()
        if not os.path.exists("data/pkl/BertTokenizer-bert-base-uncased.pkl"):
            tokenizers = BertTokenizer.from_pretrained('bert-base-uncased')
            pickle.dump(tokenizers, open(
                "data/pkl/BertTokenizer-bert-base-uncased.pkl", 'wb'))
        else:
            tokenizers = pickle.load(
                open("data/pkl/BertTokenizer-bert-base-uncased.pkl", 'rb'))
        self.dataset, self.attention_mask = encode_input(
            corpse_file, tokenizers)
        self.attention_mask = torch.tensor(self.attention_mask)
        self.dataset = torch.tensor(self.dataset)
        self.graph = dgl.from_scipy(adj, eweight_name='w')
        self.graph.ndata['label'] = self.y
        self.label_num = len(y_train[0])
        self.graph.edata['w'] = self.graph.edata['w'].float()
        self.graph.ndata['train_mask'] = torch.tensor(train_mask)
        self.graph.ndata['valid_mask'] = torch.tensor(val_mask)
        self.graph.ndata['test_mask'] = torch.tensor(test_mask)

        self.topic_graph = dgl.from_scipy(adj_topic, eweight_name='w')
        self.topic_graph.ndata['label'] = self.y
        self.label_num = len(y_train[0])
        self.topic_graph.edata['w'] = self.topic_graph.edata['w'].float()
        self.topic_graph.ndata['train_mask'] = torch.tensor(train_mask)
        self.topic_graph.ndata['valid_mask'] = torch.tensor(val_mask)
        self.topic_graph.ndata['test_mask'] = torch.tensor(test_mask)
        self.train_mask = train_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.y[self.train_index[item]], self.attention_mask[item], self.train_mask[self.train_index[item]], self.train_index[item]


if __name__ == "__main__":
    DataSet_Bert_GCN("SDwH_trump")
