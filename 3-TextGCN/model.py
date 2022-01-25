import math
import torch

from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = torch.spmm(infeatn, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, feat_dim, hid_dim, nclass, dropout, v=1):
        super(GCN, self).__init__()
        self.v = v
        self.gc1 = GraphConvolution(feat_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, nclass)
        self.gc_mask = GraphConvolution(hid_dim, 200)
        self.fc_mask = nn.Linear(200, nclass)
        self.dropout = dropout

    def forward(self, x, adj, topic_graph=None, mask_graph=None):
        # mask是个N*N的矩阵，把除了hashtag以外的都挡住，
        # option1：mask[i,j]=1 (i=句子，j=hashtag)
        # option2：mask[i,j]=1 (i=句子，j=hashtag 对应的topic word)
        if topic_graph is not None:
            adj = adj + topic_graph * self.v

        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        
        if mask_graph != None:
            x = self.gc_mask(x, adj)
            x = torch.relu(x)
            mat = torch.mm(mask_graph, x)  # mask=n*n, x=n*hid, mask*x = n*hid
            mat = F.softmax(mat.sum(-1, keepdim=True), dim=-1)  # mat = n*1
            x = mat * x  # mat=n*1, x=n*hid, mat*x = n*hid
            x = self.fc_mask(x)
        else:
            x = self.gc2(x, adj)
                    
        return x


class cross_GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout, v=1):
        super(cross_GCN, self).__init__()
        self.v = v
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 200)
        self.fc = nn.Linear(200, nclass)
        self.dropout = dropout

    def forward(self, x, adj, mask, adj_topic=None):
        # mask是个N*N的矩阵，把除了hashtag以外的都挡住，
        # option1：mask[i,j]=1 (i=句子，j=hashtag)
        # option2：mask[i,j]=1 (i=句子，j=hashtag 对应的topic word)
        if adj_topic is not None:
            adj = adj + adj_topic * self.v
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        x = torch.relu(x)

        mat = torch.mm(mask, x)  # mask=n*n, x=n*hid, mask*x = n*hid
        mat = F.softmax(mat.sum(-1, keepdim=True), dim=-1)  # mat = n*1
        x = mat * x  # mat=n*1, x=n*hid, mat*x = n*hid

        x = self.fc(x)
        return x
