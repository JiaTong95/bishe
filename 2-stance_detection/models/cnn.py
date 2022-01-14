import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义网络结构
class TextCNN(torch.nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.embed = nn.Embedding(100000, 300)
        kernel_num = 100  ## 每种卷积核的数量
        # Ks = [3,4,5]  ## 卷积核list，形如[2,3,4]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(3, opt.embed_dim), stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(4, opt.embed_dim), stride=1,
                               padding=0)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(5, opt.embed_dim), stride=1,
                               padding=0)

        # self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, opt.embed_dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(3 * kernel_num, opt.polarities_dim)  ##全连接层

    def forward(self, x):
        x = self.embed(x[0])  # (N,W,D) N=batch_size
        # print(x.shape)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        out1 = F.relu(self.conv1(x)).squeeze(3)
        # print(out1.shape)
        _1 = F.max_pool1d(out1, out1.size(2)).squeeze(2)
        out2 = F.relu(self.conv2(x)).squeeze(3)
        _2 = F.max_pool1d(out2, out2.size(2)).squeeze(2)
        out3 = F.relu(self.conv3(x)).squeeze(3)
        _3 = F.max_pool1d(out3, out3.size(2)).squeeze(2)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        # x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat([_1, _2, _3], 1)  # (N,Knum*len(Ks))

        x = self.dropout(x)
        logit = self.fc(x)
        return logit


class Pkulab_CNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Pkulab_CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.embed = nn.Embedding(10000, 300)

    def forward(self, x):
        return out
