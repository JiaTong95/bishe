# -*- coding: utf-8 -*-
# file: tarm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import math
import torch
import torch.nn as nn

"""
A Topic-Aware Reinforced Model for Weakly Supervised Stance Detection
"""


class TARM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TARM, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.hidden_dim = 2 * opt.hidden_dim
        self.topic_dim = 6
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.gru_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                       rnn_type='GRU', bidirectional=True, dropout=opt.dropout)
        self.gru_topic = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                     rnn_type='GRU', bidirectional=True,
                                     only_use_last_hidden_state=True, dropout=opt.dropout)
        self.W = nn.Parameter(torch.Tensor(self.batch_size, self.hidden_dim, self.hidden_dim).to(self.opt.device))
        self.reset_parameters()  # TODO  Why do this?
        self.alpha_softmax = nn.Softmax(dim=-1)
        self.topic_distribution = [1, 0.2783, 0.1483, 0.165, 0.2087, 0.1996]
        self.B = torch.Tensor([self.topic_distribution]).to(self.opt.device).expand(self.batch_size, 1, self.topic_dim)
        self.dense = nn.Linear(self.hidden_dim, opt.polarities_dim)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.W is not None:
            self.W.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        text_raw_indices, target_topic_words_indices = inputs[0], inputs[1]

        target_topic_words_indices = target_topic_words_indices.permute(1, 0, 2).contiguous()
        topics = []
        for topic_indices in target_topic_words_indices:
            topic_len = torch.sum(topic_indices != 0, dim=-1)
            topic = self.embed(topic_indices)
            topic = self.gru_topic(topic, topic_len).permute(1, 0, 2).contiguous()  # 16*1*600
            topic = topic.view(topic.shape[0], -1)
            topics.append(topic)
        O = torch.stack(topics, dim=-1)  # 16*600*i
        # print(O.shape)

        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        H, (_, _) = self.gru_context(context, text_raw_len)  # 16*j*600

        # H*W*OT
        # print(H.shape, self.W.shape)
        _ = torch.bmm(H, self.W)
        S = torch.bmm(_, O).permute(0, 2, 1)  # 16*i*j
        A = self.alpha_softmax(S)  # 16*i*j
        R = torch.bmm(A, H)  # 16*i*600
        B = self.B  # 16*1*i
        # print(B.shape, R.shape)
        x = torch.bmm(B, R)
        x = x.squeeze(dim=1)  # 16*1*600
        out = self.dense(x)
        return out
