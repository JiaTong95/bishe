# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class IAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                        dropout=opt.dropout)
        self.lstm_target = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                       dropout=opt.dropout)
        self.attention_target = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)  # (batch_size, context_len)
        target_len = torch.sum(target_indices != 0, dim=-1)  # (batch_size, target_len)

        context = self.embed(text_raw_indices)  # (batch_size, context_len, emb_dim)
        target = self.embed(target_indices)  # (batch_size, target_len, emb_dim)
        
        context, (h1, _) = self.lstm_context(context, text_raw_len)  # (batch_size, context_len, hidden_dim)
        target, (h2, _) = self.lstm_target(target, target_len)  # (batch_size, target_len, hidden_dim

        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        target_pool = torch.sum(target, dim=1)
        target_pool = torch.div(target_pool, target_len.view(target_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        target_final, _ = self.attention_target(target, context_pool)
        target_final = target_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, target_pool)
        context_final = context_final.squeeze(dim=1)
        x = torch.cat((target_final, context_final), dim=-1)
        # x.shape == [batch_size, hidden_dim*2]
        out = self.dense(x)
        return out
