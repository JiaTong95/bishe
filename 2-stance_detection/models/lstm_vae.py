# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model


class VAE(nn.Module):
    def __init__(self, encode_dims=None, decode_dims=None, dropout=0.0):
        super(VAE, self).__init__()
        # if encode_dims is None:
        #     encode_dims = [2000, 1024, 512, 20]
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i + 1])
            for i in range(len(encode_dims) - 2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i + 1])
            for i in range(len(decode_dims) - 1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1], encode_dims[-1])

    def encode(self, x):
        hid = x
        for i, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def inference(self, x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x, dim=1)
        return theta

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder) - 1:
                hid = F.relu(self.dropout(hid))
        return hid

    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta)
        if collate_fn != None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var


class LSTM_VAE(nn.Module):
    def __init__(self, embedding_matrix, bow_dim, opt):
        super(LSTM_VAE, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                bidirectional=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.dense1 = nn.Linear(opt.hidden_dim, opt.topic_dim)
        self.dense2 = nn.Linear(opt.topic_dim, opt.polarities_dim)
        self.vae = VAE(encode_dims=[bow_dim, 1024, 512, opt.topic_dim],
                       decode_dims=[opt.topic_dim, 1024, bow_dim],
                       dropout=opt.dropout)
        self.e = nn.Parameter(torch.randn(opt.topic_dim))  # 标准正态 topic_dim

    def forward(self, inputs):
        text_raw_indices, bow = inputs[0], inputs[1]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, c_n) = self.lstm(x, x_len)
        bows_recon, mus, log_vars = self.vae(bow, lambda x: torch.softmax(x, dim=1))
        # bows_recon    bow_dim
        # mus           topic_dim
        # log_vars      topic_dim
        theta = mus + self.e * log_vars

        out = self.dense1(h_n[0])
        out = out + theta
        out = self.dense2(out)
        return out

