# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F


class TEST(nn.Module):
    def __init__(self,bow_dim,topic_dim):
        super(TEST, self).__init__()

        # self.embed = nn.Embedding(10000, 300)
        self.vae = VAE(encode_dims=[bow_dim, 1024, 512, topic_dim],
                       decode_dims=[topic_dim, 1024, bow_dim],
                       dropout=0)
        self.e = nn.Parameter(torch.randn(topic_dim), requires_grad=True)  # 标准正态 topic_dim

    def forward(self, inputs):
        bow = inputs
        _, mus, log_vars = self.vae(bow)
        # bows_recon    bow_dim
        # mus           topic_dim
        # log_vars      topic_dim
        print(mus.shape, self.e.shape, log_vars.shape)
        theta = mus + self.e*log_vars
        print(theta.shape)
        
        return theta

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

if __name__ == "__main__":
    x = torch.rand(16, 10000)
    model = TEST(bow_dim=10000, topic_dim=20)
    _ = model(x)
