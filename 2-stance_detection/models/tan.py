from layers.dynamic_rnn import DynamicLSTM
from layers.attention import NoQueryAttention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TAN, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                        dropout=opt.dropout, bidirectional=True)
        # Output Weight (hidden_dim * 2, num_class)
        self.W = nn.Parameter(torch.Tensor(self.opt.embed_dim * 2, self.opt.polarities_dim)).to(self.opt.device)
        # Output Bias (batch_size, num_class)
        self.B = nn.Parameter(torch.Tensor(self.opt.batch_size, self.opt.polarities_dim)).to(self.opt.device)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)

        # Context Representation
        x = self.embed(text_raw_indices)  # (batch_size, max_seq_len, emb_dim)
        h, (_, _) = self.lstm_context(x, text_raw_len)  # (batch_size, context_len, hidden_dim * 2)

        # Target-augmented Embedding
        target = self.embed(target_indices[target_indices > 0]).reshape(
            self.batch_size, int(target_len[0]), -1)  # (batch_size, target_len, emb_dim)
        target = torch.mean(target, dim=1, keepdim=True)  # (batch_size, 1, emb_dim)
        target = target.expand(target.shape[0], self.opt.max_seq_len, -1)  # (batch_size, max_seq_len, emb_dim)
        z = torch.cat((x, target), dim=-1)  # (batch_size, max_seq_len, embed_dim * 2)

        # Target-specific Attention Extraction
        att = self.attention_layer(z)  # (batch_size, max_seq_len, 1)

        # Stance Classification
        s = h * att  # (batch_size, max_seq_len, hidden_dim * 2)
        s = torch.mean(s, dim=1)  # (batch_size, hidden_dim * 2)
        out = torch.matmul(s, self.W) + self.B  # (batch_size, num_class)
        out = F.softmax(out, dim=-1)  # (batch_size, num_class)
        return out
