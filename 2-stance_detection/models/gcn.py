import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention


class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__( self, embedding_matrix,
                        opt,
                        input_dim,
                        support,
                        out_dim,
                        dropout_rate=0.,
                        ):
        super(GCN, self).__init__()
        self.N = input_dim
        print(f"input_dim={input_dim}, out_dim={out_dim}, dropout_rate={dropout_rate}")
        # GraphConvolution
        self.graph_layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.graph_layer2 = GraphConvolution(200, opt.polarities_dim, support, dropout_rate=dropout_rate)
        self.graph_hidden_dim = input_dim * opt.polarities_dim
        self.graph_layer3 = nn.Linear(self.graph_hidden_dim, opt.hidden_dim)

        # RNN
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.embed = nn.Embedding(10000, 300)  # not pretrained
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                        dropout=opt.dropout)
        self.lstm_target = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                       dropout=opt.dropout)
        self.attention_cg = Attention(opt.hidden_dim, score_function='mlp')
        self.attention_ct = Attention(opt.hidden_dim, score_function='mlp')
        # self.attention_final = Attention(opt.hidden_dim, score_function='bi_linear')
        # self.attention_target = Attention(opt.hidden_dim, score_function='bi_linear')
        # self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        
    
    def forward(self, inputs):
        I = torch.eye(self.N)
        # print(f"I:{I}")
        # print(f"I.shape:{I.shape}")
        _ = self.graph_layer1(I)
        _ = self.graph_layer2(_)
        Gr = self.graph_layer3(_.reshape(self.graph_hidden_dim).expand(self.opt.batch_size, self.graph_hidden_dim))
        # print(f"out:{out}")
        # print(f"Gr.shape:{Gr.shape}")

        text_raw_indices, target_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)  # (batch_size, context_len)
        target_len = torch.sum(target_indices != 0, dim=-1)  # (batch_size, target_len)

        context = self.embed(text_raw_indices)  # (batch_size, context_len, emb_dim)
        target = self.embed(target_indices)  # (batch_size, target_len, emb_dim)

        context, (context_h, _) = self.lstm_context(context, text_raw_len)  # (batch_size, context_len, hidden_dim)
        target, (target_h, _) = self.lstm_target(target, target_len)  # (batch_size, target_len, hidden_dim)

        # print(context.shape, target.shape)
        cg, _score = self.attention_cg(context, Gr)
        ct, _score = self.attention_ct(context, target)
        # print(cg.shape, ct.shape)
        # out, _score = self.attention_final(cg, ct)
        out = torch.cat([cg, ct], 2).squeeze(1)
        # print(out.shape)
        out = self.dense(out)
        # print(out.shape)
        return out

# class GCN(nn.Module):
#     def __init__( self, embedding_matrix,
#                         opt,
#                         input_dim,
#                         support,
#                         out_dim,
#                         dropout_rate=0.,
#                         ):
#         super(GCN, self).__init__()
#         self.N = input_dim
#         print(f"input_dim={input_dim}, out_dim={out_dim}, dropout_rate={dropout_rate}")
#         # GraphConvolution
#         self.graph_layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
#         self.graph_layer2 = GraphConvolution(200, opt.polarities_dim, support, dropout_rate=dropout_rate)
#         self.graph_hidden_dim = input_dim * opt.polarities_dim
#         self.graph_layer3 = nn.Linear(self.graph_hidden_dim, opt.hidden_dim)

#         # RNN
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         # self.embed = nn.Embedding(10000, 300)  # not pretrained
#         self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
#                                         dropout=opt.dropout)
#         self.lstm_target = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
#                                        dropout=opt.dropout)
#         self.attention_cg = Attention(opt.hidden_dim, score_function='bi_linear')
#         self.attention_ct = Attention(opt.hidden_dim, score_function='bi_linear')
#         self.attention_final = Attention(opt.hidden_dim, score_function='bi_linear')
#         # self.attention_target = Attention(opt.hidden_dim, score_function='bi_linear')
#         # self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
#         self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        
    
#     def forward(self, inputs):
#         text_raw_indices, target_indices = inputs[0], inputs[1]
#         text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)  # (batch_size, context_len)
#         target_len = torch.sum(target_indices != 0, dim=-1)  # (batch_size, target_len)

#         context = self.embed(text_raw_indices)  # (batch_size, context_len, emb_dim)
#         target = self.embed(target_indices)  # (batch_size, target_len, emb_dim)

#         context, (context_h, _) = self.lstm_context(context, text_raw_len)  # (batch_size, context_len, hidden_dim)
#         target, (target_h, _) = self.lstm_target(target, target_len)  # (batch_size, target_len, hidden_dim)

#         # print(context.shape, target.shape)

#         out, _score = self.attention_final(context, target)
#         # print(out.shape)
#         out = self.dense(out.squeeze(1))
#         # print(out.shape)
#         return out
        