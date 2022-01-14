import torch
import torch.nn as nn


class BERT_DUAL_LOSS(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_DUAL_LOSS, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.num_labels = opt.polarities_dim
        self.labels = torch.tensor([i for i in range(self.num_labels)], dtype=torch.long).to(opt.device)
        self.dense = nn.Linear(opt.bert_dim + 1, self.num_labels)
        self.cosine = nn.CosineSimilarity()

    def forward(self, inputs, return_loss=False):
        text_bert_indices, bert_segments_ids, target_bert_indices = inputs[0], inputs[1], inputs[2]

        _, pooled_output1 = self.bert(text_bert_indices, bert_segments_ids)
        p1 = self.dropout(pooled_output1)

        _, pooled_output2 = self.bert(target_bert_indices)
        p2 = pooled_output2
        # p2 = self.dropout(pooled_output2)

        cos_sim = self.cosine(p1, p2).unsqueeze(1)
        combined = torch.cat([p1, cos_sim], dim=-1)
        logits = self.dense(combined)

        if return_loss:
            CrossEntropyLoss = nn.CrossEntropyLoss()
            # CosineEmbeddingLoss = nn.CosineEmbeddingLoss()
            print(logits.view(-1, self.num_labels), self.labels.view(-1))
            loss_bert = CrossEntropyLoss(logits.view(-1, self.num_labels), self.labels.view(-1))
            # loss_intent = CosineEmbeddingLoss(p1, p2, ) #TODO 三分类是不是用不了啊
            # loss = loss_bert + loss_intent
            return logits, loss_bert
        else:
            return logits

# class BertForSequenceClassificationDualLoss(BertPreTrainedModel):
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassificationDualLoss, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size + 1, num_labels)
#         self.cosine = nn.CosineSimilarity()
#         self.alpha = 0.5
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sim_labels=None):
#
#         sen1_attention_mask = (1 - token_type_ids) * attention_mask
#
#         _, pooled_output_combined = self.bert(input_ids, token_type_ids, attention_mask,
#                                               output_all_encoded_layers=False)
#         pooled_output_combined = self.dropout(pooled_output_combined)
#
#         _, pooled_output_sen1 = self.bert(input_ids, token_type_ids, sen1_attention_mask,
#                                           output_all_encoded_layers=False)
#
#         cos_sim = self.cosine(pooled_output_combined, pooled_output_sen1).unsqueeze(1)
#
#         combined = torch.cat([pooled_output_combined, cos_sim], dim=1)
#         logits = self.classifier(combined)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss_bert = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#             loss_cosine = CosineEmbeddingLoss()
#             loss_intent = loss_cosine(pooled_output_combined, pooled_output_sen1, sim_labels.float())
#
#             loss = loss_bert + loss_intent
#
#             return loss
#         else:
#             return logits
