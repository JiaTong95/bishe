import torch
import torch.nn as nn
import torch.nn.functional as F

class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel,self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(n_dim * context_size, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)

        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob

class SVM:
    def __init__(self,embedding_matrix,opt):
        super(SVM,self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        #不会写了
        self.fc = nn.Linear(,opt.polarities_dim)
    def forward(self,x):
        x=self.embed[]
        return out