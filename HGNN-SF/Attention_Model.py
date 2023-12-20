import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):#参数分别是：nei选择出的连接矩阵(4019,7)，h邻居节点特征(7167,64) 与 h_refer自身节点特征

        nei_emb = F.embedding(nei, h) #获取选择出的节点的特征，每个节点选择7个邻居，所以变成了（4019,7,64）

        h_refer = torch.unsqueeze(h_refer, 1)   #(4019,1,64)
        h_refer = h_refer.expand_as(nei_emb)    #扩展成(4019,7,64)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)     #拼接自身节点特征 与 选择出的节点特征
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)

        nei_emb = (att*nei_emb).sum(dim=1)

        return nei_emb




class AggAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(AggAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)