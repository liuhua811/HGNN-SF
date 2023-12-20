import dgl
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import random
from dgl.nn.pytorch import GATConv
from Attention import GraphAttentionNetwork

class Attention(nn.Module):
    def __init__(self, hid_dim, bias=True):
        super(Attention, self).__init__()

        self.attention = nn.MultiheadAttention(hid_dim, num_heads=1)  # Attention mechanism
        self.weight = Parameter(torch.FloatTensor(hid_dim, hid_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(hid_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, inputs, adj):
        output = torch.spmm(adj, torch.spmm(inputs, self.weight))
        if self.bias is not None:
            output += self.bias
        output = F.elu(output)
        # Apply attention mechanism
        output, _ = self.attention(output, output, output)


        return output

class GCN(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(GCN, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self,):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)
        if self.Bias is not None:
            self.Bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        output = torch.spmm(adj, torch.spmm(inputs, self.Weight))
        if self.Bias is not None:
            output = output + self.Bias
        return output

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc

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

class Attribute(nn.Module):
    def __init__(self, hidden_dim,adj_len, attn_drop, sample_rate, nei_num):
        super(Attribute, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = nn.ModuleList([inter_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.feat_drop = nn.Dropout(attn_drop)
        self.predict_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.GCN_agg = nn.ModuleList([GCN(hidden_dim) for _ in range(nei_num)])
        self.attention = nn.ModuleList([Attention(hidden_dim) for _ in range(nei_num)])
        self.final_att = inter_att(hidden_dim, attn_drop)

    def forward(self, nei_h, nei_index,ADJ,ADJ_trans):        #nei_h所有节点特征,nei_index相邻节点索引
        attribute_feature = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:   #从4019个节点中选出相连的节点放入sele_nei中
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)     #选取出的邻居节点，最终变为长度为4019的列表，列表每个元素是当前节点相连的节点编号
            sele_nei = torch.cat(sele_nei, dim=0)        #进行cat操作，变成（4019,7）
            one_type_emb = (self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))        #把选择出的连接矩阵，邻居节点特征与自身节点特征传入 self.intra
            one_type_emb = one_type_emb + nei_h[0]      #把属性特征加给原始特征
            attribute_feature.append(one_type_emb)         #将每个网络模式下得到的嵌入放入列表，heco后面是对列表内的不同模式下嵌入进行注意力聚合

        #attribute_feature中保存的为进行属性聚合后的新节点特征
        # 接下来应该对获得属性特征的节点进行异属性间节点聚合 及 同属性间节点聚合
        #属性内是PAP  属性间则是PSP --->假设还存在有 PSPSP,应考虑多情况进行代码设计
        multi_attr_feat=[]
        for i in range(self.nei_num):
            semantic_embeds = []
            attr_features =self.GCN_agg[i](attribute_feature[i],ADJ_trans[i])   #聚合的属性节点应该和其对应的元路径在矩阵中的位置是相对的
            '''
            使用注意力对不同属性间聚合
            '''
            dif_attr_feat = self.attention[i](attribute_feature[i],ADJ[i])      #一个HAN
            semantic_embeds.append(attr_features)
            semantic_embeds.append(dif_attr_feat)
            semantic_embeddings = torch.stack(semantic_embeds, dim=1)
            sem_emb_feat = self.inter[i](semantic_embeddings)      #此处应该使用不同的inter
            multi_attr_feat.append(sem_emb_feat)
        #循环结束，此时multi_attr_feat里为不同类型节点作为属性聚合后的特征表示
        #接下来应该对不同属性的特征表示进行注意力聚合

        multi_attr_feat = torch.stack(multi_attr_feat, dim=1)
        final_feat = self.final_att(multi_attr_feat)
        return final_feat











