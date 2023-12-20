import dgl
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import copy
from Attention_Model import AggAttention

class GCN(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super(GCN, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim)).data
        if bias:
            self.Bias = Parameter(torch.FloatTensor(hidden_dim)).data
        else:
            self.register_parameter('bias', True)
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

class Attribute_agg(nn.Module):
    '''
    首先进行同属性特征聚合
    然后进行非同属性特征聚合
    再将这两者进行聚合，获得特定属性的节点特征表示
    最后，将不同属性的节点特征表示进行注意力聚合
    '''
    def __init__(self,hid_dim,nei_num):
        super(Attribute_agg, self).__init__()
        self.nei_num = nei_num
        self.gcn = nn.ModuleList(GCN(hid_dim) for _ in range(self.nei_num))             #同属性特征聚合
        self.dif_attr_gcn = nn.ModuleList([GCN(hid_dim)for _ in range(self.nei_num)] )        #不同属性间特征聚合
        self.attr_att = AggAttention(hid_dim)   #获得带属性的特征表示
        self.mul_graph_attr_att = AggAttention(hid_dim)      #将不同属性特征进行最终聚合

    def forward(self,attr_feat,ADJ):
        '''
        :param attr_feat: 带属性的目标节点特征，以ACM为例:[P+A,P+S]   DBLP:[A+P,A+P,A+P]  IMDB:[M+A,M+D]
        :param ADJ:邻接矩阵，以ACM为例为：[PAP,PSP]    DBLP:[APA,APCPA,APVPA]     IMDB:[MAM,MDM]
        :return:
        '''
        mul_graph_feat = []
        for i in range(self.nei_num):
            #首先进行同属性的特征聚合
            intra_attr_feat = self.gcn[i](attr_feat[i],ADJ[i])    #得到第i个属性特征的同属性聚合结果
            ADJ_pop = copy.deepcopy(ADJ)
            ADJ_pop.pop(i)
            dif_attr_feat = self.dif_attr_gcn[i](attr_feat[i],ADJ_pop[0])    #这样意味着ADJ中只能有两个邻接矩阵，每次同属性聚合一次，另一个改为不同属性聚合
            graph_attr_feat = torch.stack([intra_attr_feat,dif_attr_feat],dim=1)        #进行拼接，准备使用注意力聚合
            graph_attr_feat = self.attr_att(graph_attr_feat)        #使用注意力聚合同属性与不同属性特征
            mul_graph_feat.append(graph_attr_feat)      #将单个属性图的聚合结果放入列表，准备对多属性图进行聚合
        mul_graph_attr_feat = self.mul_graph_attr_att(torch.stack(mul_graph_feat,dim=1))    #多属性图注意力聚合
        return mul_graph_attr_feat
