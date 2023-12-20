import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import random
from Attention_Model import intra_att

class Mix(nn.Module):
    def __init__(self,hidden_dim,nei_num,sample_rate,attn_drop):
        super(Mix, self).__init__()
        self.nei_num = nei_num      #len(NS)
        self.sample_rate = sample_rate
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])


    def forward(self, nei_h, nei_index):  # nei_h所有节点特征,nei_index相邻节点索引(NS)
        attribute_feature = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:  # 从4019个节点中选出相连的节点放入sele_nei中
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)  # 选取出的邻居节点，最终变为长度为4019的列表，列表每个元素是当前节点相连的节点编号
            sele_nei = torch.cat(sele_nei, dim=0)  # 进行cat操作，变成（4019,7）
            one_type_emb = (self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))  # 把选择出的连接矩阵，邻居节点特征与自身节点特征传入 self.intra
            one_type_emb = one_type_emb + nei_h[0]  # 把属性特征加给原始特征
            attribute_feature.append(one_type_emb)  # 将每个网络模式下得到的嵌入放入列表，heco后面是对列表内的不同模式下嵌入进行注意力聚合
        return attribute_feature

