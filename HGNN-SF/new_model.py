import torch.nn as nn
import torch.nn.functional as F
import torch
from Attribute_agg import Attribute_agg
from Mix import Mix

class models(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim,adj_len,feat_drop,nei_type_num,sample_rate):
        super(models, self).__init__()
        self.input_dim = input_dim
        self.no_linear = nn.Tanh()
        self.project1 = nn.Linear(hid_dim, hid_dim)
        self.project2 = nn.Linear(hid_dim,hid_dim)
        self.project3 = nn.Linear(hid_dim,out_dim)
        if feat_drop>0:
            self.drop_out = nn.Dropout(feat_drop)
        else:
            self.drop_out = lambda x:x
        #线性转换层
        self.fc_list=nn.ModuleList([
            nn.Linear(feat_dim,hid_dim) for feat_dim in input_dim
        ])
        self.attribute_mix = Mix(hid_dim,nei_type_num,sample_rate,feat_drop)  #进行属性融合
        self.attribute_feature = Attribute_agg(hid_dim,nei_type_num)    #属性融合的特征进行特征聚合

    def forward(self,features,ADJ,NS):
        h_all = [self.no_linear(self.fc_list[i](features[i])) for i in range(len(features))]    #特征转换 所有节点投影到相同空间
        mix_attr_feat = self.attribute_mix(h_all,NS)      #获得属性融合后的节点特征
        final_feat = self.attribute_feature(mix_attr_feat,ADJ)    #将属性融合后的节点特征放进融合层进行特征融合
        final_feat = self.drop_out(final_feat)
        z = self.no_linear(self.project2(final_feat)) + h_all[0]
        return self.no_linear(self.project3(z)),z

