import dgl
import torch
import torch.nn as nn

# 假设有一个节点特征矩阵 node_features 和一个邻接矩阵 adjacency_matrix
# num_nodes = node_features.shape[0]
#
# # 创建一个DGL图
# graph = dgl.DGLGraph()
# graph.add_nodes(num_nodes)
# src_nodes, dst_nodes = adjacency_matrix.nonzero()
# graph.add_edges(src_nodes, dst_nodes)

# 定义注意力机制的类
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attn_weights = nn.Parameter(torch.randn(out_features, out_features))

    def forward(self, adjacency_matrix, node_features):
        h = self.linear(node_features)

        # 计算注意力权重
        attention_scores = torch.matmul(h, torch.matmul(self.attn_weights, h.t()))

        # 使用 softmax 归一化得到注意力系数
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=1)

        # 使用注意力权重聚合节点特征
        h_agg = torch.matmul(attention_scores, h)

        return h_agg

# 创建一个包含多个注意力层的注意力机制模型
class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(GraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphAttentionLayer(in_features, hidden_features))
        for _ in range(num_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_features, hidden_features))
        self.out_layer = nn.Linear(hidden_features, out_features)

    def forward(self, graph, node_features):
        for layer in self.layers:
            node_features = layer(graph, node_features)
        output = self.out_layer(node_features)
        return output

# 创建注意力机制模型并进行前向传播
# in_features = node_features.shape[1]
# hidden_features = 64
# out_features = 1
# num_layers = 2
# model = GraphAttentionNetwork(in_features, hidden_features, out_features, num_layers)
# output = model(graph, node_features)

# # 输出的 output 包含了每个节点的最终特征表示
