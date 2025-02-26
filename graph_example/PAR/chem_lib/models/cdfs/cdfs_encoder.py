
import torch.nn.functional as F


import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch.distributions.normal import Normal


class FeatureWiseTransformationLayer(nn.Module):
    def __init__(self, size):
        super(FeatureWiseTransformationLayer, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Assuming 'x' is of shape [batch_size, feature_size]
        gamma_dist = Normal(1, self.softplus(self.gamma))
        beta_dist = Normal(0, self.softplus(self.beta))

        # Since 'self.gamma' and 'self.beta' are of shape [feature_size],
        # we sample 'gamma_sample' and 'beta_sample' to have the shape [1, feature_size]
        # and then expand them to match 'x' in batch dimension.
        gamma_sample = gamma_dist.rsample((1,)).expand_as(x)
        beta_sample = beta_dist.rsample((1,)).expand_as(x)

        return gamma_sample * x + beta_sample


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, d_h):
        super(GraphEncoder, self).__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.gnn = GINConv(nn1, train_eps=True).to('cuda')  # GIN 卷积层
        self.fwt = FeatureWiseTransformationLayer(d_h).to('cuda')  # 特征逐渐转换层
        self.swish = nn.SiLU().to('cuda')  # Swish 激活函数
        self.pool = global_mean_pool  # 全局平均池化

    def forward(self, x, edge_index, batch):
        # todo input should be A [N,N], X [N,d_x] input_dim
        # the output shoud be H^x [N, d_h]
        # x: enhanced node feature [N, input_dim]
        # edge_index: edge index [2, E]
        # batch: batch information, which is used for pooling operations
        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        batch = batch.to('cuda')
        h = self.gnn(x, edge_index)  # [N, d_h]
        h = self.fwt(h)  # 应用特征逐渐转换
        h = self.swish(h)  # 激活函数
        h_pool = self.pool(h, batch)  # 池化操作
        return h_pool


class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim, d_h):
        super(MLPProjectionHead, self).__init__()
        # self.fc1 = nn.Linear(input_dim, d_h)
        # self.swish = nn.SiLU()
        # input: d_z input_dim
        # output: d_h
        self.fc2 = nn.Linear(input_dim, d_h).to('cuda')

    def forward(self, z):
        # z: 特征值向量 [batch_size, d_z]
        # h = self.swish(self.fc1(z))
        h = self.fc2(z)
        return h


class AttentionModule(nn.Module):
    # 定义注意力模块
    def __init__(self, d_h):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(d_h * 3, d_h)
        self.fc2 = nn.Linear(d_h, 3)

    def forward(self, concatenated_h):
        # 计算注意力权重
        scores = F.relu(self.fc1(concatenated_h))
        alpha = F.softmax(self.fc2(scores), dim=1)
        return alpha


class PrototypicalNetwork(nn.Module):
    def __init__(self, d_x, d_u, d_z, d_h):
        super(PrototypicalNetwork, self).__init__()
        # TODO 输入应该是超参数
        self.encoder_contextual = GraphEncoder(input_dim=d_x, d_h=d_h)
        self.encoder_topological = GraphEncoder(input_dim=d_u, d_h=d_h)
        self.encoder_diffusion = MLPProjectionHead(input_dim=d_z, d_h=d_h)
        self.attention_module = AttentionModule(d_h= d_h)

    def forward(self, A_g, X_g, U_g, S_g, batch):
        # 对上下文视图进行编码并池化以得到图表征 h_x
        h_x = self.encoder_contextual(X_g, A_g, batch)
        # 对拓扑视图进行编码并池化以得到图表征 h_u
        h_u = self.encoder_topological(U_g, A_g, batch)
        # 对图扩散特征值向量进行投影得到表征 h_z
        h_z = self.encoder_diffusion(S_g)
        # 将三个表征合并并通过注意力模块计算注意力权重
        alpha = self.attention_module(torch.cat([h_x, h_u, h_z], dim=1))
        # 使用注意力权重聚合特征
        h = alpha[:, 0].unsqueeze(1) * h_x + alpha[:, 1].unsqueeze(1) * h_u + alpha[:, 2].unsqueeze(1) * h_z
        return h


