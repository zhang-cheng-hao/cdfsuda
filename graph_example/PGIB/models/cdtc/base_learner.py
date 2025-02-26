import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GINLayer, self).__init__()
        self.gin = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        )

    def forward(self, x, edge_index):
        return self.gin(x, edge_index)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

# class GINModel(nn.Module):
#     def __init__(self, args):
#         super(GINModel, self).__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(GINLayer(args.input_dim, args.hidden_dim))
#         for _ in range(args.num_layers - 1):
#             self.layers.append(GINLayer(args.hidden_dim, args.hidden_dim))
#         self.pool = global_mean_pool
#         self.classifier = nn.Linear(args.hidden_dim, args.num_classes)
#
#     def process_batch(self, x, edge_index, batch):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, else CPU
#         x = x.to(device)
#         edge_index = edge_index.to(device)
#         batch = batch.to(device)
#         for layer in self.layers:
#             x = layer(x, edge_index)
#         x = self.pool(x, batch)
#         return x
#
#     def forward(self, s_data, q_data, s_label):
#         # 处理支持集
#         s_output = self.process_batch(s_data.x, s_data.edge_index, s_data.batch)
#         s_logits = self.classifier(s_output)
#
#         # 处理查询集
#         q_output = self.process_batch(q_data.x, q_data.edge_index, q_data.batch)
#         q_logits = self.classifier(q_output)
#
#         # 暂时以这种方式返回节点表示和邻接矩阵，具体实现可以根据需要调整
#         node_emb = q_output
#         adj = q_data.edge_index
#
#         return s_logits, q_logits, adj, node_emb
#
#     def forward_query_loader(self, s_data, q_loader, s_label):
#         s_output = self.process_batch(s_data.x, s_data.edge_index, s_data.batch)
#         s_logits = self.classifier(s_output)
#         logits = []
#         labels = []
#         adj_list = []
#         for batch in q_loader:
#             out = self.process_batch(batch.x, batch.edge_index, batch.batch)
#             logits.append(self.classifier(out))
#             labels.append(batch.y)
#             adj_list.append(batch.edge_index)
#
#         # 汇总查询集输出
#         logits = torch.cat(logits, dim=0)
#         labels = torch.cat(labels, dim=0)
#         return s_logits, logits, labels, adj_list, s_label
class GINModel(nn.Module):
    def __init__(self, args):
        super(GINModel, self).__init__()
            # Assume args has the necessary dimensions and layer count
        self.gin = GIN(in_channels=args.input_emb_dim,
                       hidden_channels=args.gin_hidden_dim,
                       out_channels=args.n_class,
                       num_layers=args.num_layers)

    def forward(self, s_data, q_data, s_label):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, else CPU
        # s_data = s_data.to(device)
        # q_data = q_data.to(device)
        s_logits = self.gin(s_data.x, s_data.edge_index, s_data.batch)
        q_logits = self.gin(q_data.x, q_data.edge_index, q_data.batch)
        # Assume node embedding and adjacency matrix need to be returned
        node_emb = s_data.x  # This can be replaced with appropriate node embeddings
        adj = s_data.edge_index  # Assuming we want to return the adjacency matrix of s_data

        return s_logits, q_logits, adj, node_emb
    def forward_query_loader(self, s_data, q_loader, s_label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, else CPU
        s_data = s_data.to(device)
        s_output = self.gin(s_data.x, s_data.edge_index, s_data.batch)
        s_logits = s_output
        logits = []
        labels = []
        adj_list = []
        for batch in q_loader:
            batch = batch.to(device)
            out = self.gin(batch.x, batch.edge_index, batch.batch)
            logits.append(out)
            labels.append(batch.y)
            adj_list.append(batch.edge_index)

        # 汇总查询集输出
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        return s_logits, logits, labels, adj_list, s_label
