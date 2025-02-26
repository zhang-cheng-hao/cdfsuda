
from chem_lib.models.cdfs.augmentation import GraphAugmentation
from .cdfs_encoder import  PrototypicalNetwork
import torch
import torch.nn as nn


class PrototypicalNetworkModel(nn.Module):
    def __init__(self, args):
        super(PrototypicalNetworkModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.graph_augmentation = GraphAugmentation(d_x=args.d_x, d_u=args.d_u , d_z=args.d_z).to(device)
        self.graph_encoder = PrototypicalNetwork(d_x=args.d_x, d_u=args.d_u , d_z=args.d_z,  d_h=args.d_h).to(device)
        # 添加分类MLP层
        self.classifier = nn.Sequential(
            nn.Linear(args.d_h, args.d_h // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.d_h // 2, args.n_class)
        ).to(device)

    def label2edge(self, label, mask_diag=True):
        # Convert labels to adjacency matrix
        num_samples = label.size(1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float().to(label.device)

        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2), device=edge.device).unsqueeze(0).unsqueeze(0)
            edge = edge * diag_mask

        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1, keepdim=True)
        return edge
    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        # 数据增强
        X_s,U_s,z_s = self.graph_augmentation(x=s_data.x, edge_index=s_data.edge_index, num_nodes=s_data.num_nodes, in_channels=s_data.num_node_features, batch=s_data.batch)
        X_q,U_q,z_q = self.graph_augmentation(x=q_data.x, edge_index=q_data.edge_index, num_nodes=q_data.num_nodes, in_channels=q_data.num_node_features, batch=q_data.batch)
        
        # 对支持集和查询集图进行编码
        s_h = self.graph_encoder(s_data.edge_index, X_s, U_s, z_s, s_data.batch)  
        q_h = self.graph_encoder(q_data.edge_index, X_q, U_q, z_q, q_data.batch)

        # adj是邻接信息的占位符,具体取决于任务的需求
        # 在此示例中未计算,因为它依赖于具体任务的细节
        adj = None
        s_emb = None
        # 输出s_logits和q_logits
        s_logits = self.classifier(s_h)
        q_logits = self.classifier(q_h)
        return s_logits, q_logits, adj, s_emb

    def label2edge(self, label, mask_diag=True):
        # Convert labels to adjacency matrix
        num_samples = label.size(1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float().to(label.device)

        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2), device=edge.device).unsqueeze(0).unsqueeze(0)
            edge = edge * diag_mask

        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1, keepdim=True)
        return edge

    def compute_prototypical_logits(self, q_h, s_h, s_label):
        device = q_h.device
        num_classes = self.args.n_class
        class_prototypes = torch.stack([s_h[s_label == i].mean(0) for i in range(num_classes)]).to(device)

        # Expand dimensions for broadcasting
        q_h_expanded = q_h.unsqueeze(1).expand(q_h.size(0), class_prototypes.size(0), q_h.size(1))
        class_prototypes_expanded = class_prototypes.unsqueeze(0).expand(q_h.size(0), class_prototypes.size(0),
                                                                         q_h.size(1))

        # Compute squared Euclidean distance between query embeddings and class prototypes
        distances = torch.sum((q_h_expanded - class_prototypes_expanded) ** 2, dim=2)

        # Compute logits as the negative distance
        logits = -distances

        return logits

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move support data to CUDA
        s_data.x = s_data.x.to(device)
        s_data.edge_index = s_data.edge_index.to(device)
        s_data.batch = s_data.batch.to(device)
        if s_label is not None:
            s_label = s_label.to(device)

        # Data augmentation for support data
        X_s, U_s, z_s = self.graph_augmentation(x=s_data.x, edge_index=s_data.edge_index,
                                                num_nodes=s_data.num_nodes, in_channels=s_data.num_node_features,
                                                batch=s_data.batch)
        # Encode augmented support data
        s_h = self.graph_encoder(s_data.edge_index, X_s, U_s, z_s, s_data.batch)

        y_true_list = []
        q_logits_list, adj_list = [], []

        # Process each batch of query data
        for q_data in q_loader:
            # Move query data to CUDA
            q_data.x = q_data.x.to(device)
            q_data.edge_index = q_data.edge_index.to(device)
            q_data.batch = q_data.batch.to(device)

            # Data augmentation for query data
            X_q, U_q, z_q = self.graph_augmentation(x=q_data.x, edge_index=q_data.edge_index,
                                                    num_nodes=q_data.num_nodes, in_channels=q_data.num_node_features,
                                                    batch=q_data.batch)
            # Encode augmented query data
            q_h = self.graph_encoder(q_data.edge_index, X_q, U_q, z_q, q_data.batch)

            # Assuming query data has labels (y_true), store them for return
            y_true_list.extend(q_data.y.tolist())

            # Implement your logic for computing prototypical logits or any other inference logic based on s_h and q_h
            q_logits = self.compute_prototypical_logits(q_h, s_h, s_label)
            q_logits_list.append(q_logits)

            # Adjacency information handling if needed
            adj = None  # Placeholder, replace or compute as necessary
            adj_list.append(adj)

        # Concatenate logits and labels from all query batches
        q_logits = torch.cat(q_logits_list, dim=0)
        y_true = torch.tensor(y_true_list, device=device)

        # Collect the support and query labels for analysis or further processing
        sup_labels = {'support': s_data.y, 'query': y_true}

        s_logits = self.classifier(s_h)
        
        return s_logits, q_logits, y_true, adj_list, sup_labels


