import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils import to_dense_adj, add_self_loops, degree

def approximate_largest_eigenvalue(L):
    """
    Approximates the largest eigenvalue of the Laplacian matrix L using power iteration.
    """
    x = torch.rand(L.size(0), device=L.device)
    for _ in range(100):  # Number of power iterations
        x = L @ x
        x = x / x.norm()
    largest_eigenvalue = torch.dot(x, L @ x) / torch.dot(x, x)
    return largest_eigenvalue
class GraphAugmentation(torch.nn.Module):
    def __init__(self, d_x, d_u, d_z,):
        super(GraphAugmentation, self).__init__()

        self.d_x = d_x
        self.d_u = d_u
        self.d_z = d_z

    def heterogeneous_feature_augmentation(self, x, in_channels):
        # 混合特征增强
        # 输入: x [n, in_channels], in_channels: 输入特征维度, out_channels: 输出特征维度
        # 输出: padded [n, d_x]
        projection = torch.nn.Linear(in_channels, self.d_x).to('cuda')
        x = x.to('cuda')
        x = x.float()
        projected = projection(x)  # [n, out_channels]
        concatenated = torch.cat((x, projected), dim=1)  # [n, in_channels + out_channels]
        padded = F.pad(concatenated, (0, self.d_x - concatenated.size(1)), "constant", 0)  # [n, predefined_size]
        return padded

    def sinusoidal_node_degree_encoding(self, edge_index, num_nodes):
        """
        Create sinusoidal node degree encodings.

        Args:
            edge_index (Tensor): Edge indices of the graph.
            num_nodes (int): The number of nodes.

        Returns:
            Tensor: Sinusoidal node degree encodings with shape [n, num_encodings].
        """
        # Compute node degrees [n]
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)

        # num_encodings (int): The number of sinusoidal encoding dimensions.
        num_encodings = self.d_u
        # Initialize the encoding matrix [n, num_encodings]
        encoding = torch.zeros((deg.size(0), num_encodings), dtype=torch.float)

        # Compute sinusoidal encodings
        for i in range(num_encodings):
            if i % 2 == 0:
                encoding[:, i] = torch.sin(deg / (10000 ** (i // 2 / num_encodings)))
            else:
                encoding[:, i] = torch.cos(deg / (10000 ** (i // 2 / num_encodings)))

        return encoding.to('cuda')

    def graph_diffusion(self, edge_index, batch, num_nodes):
        largest_eigenvalues_list = []
        device = edge_index.device
        for i in range(batch.max().item() + 1):
            mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
            sub_edge_index = edge_index[:, mask]
            sub_dense_adj = to_dense_adj(sub_edge_index, max_num_nodes=num_nodes)[0]

            # Compute the degree matrix and normalized Laplacian as before
            sub_deg = degree(sub_edge_index[0], dtype=torch.float, num_nodes=sub_dense_adj.size(0))
            D_inv_sqrt = torch.diag(sub_deg.pow(-0.5))
            D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
            I = torch.eye(sub_dense_adj.size(0), device=device)
            L = I - D_inv_sqrt @ sub_dense_adj @ D_inv_sqrt

            # Approximate the largest eigenvalue
            largest_eigenvalue = approximate_largest_eigenvalue(L)

            # Expand or embed the scalar eigenvalue into a higher-dimensional space (e.g., a vector of size d_z)
            # todo 这里需要修改,改成最大的前d_z,而不是最大的第一个重复多次，
            eigenvalue_vector = largest_eigenvalue.repeat(self.d_z)
            # Compute eigenvalues and eigenvectors
            # eigenvalues, eigenvectors = torch.linalg.eigh(L)
            # # Sort eigenvalues and select the top d_z
            # sorted_indices = torch.argsort(eigenvalues, descending=True)[:self.d_z]
            # top_dz_eigenvalues = eigenvalues[sorted_indices]
            # eigenvalue_vector = top_dz_eigenvalues
            largest_eigenvalues_list.append(eigenvalue_vector)

        largest_eigenvalues_batched = torch.stack(largest_eigenvalues_list).to(device)
        # Ensure the output is of shape [batch_size, d_z]
        return largest_eigenvalues_batched

    # def graph_diffusion(self, edge_index, batch, num_nodes):
    #     top_k_eigenvalues_list = []
    #     device = edge_index.device
    #     for i in range(batch.max().item() + 1):
    #         # Filter edges to get subgraph belonging to the current graph in the batch
    #         mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
    #         sub_edge_index = edge_index[:, mask]
    #
    #         # Convert the subgraph's edge indices to a dense adjacency matrix
    #         sub_dense_adj = to_dense_adj(sub_edge_index, max_num_nodes=num_nodes)[0]
    #
    #         # Compute the degree matrix for normalization
    #         sub_deg = degree(sub_edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    #
    #         # Calculate the normalized Laplacian matrix
    #         D_inv_sqrt = torch.diag(sub_deg.pow(-0.5))
    #         D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
    #         I = torch.eye(num_nodes, device=device)
    #         L = I - D_inv_sqrt @ sub_dense_adj @ D_inv_sqrt
    #
    #         # Compute eigenvalues and eigenvectors. Note: torch.symeig is deprecated in favor of torch.linalg.eigh
    #         eigenvalues, _ = torch.linalg.eigh(L)
    #
    #         # Handle cases where the number of eigenvalues is less than self.d_z
    #         top_k_eigenvalues = eigenvalues[:self.d_z]
    #         if top_k_eigenvalues.size(0) < self.d_z:
    #             padding = torch.zeros(self.d_z - top_k_eigenvalues.size(0), device=device)
    #             top_k_eigenvalues = torch.cat([top_k_eigenvalues, padding], dim=0)
    #
    #         top_k_eigenvalues_list.append(top_k_eigenvalues)
    #
    #     # Stack the top K eigenvalues from each graph to form a batched tensor
    #     top_k_eigenvalues_batched = torch.stack(top_k_eigenvalues_list)
    #     return top_k_eigenvalues_batched
    # def graph_diffusion(self, edge_index, batch, num_nodes):
    #     top_k_eigenvalues_list = []
    #     device = edge_index.device
    #     for i in range(batch.max().item() + 1):
    #         mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
    #         sub_edge_index = edge_index[:, mask]
    #
    #         # No need to specify num_nodes_subgraph for to_dense_adj if the graphs are already isolated
    #         sub_dense_adj = to_dense_adj(sub_edge_index)[0]
    #
    #         # Ensure self-loops are added for Laplacian calculation
    #         sub_edge_index, _ = add_self_loops(sub_edge_index, num_nodes=sub_dense_adj.size(0))
    #         sub_deg = degree(sub_edge_index[0], sub_dense_adj.size(0), dtype=sub_dense_adj.dtype)
    #
    #         # Construct the normalized Laplacian matrix: L = I - D^(-1/2) * A * D^(-1/2)
    #         D_inv_sqrt = sub_deg.pow(-0.5)
    #         D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
    #         L = torch.eye(sub_dense_adj.size(0), device=sub_dense_adj.device) - sub_dense_adj * D_inv_sqrt.view(-1,
    #                                                                                                             1) * D_inv_sqrt.view(
    #             1, -1)
    #
    #         eigenvalues, eigenvectors = torch.symeig(L, eigenvectors=True)
    #         top_k_eigenvalues = eigenvalues[:self.d_z]
    #         if len(top_k_eigenvalues) < self.d_z:
    #             padding_values = torch.zeros(self.d_z - len(top_k_eigenvalues), device=device, dtype=torch.float)
    #             top_k_eigenvalues = torch.cat([top_k_eigenvalues, padding_values], dim=0)
    #
    #         top_k_eigenvalues_list.append(top_k_eigenvalues.to(device))
    #
    #     top_k_eigenvalues_batched = torch.stack(top_k_eigenvalues_list)
    #
    #     return top_k_eigenvalues_batched

    def forward(self, x, edge_index, num_nodes, in_channels, batch):
        # 前向传播
        # 输入: x [n, in_channels], edge_index [2, E], num_nodes, in_channels, out_channels
        # 输出: x_hetero [n, predefined_size], x_degree [n, 2], x_diffusion [num_eigenvalues]
        X = self.heterogeneous_feature_augmentation(x, in_channels)
        U = self.sinusoidal_node_degree_encoding(edge_index, num_nodes)
        z = self.graph_diffusion(edge_index, batch, num_nodes)

        return X, U, z
