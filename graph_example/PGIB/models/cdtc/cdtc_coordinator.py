
# ### Task Association Modeling
# - **Purpose**: Utilize labeled target domain data as prompts to model the association between meta-tasks from the source domain \(D_S\) and prompt tasks, aiming to learn accurate task reps for task selection.
# - **Prompt Data Role**: Assist in task selection, not directly involved in graph base learner training.
#
# ### Task reps Learning
# - **Objective**: Select \(B\) beneficial meta-tasks from \(M_s\) candidate meta-tasks by leveraging task instances.
# - **Meta-task reps**: For a \(N\)-way \(K\)-shot meta-task \(\tau_i\) from \(D_S\), described by \(N\) class prototypes \(p_n\), the task reps \(p_{\tau_i}\) is:
#   - \(p_{\tau_i} = MLP_{W_a}(p_1, p_2, ..., p_N)\)
#   - For binary classification, concatenate prototypes: \(p_{\tau_i} = MLP_{W_a}(p_{pos} \| p_{neg})\)
# - **Prompt Tasks**: \(N\)-way \(K\)-shot tasks constructed from target domain data to serve as prompts, denoted by \(M_t\).
#
# ### Task Bipartite Graph
# - **Graph Definition**: \(GB = (U, V, E)\) where \(U\) and \(V\) represent candidate meta-tasks and prompt tasks, respectively, each vertex characterized by task reps.
# - **Adjacency Matrix Initialization**:
#   - \(A(\tau_i, \tau_j) = \begin{cases} 1, & \text{if} \; ||p_{\tau_i} - p_{\tau_j}||_2 \geq \delta; \\ 0, & \text{otherwise} \end{cases}\)
#   - \(||p_{\tau_i} - p_{\tau_j}||_2\) measures the Euclidean distance between task reps.
#
# ### Task reps Refinement
# - **Objective**: Refine task reps through information propagation on the bipartite graph to better depict task relationships.
# - **GCN for Refinement**: Employ a two-layer GCN on \(GB\), updating task reps \(P^{(l)}\) at layer \(l\) as follows:
#   - \(P^{(l)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} P^{(l-1)} W_g^{(l)})\)
#   - \(\tilde{A} = A + I\), \(\tilde{D} = \sum_i \tilde{A}_{ij}\), \(P^{(0)}\) initialized with task reps from both \(M_s\) and \(M_t\).

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import math


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

class TaskRepresentation(nn.Module):
    def __init__(self, num_classes, prototype_dim, out_features):
        super(TaskRepresentation, self).__init__()
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        # 计算MLP输入的维度，因为我们将所有原型向量展平
        mlp_input_dim = num_classes * prototype_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, prototypes):
        # 确保原型张量的形状是我们期望的 [num_classes, prototype_dim]
        assert prototypes.shape == (self.num_classes, self.prototype_dim)
        # 展平原型张量，以便可以作为MLP的输入
        flattened_prototypes = prototypes.view(-1)
        # 通过MLP生成任务表示
        task_reps = self.mlp(flattened_prototypes)
        return task_reps

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature_matrix, bipartite_graph):
        feature_matrix = feature_matrix.to(self.device)
        bipartite_graph = bipartite_graph.to(self.device)

        degree_matrix = torch.diag(bipartite_graph.sum(1))
        D_half_inv = torch.pow(degree_matrix, -0.5).to(self.device)
        D_half_inv[torch.isinf(D_half_inv)] = 0

        norm_adjacency = torch.mm(torch.mm(D_half_inv, bipartite_graph), D_half_inv).to(self.device)
        return torch.mm(norm_adjacency, torch.mm(feature_matrix, self.weight))

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TaskRepresentationRefinement(nn.Module):
    """
    任务表示精炼模块，使用两层GCN来精炼任务表示。
    """

    def __init__(self, in_features, out_features):
        """
        初始化任务表示精炼模块。

        参数:
        in_features (int): 输入特征的维度。
        out_features (int): 输出特征的维度。
        """
        super(TaskRepresentationRefinement, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gcn1 = GCNLayer(in_features, out_features).to(device)  # 第一层GCN
        self.gcn2 = GCNLayer(out_features, out_features).to(device) # 第二层GCN

    def forward(self, feature_matrix, bipartite_graph):
        """
        前向传播。

        参数:
        feature_matrix (Tensor): 特征矩阵，大小为 [N, in_features]，N为节点数。
        adjacency_matrix (Tensor): 邻接矩阵，大小为 [N, N]。

        返回:
        Tensor: 精炼后的任务表示，大小为 [N, out_features]。
        """
        # 第一层GCN处理
        x = F.relu(self.gcn1(feature_matrix, bipartite_graph))
        # 第二层GCN处理，输出精炼后的任务表示
        x = self.gcn2(x, bipartite_graph)
        return x


class TaskSelector(nn.Module):
    def __init__(self, in_features, source_num_meta_tasks):
        super(TaskSelector, self).__init__()
        self.relevance_mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Normalize features before the final decision layer
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.importance_weights = nn.Parameter(torch.randn(source_num_meta_tasks, 1) * 0.1)  # Smaller initialization

    def forward(self, source_task_reps, mean_target_task_reps):
        # 计算相关性得分
        relevance_scores = self.relevance_mlp(source_task_reps * mean_target_task_reps)

        # 计算重要性得分，确保使用sigmoid函数后维度与relevance_scores一致
        importance_scores = torch.sigmoid(self.importance_weights).view_as(relevance_scores)

        # 计算总得分
        total_scores = relevance_scores + importance_scores
        # 打印相关性得分、重要性得分和总得分
        # print(f"Relevance scores: {relevance_scores}")
        # print(f"Importance scores: {importance_scores}")
        # print(f"Total scores: {total_scores}")
        probabilities = torch.softmax(total_scores, dim=0).squeeze(-1)
        return probabilities

class MetaTaskSelector(nn.Module):
    def __init__(self, node_attr_dim, hidden_dim, embedding_dim, proto_dim, refined_dim, source_num_meta_tasks, target_num_meta_tasks, num_classes, delta, num_tasks_to_select):
        """
        初始化元任务选择器。

        参数:
        - node_attr_dim: 节点属性的维度。
        - hidden_dim: 隐藏层维度。
        - embedding_dim: 嵌入维度。
        - proto_dim: 原型维度。
        - refined_dim: 精炼后的任务表示维度。
        - num_meta_tasks: 元任务的数量。
        - num_classes: 类的数量。
        - delta: 用于任务选择的阈值。
        - num_tasks_to_select: 需要选择的任务数量。
        """
        super(MetaTaskSelector, self).__init__()
        self.selected_probabilities = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化图嵌入模块
        self.graph_embedding = GIN(node_attr_dim, hidden_dim, embedding_dim, num_layers=2).to(device)
        # 初始化任务表示模块
        self.task_reps_model = TaskRepresentation(num_classes, proto_dim, refined_dim).to(device)
        # 初始化任务表示精炼模块
        self.task_reps_refinement = TaskRepresentationRefinement(refined_dim, refined_dim).to(device)
        # 初始化任务选择模块
        self.task_selector = TaskSelector(refined_dim, source_num_meta_tasks).to(device)
        self.num_classes = num_classes
        self.delta = delta
        self.num_tasks_to_select = num_tasks_to_select

        self.saved_log_probs = []  # 用于保存选择动作的对数概率

    def compute_prototypes(self, task_data):
        # 首先，从 task_data 中提取图数据 ('s_data') 和对应的图级别标签 ('s_label')
        graph_data = task_data[0]['s_data']
        graph_labels = task_data[0]['s_label']
        # 获取每个图示例的索引（在批处理中，每个节点属于哪个图示例）
        # 使用图嵌入模型处理图数据，获取每个图示例的嵌入
        embeddings = self.graph_embedding(graph_data.x, graph_data.edge_index, graph_data.batch)
        # 使用全局平均池化将节点嵌入聚合为图示例级别的嵌入
        # graph_embeddings = global_mean_pool(embeddings, batch_index)
        # 对于每个类别，计算该类别所有图示例嵌入的平均值，作为类原型
        prototypes = torch.stack([embeddings[graph_labels == i].mean(0) for i in range(self.num_classes)])

        return prototypes

    def process_tasks(self, task_data_dict):
        """
        处理一系列任务的数据，并将其转换为任务表示向量。

        参数:
            task_data_dict: 一个字典，键为任务ID，值为任务的数据。

        返回:
            task_reps: 一个tensor，包含了所有任务的表示向量。
        """
        task_reps = []
        # 遍历任务数据字典，计算每个任务的原型并生成任务表示
        for task_id, task_data in task_data_dict.items():
            prototypes = self.compute_prototypes(task_data)  # 计算任务数据的原型
            task_rep = self.task_reps_model(prototypes)  # 通过模型生成任务表示
            task_reps.append(task_rep.squeeze(0))  # 移除单个维度的0轴以整理tensor形状
        task_reps = torch.stack(task_reps)
        return task_reps

    def build_bipartite_graph(self, task_reps):
        num_tasks = task_reps.size(0)  # 获取任务的数量
        bipartite_graph = torch.zeros((num_tasks, num_tasks))  # 初始化邻接矩阵为全0

        # 遍历所有任务对，并根据它们的表示距离设置邻接矩阵的值
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                distance = torch.norm(task_reps[i] - task_reps[j])  # 计算任务i和j的表示距离
                # 根据距离设置邻接矩阵的值，并保证邻接矩阵是对称的
                bipartite_graph[i, j] = bipartite_graph[j, i] = 1 if distance >= self.delta else 0
        return bipartite_graph

    def select_best_tasks(self, refined_task_reps, target_reps):
        """
        从给定的任务表示中选择最优的任务集合。

        参数:
        - refined_task_reps: 经过加工后的任务表示，通常为向量形式，用于描述各个任务的特征。
        - target_reps: 目标表示，用于与任务表示进行比较，以决定哪些任务是最相关的。

        返回值:
        - selected_indices: 一个包含所选任务索引的集合，这些索引对应于输入的refined_task_reps中被选中的任务。
        """
        # 计算选择概率
        selection_probabilities = self.task_selector(refined_task_reps, target_reps)
        # 获取选择的任务索引
        _, selected_indices = torch.topk(selection_probabilities, self.num_tasks_to_select)
        selected_probabilities = torch.gather(selection_probabilities, 0, selected_indices)
        # 保存选择任务的对数概率，用于之后的更新
        self.saved_log_probs.append(torch.log(selected_probabilities + 1e-10))  # 加上小值以避免对数0的情况

        return selected_indices, selected_probabilities, selection_probabilities

    def select_tasks(self, task_data_dict, target_task_data_dict):
        """
        根据给定的任务数据和目标表示，选择一组任务。

        :param task_data_dict: 一个字典，包含任务的数据表示。
        :param target_task_data_dict:一个字典，包含目标域任务的数据表示。
        :return: 一个列表，包含选中任务的索引。
        """
        # 处理任务数据，获取任务的表示
        source_task_reps = self.process_tasks(task_data_dict)
        target_task_reps = self.process_tasks(target_task_data_dict)
        # 把task_reps和target_reps 合并成一个tensor
        task_reps = torch.cat([source_task_reps, target_task_reps], dim=0)
        # 基于任务表示构建邻接矩阵
        bipartite_graph = self.build_bipartite_graph(task_reps)
        # 精细调整任务表示，考虑任务间的关联性
        refined_task_reps = self.task_reps_refinement(task_reps, bipartite_graph)
        # 从refined_task_reps拆分成refined_source_task_reps和refined_target_task_reps
        refined_source_task_reps = refined_task_reps[:source_task_reps.size(0)]
        refined_target_task_reps = refined_task_reps[source_task_reps.size(0):]
        # 处理目标任务数据，计算目标表示
        mean_target_reps = torch.mean(refined_target_task_reps, dim=0, keepdim=True)
        # 基于优化后的任务表示和目标表示，选择最佳任务
        selected_indices, self.selected_probabilities, selection_probabilities = self.select_best_tasks(refined_source_task_reps, mean_target_reps)

        return selected_indices, self.selected_probabilities, selection_probabilities

    def update_parameters(self, optimizer, reward, baseline):
        """
        使用REINFORCE算法来更新TaskSelector模块的参数。

        参数:
        - optimizer (torch.optim.Optimizer): 用于参数更新的优化器。
        - reward (float): 接收到的奖励信号。
        - baseline (float): 用于奖励标准化的基线值，通常为奖励的移动平均。
        """
        # 使用累积奖励减去基线值作为优化的目标，并且对所有保存的对数概率求和，得到损失函数
        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * (reward - baseline))
        policy_loss = torch.cat(policy_loss).sum()
        # policy_loss = -log_prob * (reward - baseline)
        # 执行反向传播和参数更新
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # 清除保存的对数概率，准备下一个迭代
        del self.saved_log_probs[:]
    # def update_parameters(self, reward, learning_rate):
    #     """
    #     使用REINFORCE算法更新TaskSelector模块的参数。
    #
    #     Args:
    #     - reward (float): 接收到的奖励信号。
    #     - learning_rate (float): 参数更新的学习率。
    #     """
    #     # 假设self.selection_probabilities保存了上次任务选择过程中采取的动作的概率，并且需要梯度。
    #     if not hasattr(self, 'selection_probabilities') or self.selection_probabilities is None:
    #         raise ValueError("未设置选择概率。")
    #
    #     # Calculate the gradients
    #     reward_tensor = torch.tensor(reward, device=self.selection_probabilities.device)
    #     outputs = torch.log(self.selection_probabilities).sum()
    #     policy_gradients = torch.autograd.grad(
    #         outputs= outputs,
    #         inputs=list(self.task_selector.parameters()),
    #         grad_outputs=reward_tensor,  # Ensure this is a tensor with the same device as the probabilities
    #         only_inputs=True,
    #         retain_graph=True
    #     )
    #
    #     # Update parameters using calculated policy gradients
    #     with torch.no_grad():
    #         for param, grad in zip(self.task_selector.parameters(), policy_gradients):
    #             if grad is not None:
    #                 param.data -= learning_rate * grad