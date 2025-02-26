import torch
import torch.nn as nn

from graph_lib.models.par.encoder import GNN_Encoder
from graph_lib.models.par.relation import ContextMLP, TaskAwareRelation


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.gpu_id = args.gpu_id

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)


        self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                batch_norm=args.batch_norm,dropout=args.map_dropout,
                                pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)

        inp_dim = args.map_dim
        self.adapt_relation = TaskAwareRelation(inp_dim=inp_dim, hidden_dim=args.rel_hidden_dim,
                                                num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                top_k=args.rel_k, res_alpha=args.rel_res,
                                                batch_norm=args.batch_norm, adj_type=args.rel_adj,
                                                activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                pre_dropout=args.rel_dropout2, num_class=args.n_class)

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False, return_adj=False, return_emb=False):
        """
        根据给定的实体嵌入和关系适应模块，计算实体之间的关系得分或适应后的邻接矩阵。

        :param s_emb: 发起实体的嵌入向量
        :param q_emb: 接收实体的嵌入向量
        :param s_label: 发起实体的真实标签（可选），用于条件关系适应
        :param q_pred_adj: 是否使用接收实体的预测邻接矩阵（而非真实的邻接矩阵）
        :param return_adj: 是否返回邻接矩阵
        :param return_emb: 是否返回适应后的实体关系嵌入
        :return: 根据参数设置返回关系得分、邻接矩阵以及可能的关系嵌入向量
        """
        # 根据是否返回实体嵌入，适配关系模块
        if not return_emb:
            s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb, return_adj=return_adj, return_emb=return_emb)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb, return_adj=return_adj, return_emb=return_emb)

        # 如果预测邻接矩阵，则使用它来调整查询实体的关系得分
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)

        # 根据参数设置返回计算结果
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        """
        前向传播函数：用于计算给定分子对(s_data, q_data)的关系得分。

        参数:
        - s_data: support分子的数据结构，包含分子的节点特征(x), 边索引(edge_index), 边属性(edge_attr), 和批次信息(batch)。
        - q_data: query分子的数据结构，格式同s_data。
        - s_label: support分子的真实标签，用于监督学习。默认为None，表示是无监督学习任务。
        - q_pred_adj: 是否预测分子对之间的调整矩阵。用于一些特定的下游任务，如分子属性预测。默认为False。

        返回值:
        - s_logits: support分子对query分子的关系得分。
        - q_logits: query分子对support分子的关系得分。
        - adj: 预测的分子对之间的调整矩阵，仅当q_pred_adj为True时返回。
        - s_node_emb: support分子的节点嵌入，用于后续处理或可视化。
        """

        # 使用分子编码器分别对support和query分子进行编码，得到整体和节点的嵌入。
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        # 将support和query分子的嵌入通过投影编码器进行映射，以便于关系预测。
        s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)

        # 使用关系前向传播模型计算分子对之间的关系得分，并可选地预测调整矩阵。
        s_logits, q_logits, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)

        return s_logits, q_logits, adj, s_node_emb

    def forward_query_list(self, s_data, q_data_list, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb_list = [self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)[0] for q_data in
                      q_data_list]

        q_logits_list, adj_list = [], []
        for q_emb in q_emb_list:
            s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit.detach())
            if adj is not None:
                sim_adj = adj[-1][:,0].detach()
                q_adj = sim_adj[:,-1]
                adj_list.append(q_adj)

        q_logits = torch.cat(q_logits_list, 0)
        adj_list = torch.cat(adj_list, 0)
        return s_logit.detach(),q_logits, adj_list

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        """
        前向传播查询加载器，用于处理支持集和查询集的数据，并计算相关预测结果。

        :param s_data: 支持集数据，包含分子的结构信息。
        :param q_loader: 查询集数据加载器，包含一系列查询分子的数据。
        :param s_label: 支持集的标签，可选参数，默认为None。
        :param q_pred_adj: 是否预测查询集分子之间的调整信息，默认为False。
        :return: 支持集的预测logits，查询集的预测logits，真实标签，调整信息列表，以及支持集和查询集的标签字典。
        """
        # 对支持集数据进行编码
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            # 将查询集数据移动到支持集embedding的设备上
            q_data = q_data.to(s_emb.device)
            y_true_list.append(q_data.y)
            # 对查询集数据进行编码
            q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            # 进行投影编码
            s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            # 计算关系预测的logits和调整信息
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit)
            if adj is not None:
                # 仅保存查询集分子之间的相似调整信息
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        # 整合所有查询集的logits和真实标签
        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)
        # 构建支持集和查询集标签的字典
        sup_labels={'support':s_data.y,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels