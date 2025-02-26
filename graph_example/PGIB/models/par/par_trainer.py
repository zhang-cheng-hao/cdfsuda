import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from torch_geometric.data import DataLoader

from graph_lib.models import Base_Meta_Trainer
from graph_lib.models.maml import MAML
from graph_lib.datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from graph_lib.utils import Logger
from sklearn.metrics import roc_auc_score, precision_score, recall_score


class Par_Meta_Trainer(Base_Meta_Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag=0):
        # 初始化训练和测试阶段的支持样本数
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query  # 查询样本数
        n_class = self.args.n_class  # 类别数

        # 如果处于非训练阶段，计算损失时使用支持集的数据
        if not train:
            # 将支持集的预测结果按照测试阶段的样本布局重塑
            pre = pred_dict['s_logits'].reshape(n_class * n_support_test * n_query, n_class)
            ans = batch_data['s_label'].repeat(n_query)
            # 计算交叉熵损失
            losses_adapt = self.criterion(pre, ans)
        else:
            # 训练阶段，根据flag值选择使用支持集还是查询集计算损失
            if flag:
                # flag为真，使用支持集的数据计算损失
                pre = pred_dict['s_logits'].reshape(n_class * n_support_train * n_query, n_class)
                ans = batch_data['s_label'].repeat(n_query)
                losses_adapt = self.criterion(pre, ans)
            else:
                # flag为假，直接使用查询集的预测结果和标签计算损失
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

        # 检查损失值是否有NaN或Inf，若有则将其设置为0以避免影响训练稳定性
        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            losses_adapt = torch.zeros_like(losses_adapt)

        # 如果设置了正则化邻接矩阵损失
        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].size(0)
            adj = pred_dict['adj'][-1]  # 获取预测的邻接矩阵
            # 根据训练或测试阶段，以及flag值，选择不同数据计算邻接矩阵损失
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * n_support
                    label_edge = model.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:, :, :-1, :-1].reshape((n_d, -1))
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    label_edge = model.label2edge(total_label)[:, :, -1, :-1]
                    pred_edge = adj[:, :, -1, :-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            # 计算预测的邻接矩阵与真实标签生成的邻接矩阵之间的均方误差损失
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = torch.zeros_like(adj_loss_val)

            # 将邻接矩阵损失添加到总损失中
            losses_adapt += self.args.reg_adj * adj_loss_val

        return losses_adapt

    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches = {}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id] = db

        for k in range(self.update_step):
            losses_eval = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)

                for inner_step in range(self.inner_update_step):
                    # 用支持集计算适应损失
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag=1)
                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                # 用查询集计算评估损失
                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag=0)

                losses_eval.append(loss_eval)

            losses_eval = torch.stack(losses_eval)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            print('Train Epoch:', self.train_epoch, ', train update step:', k, ', loss_eval:', losses_eval.item())

        return self.model.module

    def test_step(self):
        step_results = {'query_preds': [], 'query_labels': [], 'query_adj': [], 'task_index': [], 'precision': [],
                        'recall': []}
        auc_scores = []
        precision_scores = []
        recall_scores = []

        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            if self.update_step_test > 0:
                model.train()

                for i, batch in enumerate(adapt_data['data_loader']):
                    batch = batch.to(self.device)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                      'q_data': batch, 'q_label': None}

                    # adaptable_weights = self.get_adaptable_weights(model)
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)
                    # model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                    model.adapt(loss_adapt)
                    if i >= self.update_step_test - 1:
                        break

            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'], dim=-1).detach()[:, 1]
                y_true = pred_eval['labels']
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'], dim=-1).detach()[:, 1]
                    y_s_true = eval_data['s_label']
                    y_score = torch.cat([y_score, y_s_score])
                    y_true = torch.cat([y_true, y_s_true])

                y_true_cpu = y_true.cpu().numpy()
                y_score_cpu = y_score.cpu().numpy()

                # Compute AUC for each class and then average them
                auc = roc_auc_score(y_true_cpu, y_score_cpu, average='macro')

                # Compute Precision and Recall
                # Note: Adjust the threshold as needed for calculating precision and recall
                threshold = 0.5
                y_pred = (y_score_cpu > threshold).astype(int)
                precision = precision_score(y_true_cpu, y_pred, average='macro')
                recall = recall_score(y_true_cpu, y_pred, average='macro')

            auc_scores.append(auc)
            precision_scores.append(precision)
            recall_scores.append(recall)

            print('Test Epoch:', self.train_epoch, ', test for task:', task_id, ', AUC:', round(auc, 4), ', Precision:',
                  round(precision, 4), ', Recall:', round(recall, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score_cpu)
                step_results['query_labels'].append(y_true_cpu)
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])
                step_results['precision'].append(precision)
                step_results['recall'].append(recall)

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc, avg_auc)
        self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4), ', Precision_Avg:', np.mean(precision_scores),
              ', Recall_Avg:', np.mean(recall_scores))

        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

