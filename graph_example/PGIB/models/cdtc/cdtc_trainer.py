import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score

from graph_lib.models import Base_Meta_Trainer
from graph_lib.models.cdtc.cdtc_coordinator import MetaTaskSelector

## 继承自trainer.py的Meta_Trainer
class CDTC_TrainerBase(Base_Meta_Trainer):
    def __init__(self, args, model, meta_task_selector):
        """
        初始化Meta_Trainer类的实例。

        参数:
        - args: 包含各种训练和模型参数的对象。
        - model: 要进行训练的模型。

        属性:
        - args: 训练和模型参数。
        - model: 经过MAML封装的模型，用于meta学习。
        - optimizer: 模型的优化器，使用AdamW优化器。
        - criterion: 损失函数，使用交叉熵损失。
        - dataset, test_dataset: 训练和测试用的数据集标识。
        - data_dir: 数据集的根目录。
        - train_tasks, test_tasks: 训练和测试任务的列表。
        - n_shot_train, n_shot_test: 训练和测试时的样本数。
        - n_query: 测试时的查询样本数。
        - n_class: 类别数。
        - device: 执行计算的设备。
        - emb_dim: 特征嵌入维度。
        - batch_task: 每个batch处理的任务数。
        - update_step, update_step_test: 更新步长。
        - inner_update_step: 内部更新步长。
        - trial_path: 试验结果的存储路径。
        - logger: 日志记录器，用于记录训练结果。
        - preload_train_data, preload_test_data: 预加载的训练和测试数据。
        - preload_valid_data: 预加载的验证数据（如果设置）。
        - train_epoch: 当前训练的epoch数。
        - best_auc: 训练过程中最好的AUC分数。
        - res_logs: 存储训练结果的列表。
        """
        super().__init__(args, model)
        # todo：设置cdtc
        self.meta_task_selector = meta_task_selector

    def compute_prototypes(self, embeddings, labels, num_classes):
        """
        Compute class prototypes from embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings of the support samples.
            labels (torch.Tensor): The labels of the support samples.
            num_classes (int): The number of classes.

        Returns:
            torch.Tensor: The computed class prototypes.
        """
        prototypes = torch.stack([embeddings[labels == i].mean(0) for i in range(num_classes)])
        return prototypes

    def prototypical_loss(self, query_embeddings, query_labels, class_prototypes):
        """
        Compute the prototypical loss with manual computation of squared Euclidean distance.

        Args:
            query_embeddings (torch.Tensor): The embeddings of the query samples, shape [num_queries, embedding_dim].
            query_labels (torch.Tensor): The labels of the query samples, shape [num_queries].
            class_prototypes (torch.Tensor): The class prototypes, shape [num_classes, embedding_dim].

        Returns:
            torch.Tensor: The prototypical loss.
        """
        # Expand dimensions for broadcasting to compute pairwise differences
        query_embeddings_exp = query_embeddings.unsqueeze(1)  # Shape: [num_queries, 1, embedding_dim]
        class_prototypes_exp = class_prototypes.unsqueeze(0)  # Shape: [1, num_classes, embedding_dim]

        # Compute squared differences
        differences = query_embeddings_exp - class_prototypes_exp  # Broadcasting
        squared_differences = differences ** 2

        # Sum over the embedding dimensions to get the squared Euclidean distance
        squared_distances = torch.sum(squared_differences, dim=2)  # Shape: [num_queries, num_classes]

        # Convert distances to logits (negative distance because logits are higher for closer distances)
        logits = -squared_distances

        # Compute the loss using cross-entropy
        loss = F.cross_entropy(logits, query_labels)
        return loss

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    # def get_prediction(self, model, data, train=True, flag=0):
    #     if not train:
    #         s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
    #         pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}
    #     else:
    #         # 训练阶段，根据flag值选择使用支持集还是查询集计算损失
    #         if flag:
    #             s_logits = model(data['s_data'],data['q_data'],data['s_label'])
    #             pred_dict = {'s_logits': s_logits}
    #         else:
    #             q_logits = model(data['q_data'])
    #             pred_dict = {'q_logits': q_logits}
    #     return pred_dict

    def cal_loss(self, pred_dict, batch_data, train=True, flag=0):
        """
        计算评估阶段的损失。

        参数:
        - pred_dict: 包含模型在评估数据上的预测信息的字典，通常包括logits等。
        - batch_data: 包含训练数据的信息字典，用于计算损失，通常包括标签等。

        返回:
        - loss_eval: 评估阶段的损失值。
        """
        # 初始化训练和测试阶段的支持样本数
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query  # 查询样本数
        n_class = self.args.n_class  # 类别数
        # 如果处于非训练阶段，计算损失时使用支持集的数据
        if not train:
            # 将支持集的预测结果按照测试阶段的样本布局重塑
            pre = pred_dict['logits']
            ans = pred_dict['labels']
            # 计算交叉熵损失
            losses_adapt = self.criterion(pre, ans)
        else:
            # 训练阶段，根据flag值选择使用支持集还是查询集计算损失
            if flag:
                # flag为真，使用支持集的数据计算损失
                # pre = pred_dict['s_logits'].reshape(n_support_train * n_class, n_class)
                pre = pred_dict['s_logits'].reshape(len(batch_data['s_label']), n_class)
                ans = batch_data['s_label']

                losses_adapt = self.criterion(pre, ans)
                # pre先过一遍softmax，用于debug
                pre = F.softmax(pre, dim=-1).detach()
            else:
                # flag为假，直接使用查询集的预测结果和标签计算损失
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
        return losses_adapt

    # 重载train_step
    def train_step(self):
        """
        执行一次CDTC（Continual Domain Transfer Clustering）的训练步骤。
        此函数不接受参数，也不返回任何值，但会更新模型的状态。

        步骤包括：
        1. 增加当前训练周期；
        2. 根据设置选择一批任务进行训练；
        3. 对选中的任务分别进行数据采样；
        4. 在多个更新步骤中，对每个任务进行适应和评估，更新模型参数；
        5. 输出当前训练周期、更新步骤和评估损失。

        """

        self.train_epoch += 1  # 增加当前训练周期数

        # 根据是否设置批量任务，选择一批任务进行训练
        task_id_list = list(range(len(self.train_tasks)))
        # if self.batch_task > 0:
        #     batch_task = min(self.batch_task, len(task_id_list))
        #     task_id_list = random.sample(task_id_list, batch_task) # todo： cdtc就是把随机采样改成由cdtc进行选择推荐


        # 为每个选中的任务采样数据
        source_data_batches = {}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            source_data_batches[task_id] = db
        target_data_batches = {}
        for task_id in range(len(self.test_tasks)):
            db = self.get_data_sample(task_id, train=False)
            target_data_batches[task_id] = db

        # cdtc的前向过程
        CDTC_optimizer = optim.Adam(self.meta_task_selector.parameters(), lr=0.001)
        task_id_list, selected_probabilities, selection_probabilities = self.meta_task_selector.select_tasks(source_data_batches, target_data_batches)

        task_id_list = task_id_list.tolist()
        selected_probabilities = selected_probabilities.tolist()
        selection_probabilities = selection_probabilities.tolist()

        # 打印选择的任务的列表
        print(task_id_list)
        # # 打印选中的任务的概率
        print(selected_probabilities)
        # # 打印选择所有任务的概率
        print(selection_probabilities)
        old_model = self.model.clone()
        # 对模型进行多次更新步骤
        for k in range(self.update_step):
            losses_eval = []  # 用于记录每个任务的评估损失
            for task_id in task_id_list:
                train_data, _ = source_data_batches[task_id]
                model = self.model.clone()  # 克隆模型以进行适应
                model.train()
                # adaptable_weights = self.get_adaptable_weights(model)  # 获取可适应的权重

                # 在内部更新步骤中，进行模型适应
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.cal_loss(pred_adapt, train_data, train=True, flag=1)
                    # model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                    model.adapt(loss_adapt)

                # 对模型进行评估，计算损失
                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.cal_loss(pred_eval, train_data, train=True, flag=0)
                losses_eval.append(loss_eval)  # 记录损失

            # 计算所有任务的平均评估损失，并进行反向传播更新模型
            losses_eval = torch.stack(losses_eval)
            losses_eval = torch.sum(losses_eval)
            losses_eval = losses_eval / len(task_id_list)
            self.train_loss_eval.append(losses_eval.item())
            self.optimizer.zero_grad()

            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 规范化梯度
            self.optimizer.step()  # 更新模型参数

            # 输出当前的训练信息
            print('Train Epoch:', self.train_epoch, ', train update step:', k, ', loss_eval:', losses_eval.item())



        # 在目标域上进行评估
        reward_list = []
        for task_id in range(len(self.test_tasks)):
            target_data, _ = target_data_batches[task_id]
            old_pred = self.get_prediction(old_model, target_data, train=False)
            new_pred = self.get_prediction(self.model, target_data, train=False)
            old_loss = self.cal_loss(old_pred, target_data, train=False)
            new_loss = self.cal_loss(new_pred, target_data, train=False)
            reward_list.append(new_loss.item() - old_loss.item())
        # 打印出奖励列表
        print(f"reward_list: {reward_list}")
        # 计算奖励平均值
        reward = np.mean(reward_list)
        self.meta_task_selector.update_parameters(CDTC_optimizer, reward, 0)
        return self.model.module  # 返回模型模块
        # Compute policy gradient using REINFORCE
        # policy_loss = []
        # for log_prob, R in zip(self.meta_task_selector.saved_log_probs, reward):
        #     policy_loss.append(-log_prob * R)
        # policy_loss = torch.cat(policy_loss).sum()
        #
        # # Update CDTC's parameters using policy gradient
        # CDTC_optimizer.zero_grad()
        # policy_loss.backward()
        # CDTC_optimizer.step()

        # meta_lr = 0.01
        # print("before update:")
        # # Step 1: Store parameters before update
        # prev_params = {name: param.clone() for name, param in self.meta_task_selector.named_parameters()}

        # Step 2: Update parameters
        # self.meta_task_selector.update_parameters(CDTC_optimizer, reward, 0)

        # Step 3: Compare and print changed parameters
        # for name, param in self.meta_task_selector.named_parameters():
        #     if not torch.equal(prev_params[name], param.data):
        #         print(f"Changed Parameter: {name}\nOld Value: {prev_params[name]}\nNew Value: {param.data}\n")

        # return self.model.module  # 返回模型模块

    # def cal_loss(self, pred_dict, batch_data, train=True, flag=0):
    #     """
    #     计算评估阶段的损失。
    #
    #     参数:
    #     - pred: 包含模型在评估数据上的预测信息的字典，通常包括logits等。
    #     - data: 包含训练数据的信息字典，用于计算损失，通常包括标签等。
    #
    #     返回:
    #     - loss_eval: 评估阶段的损失值。
    #     """
    #     # 初始化训练和测试阶段的支持样本数
    #     n_support_train = self.args.n_shot_train
    #     n_support_test = self.args.n_shot_test
    #     n_query = self.args.n_query  # 查询样本数
    #     n_class = self.args.n_class  # 类别数
    #     # if train == True:
    #     #     if flag == 1:
    #     #         # 计算评估数据的类原型
    #     #         class_prototypes_eval = self.compute_prototypes(pred_dict['s_logits'], batch_data['s_label'], n_class)
    #     #         # 使用类原型计算评估损失
    #     #         loss_eval = self.prototypical_loss(pred_dict['q_logits'], batch_data['q_label'], class_prototypes_eval)
    #     #         loss = loss_eval
    #     #     else:
    #     #         # 计算评估数据的类原型
    #     #         class_prototypes_adapt = self.compute_prototypes(pred_dict['s_logits'], batch_data['s_label'], n_class)
    #     #         # 使用类原型计算评估损失
    #     #         loss_adapt = self.prototypical_loss(pred_dict['s_logits'], batch_data['s_label'], class_prototypes_adapt)
    #     #         loss = loss_adapt
    #     # else:
    #     #     class_prototypes_adapt = self.compute_prototypes(pred_dict['s_logits'], batch_data['s_label'], n_class)
    #     #     # 使用类原型计算评估损失
    #     #     loss_adapt = self.prototypical_loss(pred_dict['s_logits'], batch_data['s_label'], class_prototypes_adapt)
    #     #     loss = loss_adapt
    #     # return loss
    #     # 如果处于非训练阶段，计算损失时使用支持集的数据
    #
    #     if not train:
    #         # 将支持集的预测结果按照测试阶段的样本布局重塑
    #         pre = pred_dict['s_logits'].reshape( n_support_test * n_class, n_class)
    #         ans = batch_data['s_label']
    #         # 计算交叉熵损失
    #         losses_adapt = self.criterion(pre, ans)
    #     else:
    #         # 训练阶段，根据flag值选择使用支持集还是查询集计算损失
    #         if flag:
    #             # flag为真，使用支持集的数据计算损失
    #             pre = pred_dict['s_logits'].reshape(n_support_train * n_class, n_class)
    #             ans = batch_data['s_label']
    #             losses_adapt = self.criterion(pre, ans)
    #         else:
    #             # flag为假，直接使用查询集的预测结果和标签计算损失
    #             losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
    #
    #     return losses_adapt
    # 重载test_step
    def test_step(self):
        """
        执行一次CDTC（类分布转移分类）测试步骤。
        此函数针对每个测试任务，进行模型适应和评估，计算AUC分数，并更新最佳AUC记录。

        返回:
            self.best_auc: 测试步骤中计算出的平均AUC分数中的最佳值。
        """
        # 初始化存储查询预测、标签、调整数据以及任务索引的结果字典
        step_results = {'query_preds': [], 'query_labels': [], 'query_adj': [], 'task_index': []}  # 用于存储每个任务的结果
        auc_scores = []  # 用于存储每个任务的AUC分数
        losses_eval = []

        for task_id in range(len(self.test_tasks)):
            # 获取适应数据和评估数据样本
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()  # 克隆模型

            # 如果设置了测试步骤更新次数，则进行模型适应
            if self.update_step_test > 0:
                model.train()
                # 迭代适应数据集的批次
                for i, batch in enumerate(adapt_data['data_loader']):
                    batch = batch.to(self.device)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                      'q_data': batch, 'q_label': None}

                    # pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    # # loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)
                    # class_prototypes_adapt = self.compute_prototypes(pred_adapt['s_logits'], cur_adapt_data['s_label'],
                    #                                                  self.args.n_class)
                    # loss_adapt = self.prototypical_loss(pred_adapt['s_logits'], cur_adapt_data['s_label'],
                    #                                     class_prototypes_adapt)
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    loss_adapt = self.cal_loss(pred_adapt, cur_adapt_data, train=True, flag=1)
                    # 根据损失进行模型适应
                    model.adapt(loss_adapt)

                    # 达到指定的更新步数后停止适应
                    if i >= self.update_step_test - 1:
                        break

            # 将模型设置为评估模式，这在进行预测时非常重要，因为它会禁用 dropout 和 batch normalization 等只在训练时有用的特性。
            model.eval()

            # 使用torch.no_grad()来停止对模型中参数的梯度计算，这样可以加速计算并减少内存使用，因为在评估模式下我们不需要进行反向传播。
            with torch.no_grad():
                # 获取模型预测结果，train=False表示我们现在是在评估模式下运行。
                pred_eval = self.get_prediction(model, eval_data, train=False)
                loss_eval = self.cal_loss(pred_eval, eval_data, train=False)
                # 把loss_eval 变成一个list，再平均，再加到loss_values里
                losses_eval.append(loss_eval)  # 记录损失
                # 应用softmax函数到模型的输出上，softmax用于将模型输出的logits转换成概率分布，dim=-1表示对最后一个维度进行操作。
                y_score = F.softmax(pred_eval['logits'], dim=-1).detach()  # 使用完整的概率数组

                # 获取真实标签，这些是我们将用来评估模型预测的正确性的标签。
                y_true = pred_eval['labels']
                # 打印预测结果和真实标签
                # print('pred_eval:', y_score)
                # print('true:', y_true)
                # 如果启用了支持集评估（一个可选功能，根据args.eval_support决定）
                if self.args.eval_support:
                    # 对支持集的输出logits也应用softmax，得到概率分布
                    y_s_score = F.softmax(pred_eval['s_logits'], dim=-1).detach()
                    # 获取支持集的真实标签
                    y_s_true = eval_data['s_label']
                    # 将查询集和支持集的预测结果概率合并，这是为了一起评估整体性能。
                    y_score = torch.cat([y_score, y_s_score], dim=0)  # 直接合并所有类的概率
                    # 同样，将查询集和支持集的真实标签合并。
                    y_true = torch.cat([y_true, y_s_true])

                # 使用roc_auc_score来计算AUC分数，这是一个常用的性能度量标准，用于评价分类器的性能。
                # 参数'average=macro'表示对每个类别的AUC进行简单平均，'multi_class=ovr'表示采用一对余方式处理多类别问题。
                auc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy(), average='macro', multi_class='ovr')

                # 将计算出的AUC分数添加到列表中，用于后续分析或平均计算。
                auc_scores.append(auc)

            # 打印并记录当前任务的AUC分数
            # print('Test Epoch:', self.train_epoch, ', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'])
                step_results['task_index'].append(self.test_tasks[task_id])
                # step_results['loss_values'].append(avg_task_loss)
                # step_results['auc_scores'].append(auc)

        # 打印输出AUC分数的列表
        print('Test Epoch:', self.train_epoch, ', AUCs:', auc_scores)
        # 计算AUC分数的中位数和平均数，并更新最佳AUC记录
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc, avg_auc)
        self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)


        # 打印并记录当前epoch的中位数AUC、平均AUC和最佳平均AUC
        print('Test Epoch:', self.train_epoch, ', AUC_Mid of last 10 epochs:', round(mid_auc, 4), ', AUC_Avg of last 10 epochs: ', round(avg_auc, 4),
              ', Best_Avg_AUC of all epochs: ', round(self.best_auc, 4), )

        if self.args.save_logs:
            self.res_logs.append(step_results)
            # self.auc_list append auc_scores的平均值
            self.auc_list.append(avg_auc)

            losses_eval = torch.stack(losses_eval)
            losses_eval = torch.sum(losses_eval)
            losses_eval = losses_eval / len(self.test_tasks)
            self.loss_list.append(losses_eval.item())
        return self.best_auc 