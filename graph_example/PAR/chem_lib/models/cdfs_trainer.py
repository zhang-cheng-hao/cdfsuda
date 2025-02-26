import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.loader import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

from sklearn.metrics import roc_auc_score, precision_score, recall_score
class CDFS_Trainer(nn.Module):
    def __init__(self, args, model):
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
        super(CDFS_Trainer, self).__init__()

        self.args = args
        # 使用MAML方法封装模型
        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        # 设置优化器和损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        # 初始化数据集和任务相关属性
        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query
        self.n_class = args.n_class
        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        # 更新步长相关属性
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        # 日志设置
        self.trial_path = args.trial_path
        self.loss_list = []
        self.auc_list = []
        self.train_loss_eval = []
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid', 'AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        # 预加载训练和测试数据
        preload_train_data = {}
        if args.preload_train_data:
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data

        # 如果设置，则预加载验证数据
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train', 'valid')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc = 0

        self.res_logs = []


    def loader_to_samples(self, data):
        """
        将数据加载器中的数据转换为设备上的样本。

        参数:
        - self: 对象自身的引用。
        - data: 要加载的数据集，通常是一个包含训练或测试数据的TensorDataset。

        返回值:
        - samples: 转换后位于指定设备上的样本数据。
        """
        # 删除data.data的edge_name_list, node_name_list属性
        if hasattr(data.data, 'edge_name_list'):
            del data.data.edge_name_list

        if hasattr(data.data, 'node_name_list'):
            del data.data.node_name_list

        # 创建一个数据加载器，以整个数据集为一个批次，不进行打乱，不使用额外的worker
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)

        for samples in loader:
            # 将样本数据移动到指定的设备上
            samples=samples.to(self.device)
            # 返回处理后的样本数据
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)
            # 使查询数据的标签偏移
            for batch in q_loader:
                batch.y += self.n_class
            for batch in q_loader_adapt:
                batch.y += self.n_class

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

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
    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

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

        # 为每个选中的任务采样数据
        source_data_batches = {}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            source_data_batches[task_id] = db
        target_data_batches = {}
        for task_id in range(len(self.test_tasks)):
            db = self.get_data_sample(task_id, train=False)
            target_data_batches[task_id] = db

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

        return self.model.module  # 返回模型模块
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
                # 打印所有输入的参数：
                            # 转换为NumPy数组并最终检查
                y_score_np = y_score.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                
                if np.isnan(y_score_np).any():
                    print(f"Warning: Still found NaN in predictions for task {task_id}")
                    y_score_np = np.nan_to_num(y_score_np, nan=1.0/self.args.n_class)
                
                try:
                    auc = roc_auc_score(y_true_np, y_score_np, 
                                    average='macro', 
                                    multi_class='ovr')
                    auc_scores.append(auc)
                except ValueError as e:
                    print(f"Error calculating AUC for task {task_id}: {e}")
                    print(f"y_true unique values: {np.unique(y_true_np)}")
                    print(f"y_score shape: {y_score_np.shape}")
                    print(f"y_true shape: {y_true_np.shape}")
                    # 使用一个默认的AUC分数
                    auc_scores.append(0.5)


                # 将计算出的AUC分数添加到列表中，用于后续分析或平均计算。
                # auc_scores.append(auc)

            # 打印并记录当前任务的AUC分数
            # print('Test Epoch:', self.train_epoch, ', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'])
                step_results['task_index'].append(self.test_tasks[task_id])

        # 打印输出AUC分数的列表
        print('Test Epoch:', self.train_epoch, ', AUCs:', auc_scores)
        # 计算AUC分数的中位数和平均数，并更新最佳AUC记录
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc, avg_auc)
        # self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)

        # 打印并记录当前epoch的中位数AUC、平均AUC和最佳平均AUC
        print('Test Epoch:', self.train_epoch, ', AUC_Mid of last 10 epochs:', round(mid_auc, 4), ', AUC_Avg of last 10 epochs: ', round(avg_auc, 4),
              ', Best_Avg_AUC of all epochs: ', round(self.best_auc, 4), )

        if avg_auc >= self.best_auc:
            self.save_model()  # 只有当当前的平均AUC分数是最好的时候才保存模型

        if self.args.save_logs:
            self.res_logs.append(step_results)
            # self.auc_list append auc_scores的平均值
            self.auc_list.append(avg_auc)

            losses_eval = torch.stack(losses_eval)
            losses_eval = torch.sum(losses_eval)
            losses_eval = losses_eval / len(self.test_tasks)
            self.loss_list.append(losses_eval.item())
        return self.best_auc
   
# class CDFS_Trainer(nn.Module):
#     def __init__(self, args, model):
#         super(CDFS_Trainer, self).__init__()

#         self.args = args

#         self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
#         self.criterion = nn.CrossEntropyLoss().to(args.device)

#         self.dataset = args.dataset
#         self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
#         self.tar_test_dataset = args.tar_test_dataset
#         self.data_dir = args.data_dir
#         self.train_tasks = args.train_tasks
#         self.test_tasks = args.test_tasks
#         self.tar_test_tasks = args.tar_test_tasks
#         self.n_shot_train = args.n_shot_train
#         self.n_shot_test = args.n_shot_test
#         self.n_query = args.n_query

#         self.device = args.device

#         self.emb_dim = args.emb_dim

#         self.batch_task = args.batch_task

#         self.update_step = args.update_step
#         self.update_step_test = args.update_step_test
#         self.inner_update_step = args.inner_update_step

#         self.trial_path = args.trial_path
#         trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
#         print(trial_name)
#         logger = Logger(self.trial_path + '/results.txt', title=trial_name)
#         log_names = ['Epoch']
#         log_names += ['AUC-' + str(t) for t in args.test_tasks]
#         log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
#         logger.set_names(log_names)
#         self.logger = logger

#         preload_train_data = {}
#         if args.preload_train_data:
#             print('preload train data')
#             for task in self.train_tasks:
#                 dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
#                                           dataset=self.dataset)
#                 preload_train_data[task] = dataset
#         preload_test_data = {}
#         if args.preload_test_data:
#             print('preload_test_data')
#             for task in self.test_tasks:
#                 dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
#                                           dataset=self.test_dataset)
#                 preload_test_data[task] = dataset
#         preload_tar_test_data = {}
#         print('preload_tar_test_tasks')
#         for task in self.tar_test_tasks:
#             dataset = MoleculeDataset(self.data_dir + self.tar_test_dataset + "/new/" + str(task + 1),
#                                       dataset=self.tar_test_dataset)
#             preload_tar_test_data[task] = dataset
#         self.preload_train_data = preload_train_data
#         self.preload_test_data = preload_test_data
#         self.preload_tar_test_data = preload_tar_test_data
#         if 'train' in self.dataset and args.support_valid:
#             val_data_name = self.dataset.replace('train','valid')
#             print('preload_valid_data')
#             preload_val_data = {}
#             for task in self.train_tasks:
#                 dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
#                                           dataset=val_data_name)
#                 preload_val_data[task] = dataset
#             self.preload_valid_data = preload_val_data

#         self.train_epoch = 0
#         self.best_auc=0 
        
#         self.res_logs=[]

#     def loader_to_samples(self, data):
#         loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
#         for samples in loader:
#             samples=samples.to(self.device)
#             return samples

#     def get_data_sample(self, task_id, train=True):
#         if train:
#             task = self.train_tasks[task_id]
#             if task in self.preload_train_data:
#                 dataset = self.preload_train_data[task]
#             else:
#                 dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

#             s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

#             s_data = self.loader_to_samples(s_data)
#             q_data = self.loader_to_samples(q_data)

#             adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
#                             'label': torch.cat([s_data.y, q_data.y], 0)}
#             eval_data = { }
#         else:
#             task = self.test_tasks[task_id]
#             if task in self.preload_test_data:
#                 dataset = self.preload_test_data[task]
#             else:
#                 dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
#                                           dataset=self.test_dataset)
#             s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
#             s_data = self.loader_to_samples(s_data)
#             q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
#             q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

#             adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
#             eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

#         return adapt_data, eval_data

#     def get_tar_data_sample(self, task_id):

#         task = self.tar_test_tasks[task_id]
#         if task in self.preload_tar_test_data:
#             dataset = self.preload_tar_test_data[task]
#         else:
#             dataset = MoleculeDataset(self.data_dir + self.tar_test_dataset + "/new/" + str(task + 1),
#                                       dataset=self.tar_test_dataset)
#         s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.tar_test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
#         s_data = self.loader_to_samples(s_data)
#         q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
#         q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

#         adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
#         eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

#         return adapt_data, eval_data

#     def get_prediction(self, model, data, train=True):
#         if train:
#             s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
#             pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

#         else:
#             s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
#             pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

#         return pred_dict

#     def get_adaptable_weights(self, model, adapt_weight=None):
#         if adapt_weight is None:
#             adapt_weight = self.args.adapt_weight
#         fenc = lambda x: x[0]== 'mol_encoder'
#         frel = lambda x: x[0]== 'adapt_relation'
#         fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
#         fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
#         fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
#         if adapt_weight==0:
#             flag=lambda x: not fenc(x)
#         elif adapt_weight==1:
#             flag=lambda x: not frel(x)
#         elif adapt_weight==2:
#             flag=lambda x: not (fenc(x) or frel(x))
#         elif adapt_weight==3:
#             flag=lambda x: not (fenc(x) or fedge(x))
#         elif adapt_weight==4:
#             flag=lambda x: not (fenc(x) or fnode(x))
#         elif adapt_weight==5:
#             flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
#         elif adapt_weight==6:
#             flag=lambda x: not (fenc(x) or fclf(x))
#         else:
#             flag= lambda x: True
#         if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
#             adaptable_weights = None
#         else:
#             adaptable_weights = []
#             adaptable_names=[]
#             for name, p in model.module.named_parameters():
#                 names=name.split('.')
#                 if flag(names):
#                     adaptable_weights.append(p)
#                     adaptable_names.append(name)
#         return adaptable_weights

#     def get_loss(self, model, batch_data, pred_dict, train=True, flag=0):
#         # 打印输入变量的值
#         # print("model:", model)
#         print("batch_data:", batch_data)
#         print("pred_dict:", pred_dict)
#         # print("train:", train) 
#         # print("flag:", flag)

#         n_support_train = self.args.n_shot_train
#         n_support_test = self.args.n_shot_test
#         n_query = self.args.n_query
#         n_class = self.args.n_class  # 类别数
#         # 打印出关键变量的形状
#         print("n_support_train:", n_support_train)
#         print("n_support_test:", n_support_test)
#         print("n_query:", n_query)
#         print("n_class:", n_class)
#         print("pred_dict['s_logits']:", pred_dict['s_logits'].shape)
#         print("pred_dict['q_logits']:", pred_dict['q_logits'].shape)
#         # 如果处于非训练阶段，计算损失时使用支持集的数据
#         if not train:
#             # 将支持集的预测结果按照测试阶段的样本布局重塑
#             pre = pred_dict['s_logits'].reshape(n_class * n_support_test, n_class)
#             # ans = batch_data['s_label'].repeat(n_query)
#             ans = pred_dict['s_label']
#             # 计算交叉熵损失
#             losses_adapt = self.criterion(pre, ans)
#         else:
#             # 训练阶段，根据flag值选择使用支持集还是查询集计算损失
#             if flag:
#                 # flag为真，使用支持集的数据计算损失
#                 pre = pred_dict['s_logits'].reshape(n_class * n_support_train, n_class)
#                 # ans = batch_data['s_label'].repeat(n_query)
#                 ans = pred_dict['s_label']
#                 losses_adapt = self.criterion(pre, ans)
#             else:
#                 # flag为假，直接使用查询集的预测结果和标签计算损失
#                 losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

#         if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
#             print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
#             print(pred_dict['s_logits'])
#             losses_adapt = torch.zeros_like(losses_adapt)
#         if self.args.reg_adj > 0:
#             n_support = batch_data['s_label'].size(0)
#             adj = pred_dict['adj'][-1]
#             if train:
#                 if flag:
#                     s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
#                     n_d = n_query * n_support
#                     label_edge = model.label2edge(s_label).reshape((n_d, -1))
#                     pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
#                 else:
#                     s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
#                     q_label = batch_data['q_label'].unsqueeze(1)
#                     total_label = torch.cat((s_label, q_label), 1)
#                     label_edge = model.label2edge(total_label)[:,:,-1,:-1]
#                     pred_edge = adj[:,:,-1,:-1]
#             else:
#                 s_label = batch_data['s_label'].unsqueeze(0)
#                 n_d = n_support
#                 label_edge = model.label2edge(s_label).reshape((n_d, -1))
#                 pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
#             adj_loss_val = F.mse_loss(pred_edge, label_edge)
#             if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
#                 print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
#                 adj_loss_val = torch.zeros_like(adj_loss_val)

#             losses_adapt += self.args.reg_adj * adj_loss_val

#         return losses_adapt

#     def train_step(self):

#         self.train_epoch += 1

#         task_id_list = list(range(len(self.train_tasks)))
#         if self.batch_task > 0:
#             batch_task = min(self.batch_task, len(task_id_list))
#             task_id_list = random.sample(task_id_list, batch_task)
#         data_batches={}
#         for task_id in task_id_list:
#             db = self.get_data_sample(task_id, train=True)
#             data_batches[task_id]=db

#         for k in range(self.update_step):
#             losses_eval = []
#             for task_id in task_id_list:
#                 train_data, _ = data_batches[task_id]
#                 model = self.model.clone()
#                 model.train()
#                 adaptable_weights = self.get_adaptable_weights(model)
                
#                 for inner_step in range(self.inner_update_step):
#                     pred_adapt = self.get_prediction(model, train_data, train=True)
#                     loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
#                     model.adapt(loss_adapt, adaptable_weights = adaptable_weights)

#                 pred_eval = self.get_prediction(model, train_data, train=True)
#                 loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)

#                 losses_eval.append(loss_eval)

#             losses_eval = torch.stack(losses_eval)

#             losses_eval = torch.sum(losses_eval)

#             losses_eval = losses_eval / len(task_id_list)
#             self.optimizer.zero_grad()
#             losses_eval.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
#             self.optimizer.step()

#             print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

#         return self.model.module

#     def test_step(self):
#         step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
#         auc_scores = []
#         for task_id in range(len(self.test_tasks)):
#             adapt_data, eval_data = self.get_data_sample(task_id, train=False)
#             model = self.model.clone()
#             if self.update_step_test>0:
#                 model.train()
                
#                 for i, batch in enumerate(adapt_data['data_loader']):
#                     batch = batch.to(self.device)
#                     cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
#                                         'q_data': batch, 'q_label': None}

#                     adaptable_weights = self.get_adaptable_weights(model)
#                     pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
#                     loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

#                     model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

#                     if i>= self.update_step_test-1:
#                         break

#             model.eval()
#             with torch.no_grad():
#                 pred_eval = self.get_prediction(model, eval_data, train=False)
#                 y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
#                 y_true = pred_eval['labels']
#                 if self.args.eval_support:
#                     y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
#                     y_s_true = eval_data['s_label']
#                     y_score=torch.cat([y_score, y_s_score])
#                     y_true=torch.cat([y_true, y_s_true])
#                 auc = auroc(y_score, y_true, task="binary").item()

#             auc_scores.append(auc)

#             print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
#             if self.args.save_logs:
#                 step_results['query_preds'].append(y_score.cpu().numpy())
#                 step_results['query_labels'].append(y_true.cpu().numpy())
#                 step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
#                 step_results['task_index'].append(self.test_tasks[task_id])

#         mid_auc = np.median(auc_scores)
#         avg_auc = np.mean(auc_scores)
#         self.best_auc = max(self.best_auc,avg_auc)
#         self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

#         print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
#               ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
#         if self.args.save_logs:
#             self.res_logs.append(step_results)

#         return self.best_auc

#     def tar_test_step(self):
#         step_results = {'query_preds': [], 'query_labels': [], 'query_adj': [], 'task_index': []}
#         auc_scores = []
#         for task_id in range(len(self.tar_test_tasks)):
#             adapt_data, eval_data = self.get_tar_data_sample(task_id)
#             model = self.model.clone()
#             if self.update_step_test > 0:
#                 model.train()

#                 for i, batch in enumerate(adapt_data['data_loader']):
#                     batch = batch.to(self.device)
#                     cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
#                                       'q_data': batch, 'q_label': None}

#                     adaptable_weights = self.get_adaptable_weights(model)
#                     pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
#                     loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

#                     model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

#                     if i >= self.update_step_test - 1:
#                         break

#             model.eval()
#             with torch.no_grad():
#                 pred_eval = self.get_prediction(model, eval_data, train=False)
#                 y_score = F.softmax(pred_eval['logits'], dim=-1).detach()[:, 1]
#                 y_true = pred_eval['labels']
#                 if self.args.eval_support:
#                     y_s_score = F.softmax(pred_eval['s_logits'], dim=-1).detach()[:, 1]
#                     y_s_true = eval_data['s_label']
#                     y_score = torch.cat([y_score, y_s_score])
#                     y_true = torch.cat([y_true, y_s_true])
#                 auc = auroc(y_score, y_true, task="binary").item()

#             auc_scores.append(auc)

#             print('TAR Test Epoch:', self.train_epoch, ',tar test for task:', task_id, ', AUC:', round(auc, 4))
#             if self.args.save_logs:
#                 step_results['query_preds'].append(y_score.cpu().numpy())
#                 step_results['query_labels'].append(y_true.cpu().numpy())
#                 step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
#                 step_results['task_index'].append(self.tar_test_tasks[task_id])

#         mid_auc = np.median(auc_scores)
#         avg_auc = np.mean(auc_scores)
#         self.best_auc = max(self.best_auc, avg_auc)
#         # self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)

#         print('TAR Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
#               ', Best_Avg_AUC: ', round(self.best_auc, 4), )

#         if self.args.save_logs:
#             self.res_logs.append(step_results)

#         return avg_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
