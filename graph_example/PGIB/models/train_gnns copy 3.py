import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets
from load_dataset import get_dataset, get_dataloader, get_dataloaders
from Configures import data_args, train_args, model_args, mcts_args
from my_mcts import mcts
from tqdm import tqdm
from proto_join import join_prototypes_by_activations
from utils import PlotUtils
from torch_geometric.utils import to_networkx
from itertools import accumulate
from torch_geometric.datasets import MoleculeNet
import logging
import random
import matplotlib.pyplot as plt
import json


# 设置日志记录
def setup_logging(log_dir, log_file='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建格式器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器到 logger
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


# 优化后的模型保存函数
def save_best(ckpt_dir, epoch, gnnNets, model_name, model_type, eval_acc, is_best):
    """
    保存模型检查点。

    Args:
        ckpt_dir (str): 检查点目录。
        epoch (int): 当前训练轮数。
        gnnNets (GnnNets): 模型实例。
        model_name (str): 模型名称。
        model_type (str): 模型类型（'cont' 或 'var'）。
        eval_acc (float): 当前评估准确率。
        is_best (bool): 是否为最佳模型。
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # 将模型移至 CPU 以节省 GPU 内存
    gnnNets.to('cpu')

    # 定义状态字典
    state = {
        'net_state_dict': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }

    # 定义保存路径
    latest_pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest.pth"
    latest_ckpt_path = os.path.join(ckpt_dir, latest_pth_name)

    # 保存最新模型
    torch.save(state, latest_ckpt_path)
    logger.info(f"Saved latest model checkpoint at epoch {epoch} to {latest_ckpt_path}")

    # 如果是最佳模型，另外保存一份
    if is_best:
        best_pth_name = f"{model_name}_{model_type}_{model_args.readout}_best.pth"
        best_ckpt_path = os.path.join(ckpt_dir, best_pth_name)
        torch.save(state, best_ckpt_path)
        logger.info(f"Saved best model checkpoint with accuracy {eval_acc:.4f} to {best_ckpt_path}")

    # 将模型移回原设备
    gnnNets.to(model_args.device)


# 绘制并保存损失和准确率曲线
def plot_curves(history, experiment_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    # 绘制训练和验证损失曲线
    plt.figure()
    plt.plot(epochs, history['train_loss'], 'b', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(experiment_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Saved loss curves to {loss_plot_path}")

    # 绘制训练和验证准确率曲线
    plt.figure()
    plt.plot(epochs, history['train_acc'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(experiment_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Saved accuracy curves to {acc_plot_path}")


# 保存实验结果到JSON文件
def save_results(experiment_dir, results):
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved experiment results to {results_path}")


# 清理旧的日志文件（如果需要）
def clear_log(log_file_path):
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)
        logger.info(f"Removed existing log file at {log_file_path}")


# train for graph classification
def train_GC(model_type, experiment_dir, history):
    logger.info('Start loading data')
    source_dataset = get_dataset(data_args.dataset_dir, data_args.source_dataset_name, task=data_args.task,
                                 domain='source')
    target_dataset = get_dataset(data_args.dataset_dir, data_args.target_dataset_name, task=data_args.task,
                                 domain='target')

    input_dim = source_dataset.num_node_features
    output_dim = int(source_dataset.num_classes)

    dataloaders = get_dataloaders(source_dataset, target_dataset, train_args.batch_size, num_shots=data_args.num_shots,
                                  data_split_ratio={'source_val': 0.1, 'target_test': 0.2}, seed=42)

    logger.info('Start training model')

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(source_dataset)):
        avg_nodes += source_dataset[i].x.shape[0]
        avg_edge_index += source_dataset[i].edge_index.shape[1]

    avg_nodes /= len(source_dataset)
    avg_edge_index /= len(source_dataset)
    logger.info(f"Dataset: {data_args.source_dataset_name}")
    logger.info(
        f"Graphs: {len(source_dataset)}, Avg Nodes: {avg_nodes:.4f}, Avg Edges per Graph: {avg_edge_index / 2:.4f}")

    best_acc = 0.0
    data_size = len(source_dataset)

    # 确保检查点目录存在
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stop_count = 0

    data_indices = dataloaders['source_labeled_train'].dataset.indices

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()

            # prototype projection
            for i in range(gnnNets.model.prototype_vectors.shape[0]):
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i * 10, len(data_indices)):
                    data = source_dataset[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        logger.info('Projection of prototype completed')
                        break

            # prototype merge
            if train_args.share:
                threshold = round(output_dim * model_args.num_prototypes_per_class * (1 - train_args.merge_p))
                if gnnNets.model.prototype_vectors.shape[0] > threshold:
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,
                                                               dataloaders['source_labeled_train'], optimizer)
                    logger.info(f"Prototype merge completed: {join_info}")

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for i, batch in enumerate(dataloaders['source_labeled_train']):
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(
                    batch)

            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(
                    model_args.device)
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            # diversity loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[:, i])))
            prototype_numbers = list(accumulate(prototype_numbers))
            n = 0
            ld = 0

            for k in prototype_numbers:
                p = gnnNets.model.prototype_vectors[n: k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            if model_args.cont:
                loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight * connectivity_loss + train_args.alpha1 * KL_Loss
            else:
                loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight * connectivity_loss + train_args.alpha1 * KL_Loss

                # 优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            # 记录
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # 计算并记录训练指标
        avg_loss = np.average(loss_list)
        avg_ld_loss = np.average(ld_loss_list)
        avg_acc = np.concatenate(acc, axis=0).mean()
        logger.info(f"Train Epoch: {epoch} | Loss: {avg_loss:.3f} | Ld: {avg_ld_loss:.3f} | Acc: {avg_acc:.3f}")

        # 更新历史记录
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)

        # 评估
        eval_state = evaluate_GC(dataloaders['source_val'], gnnNets, criterion)
        logger.info(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")

        # 更新历史记录
        history['val_loss'].append(eval_state['loss'])
        history['val_acc'].append(eval_state['acc'])

        # 测试
        test_state, _, _ = test_GC(dataloaders['target_test'], gnnNets, criterion)
        logger.info(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")

        # 更新历史记录
        history['test_loss'].append(test_state['loss'])
        history['test_acc'].append(test_state['acc'])

        # 判断是否为最佳模型
        is_best = eval_state['acc'] > best_acc

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        else:
            early_stop_count += 1

        # 早停
        if early_stop_count > train_args.early_stopping:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # 保存模型
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, model_type, eval_state['acc'], is_best)

    logger.info(f"The best validation accuracy is {best_acc:.4f}")

    # 加载最佳模型并在测试集上评估
    best_model_path = os.path.join(ckpt_dir, f"{model_args.model_name}_{model_type}_{model_args.readout}_best.pth")
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path)
        gnnNets.to_device()
        test_state, _, _ = test_GC(dataloaders['target_test'], gnnNets, criterion)
        logger.info(
            f"Final Test | Dataset: {data_args.target_dataset_name} | Model: {model_args.model_name}_{model_type} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")

        # 更新历史记录
        history['final_test_loss'] = test_state['loss']
        history['final_test_acc'] = test_state['acc']
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}")

    return best_acc, test_state['acc'], history


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            if data_args.source_dataset_name.lower() == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)

            # 记录
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            logits, probs, active_node_index, _, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # 记录
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0))}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


if __name__ == '__main__':
    # 定义跨域任务
    transfer_tasks = [
        ('PROTEINS', 'MUTAG'),
        ('PROTEINS', 'BZR'),
        ('PROTEINS', 'COX2'),
        ('BZR', 'MUTAG'),
        ('BZR', 'PROTEINS'),
        ('BZR', 'COX2'),
        ('MUTAG', 'COX2'),
        ('MUTAG', 'BZR'),
        ('MUTAG', 'PROTEINS'),
        ('COX2', 'MUTAG'),
        ('COX2', 'BZR'),
        ('COX2', 'PROTEINS')
    ]
    # PROTEINS 3
    # MUTAG 7
    # BZR 53
    # COX2 35

    # few-shot设置
    n_shots = [1, 5]  # 1-shot和5-shot

    # 记录结果
    results = {}

    for source, target in transfer_tasks:
        for k in n_shots:
            # 创建实验目录
            experiment_name = f"{source}->{target}_{k}-shot"
            experiment_dir = os.path.join("experiments", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # 初始化日志
            logger = setup_logging(experiment_dir)

            logger.info(f"\n=== Transfer from {source} to {target}, {k}-shot ===")

            # 更新数据集配置
            data_args.source_dataset_name = source
            data_args.target_dataset_name = target
            data_args.num_shots = k

            # 初始化历史记录
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': [],
                'test_acc': []
            }

            # 训练模型并获取准确率
            best_val_acc, test_acc, history = train_GC('cont' if model_args.cont else 'var', experiment_dir, history)

            # 保存历史曲线
            plot_curves(history, experiment_dir)

            # 保存结果
            task_name = f"{source}->{target}"
            if task_name not in results:
                results[task_name] = {}
            results[task_name][f"{k}-shot"] = {
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'history': history
            }

            logger.info(f"Accuracy: {test_acc:.4f}")

    # 保存汇总结果到JSON
    summary_dir = "summary_results"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\n=== Summary of Results saved to {summary_path} ===")

    # 打印汇总结果
    print("\n=== Summary of Results ===")
    for task, scores in results.items():
        print(f"\n{task}:")
        for shot, acc in scores.items():
            print(f"{shot}: Best Val Acc: {acc['best_val_acc']:.4f}, Test Acc: {acc['test_acc']:.4f}")
