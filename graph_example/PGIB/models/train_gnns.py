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
import time
from sklearn.metrics import roc_auc_score  # Added import for ROC-AUC

# 设置日志记录
def setup_logging(log_dir, log_file='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # 创建独立的 logger
    logger = logging.getLogger(log_dir)  # 使用 log_dir 作为 logger 名称，确保每个实验有独立的 logger
    logger.setLevel(logging.INFO)

    # 清除旧的处理器，以避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

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
def save_best(ckpt_dir, epoch, gnnNets, model_name, model_type, eval_auc, is_best, logger):
    """
    保存模型检查点。

    Args:
        ckpt_dir (str): 检查点目录。
        epoch (int): 当前训练轮数。
        gnnNets (GnnNets): 模型实例。
        model_name (str): 模型名称。
        model_type (str): 模型类型（'cont' 或 'var'）。
        eval_auc (float): 当前评估ROC-AUC。
        is_best (bool): 是否为最佳模型。
        logger (logging.Logger): 日志记录器。
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    gnnNets.to('cpu')

    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'auc': eval_auc  # Changed from 'acc' to 'auc'
    }

    # 保存最新模型
    latest_pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest.pth"
    latest_ckpt_path = os.path.join(ckpt_dir, latest_pth_name)
    torch.save(state, latest_ckpt_path)
    logger.info(f"Saved latest model checkpoint at epoch {epoch} to {latest_ckpt_path}")

    # 如果是最佳模型，另外保存一份
    if is_best:
        best_pth_name = f"{model_name}_{model_type}_{model_args.readout}_best.pth"
        best_ckpt_path = os.path.join(ckpt_dir, best_pth_name)
        torch.save(gnnNets, best_ckpt_path)  # 保存整个模型对象
        logger.info(f"Saved best model checkpoint with AUC {eval_auc:.4f} to {best_ckpt_path}")

    # 将模型移回原设备
    gnnNets.to(model_args.device)

# 绘制并保存损失和ROC-AUC曲线
def plot_curves(history, experiment_dir, logger):
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

    # 绘制训练和验证ROC-AUC曲线
    plt.figure()
    plt.plot(epochs, history['train_auc'], 'b', label='Training AUC')
    plt.plot(epochs, history['val_auc'], 'r', label='Validation AUC')
    plt.title('Training and Validation ROC-AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.legend()
    auc_plot_path = os.path.join(experiment_dir, 'auc_curve.png')
    plt.savefig(auc_plot_path)
    plt.close()
    logger.info(f"Saved ROC-AUC curves to {auc_plot_path}")

# 保存实验结果到JSON文件
def save_results(experiment_dir, results, logger):
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved experiment results to {results_path}")

# train for graph classification
def train_GC(model_type, experiment_dir, history, logger):
    logger.info('Start loading data')
    source_dataset = get_dataset(data_args.dataset_dir, data_args.source_dataset_name, task=data_args.task, domain='source')
    target_dataset = get_dataset(data_args.dataset_dir, data_args.target_dataset_name, task=data_args.task, domain='target')

    input_dim = source_dataset.num_node_features
    output_dim = int(source_dataset.num_classes)

    dataloaders = get_dataloaders(
        source_dataset,
        target_dataset,
        train_args.batch_size,
        num_shots=data_args.num_shots,
        data_split_ratio={'source_val': 0.1, 'target_test': 0.2},
        seed=42
    )

    logger.info('Start training model')

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    gnnNets.to(model_args.device)
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
    logger.info(f"Graphs: {len(source_dataset)}, Avg Nodes: {avg_nodes:.4f}, Avg Edges per Graph: {avg_edge_index / 2:.4f}")

    best_auc = 0.0  # Changed from best_acc to best_auc
    data_size = len(source_dataset)

    # 确保检查点目录存在
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stop_count = 0

    data_indices = dataloaders['source_labeled_train'].dataset.indices

    for epoch in range(train_args.max_epochs):
        all_train_labels = []
        all_train_probs = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            logger.info(f"Epoch {epoch}: Starting prototype projection")
            gnnNets.eval()

            # Prototype Projection
            start_time = time.time()
            for i in range(gnnNets.model.prototype_vectors.shape[0]):
                logger.info(f"Epoch {epoch}: Projecting prototype {i}")
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                max_j = min(i * 10 + 1000, len(data_indices))  # 限制每个原型最多遍历 1000 个数据点
                for j in range(i * 10, max_j):
                    data = source_dataset[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        logger.info(f'Epoch {epoch}: Projection of prototype {i} completed')
                        break
            logger.info(f"Epoch {epoch}: Prototype projection completed in {time.time() - start_time:.2f} seconds")

            # Prototype Merge
            if train_args.share:
                logger.info(f"Epoch {epoch}: Starting prototype merge")
                threshold = round(output_dim * model_args.num_prototypes_per_class * (1 - train_args.merge_p))
                if gnnNets.model.prototype_vectors.shape[0] > threshold:
                    start_time = time.time()
                    join_info = join_prototypes_by_activations(
                        gnnNets,
                        train_args.proto_percnetile,
                        dataloaders['source_labeled_train'],
                        optimizer
                    )
                    logger.info(f"Epoch {epoch}: Prototype merge completed: {join_info} in {time.time() - start_time:.2f} seconds")

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for i, batch in enumerate(dataloaders['source_labeled_train']):
            batch = batch.to(model_args.device)  # 确保数据在正确的设备上
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch)

            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device)
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            # Diversity Loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[:, i])))
            prototype_numbers = list(accumulate(prototype_numbers))
            n = 0
            ld = 0

            for k in prototype_numbers:
                p = gnnNets.model.prototype_vectors[n: k]
                n = k
                if p.shape[0] == 0:
                    continue
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

            # Collect labels and probabilities for ROC-AUC
            all_train_labels.append(batch.y.cpu().numpy())
            all_train_probs.append(probs.cpu().detach().numpy())

            # 记录
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())

        # Concatenate all labels and probabilities
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        all_train_probs = np.concatenate(all_train_probs, axis=0)

        # Compute ROC-AUC
        if all_train_probs.shape[1] == 2:
            # Binary classification
            all_train_probs = all_train_probs[:, 1]  # Probability of the positive class
            train_auc = roc_auc_score(all_train_labels, all_train_probs)
        else:
            # Multi-class classification
            train_auc = roc_auc_score(all_train_labels, all_train_probs, multi_class='ovr', average='macro')

        avg_loss = np.average(loss_list)
        avg_ld_loss = np.average(ld_loss_list)
        logger.info(f"Train Epoch: {epoch} | Loss: {avg_loss:.3f} | Ld: {avg_ld_loss:.3f} | AUC: {train_auc:.3f}")

        # 更新历史记录
        history['train_loss'].append(avg_loss)
        history['train_auc'].append(train_auc)

        # 评估
        eval_state = evaluate_GC(dataloaders['source_val'], gnnNets, criterion)
        logger.info(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | AUC: {eval_state['auc']:.3f}")

        # 更新历史记录
        history['val_loss'].append(eval_state['loss'])
        history['val_auc'].append(eval_state['auc'])

        # 测试
        test_state, _, _ = test_GC(dataloaders['target_test'], gnnNets, criterion)
        logger.info(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | AUC: {test_state['auc']:.3f}")

        # 更新历史记录
        history['test_loss'].append(test_state['loss'])
        history['test_auc'].append(test_state['auc'])

        # 判断是否为最佳模型
        is_best = eval_state['auc'] > best_auc  # Changed from 'acc' to 'auc'

        if is_best:
            best_auc = eval_state['auc']
            early_stop_count = 0
        else:
            early_stop_count += 1

        # 早停
        if early_stop_count > train_args.early_stopping:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # 保存模型
        save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, model_type, eval_state['auc'], is_best, logger)

    logger.info(f"The best validation AUC is {best_auc:.4f}")

    # 加载最佳模型并在测试集上评估
    best_model_path = os.path.join(ckpt_dir, f"{model_args.model_name}_{model_type}_{model_args.readout}_best.pth")
    if os.path.isfile(best_model_path):
        gnnNets = torch.load(best_model_path)  # 加载整个模型
        gnnNets.to(model_args.device)
        test_state, _, _ = test_GC(dataloaders['target_test'], gnnNets, criterion)
        logger.info(f"Final Test | Dataset: {data_args.target_dataset_name} | Model: {model_args.model_name}_{model_type} | Loss: {test_state['loss']:.3f} | AUC: {test_state['auc']:.3f}")

        # 更新历史记录
        history['final_test_loss'] = test_state['loss']
        history['final_test_auc'] = test_state['auc']
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}")

    return best_auc, test_state['auc'], history

def evaluate_GC(eval_dataloader, gnnNets, criterion):
    all_labels = []
    all_probs = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(model_args.device)
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            if data_args.source_dataset_name.lower() == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)

            # Collect labels and probabilities
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().detach().numpy())

            loss_list.append(loss.item())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if all_probs.shape[1] == 2:
        # Binary classification
        all_probs = all_probs[:, 1]
        auc = roc_auc_score(all_labels, all_probs)
    else:
        # Multi-class classification
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    eval_state = {'loss': np.average(loss_list),
                  'auc': auc}

    return eval_state

def test_GC(test_dataloader, gnnNets, criterion):
    all_labels = []
    all_probs = []
    loss_list = []
    gnnNets.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            batch = batch.to(model_args.device)
            logits, probs, active_node_index, _, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # Collect labels and probabilities
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().detach().numpy())

            loss_list.append(loss.item())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if all_probs.shape[1] == 2:
        # Binary classification
        all_probs = all_probs[:, 1]
        auc = roc_auc_score(all_labels, all_probs)
    else:
        # Multi-class classification
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    test_state = {'loss': np.average(loss_list),
                  'auc': auc}

    return test_state, all_probs, all_labels

if __name__ == '__main__':
    # 定义跨域任务
    transfer_tasks = [
        ('PROTEINS', 'COX2'),
        ('PROTEINS', 'MUTAG'),
        ('PROTEINS', 'BZR'),
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
    n_shots = [3, 6, 9]  # 1-shot和5-shot

    # 记录结果
    results = {}

    # Number of repetitions per experiment
    num_repeats = 3

    for source, target in transfer_tasks:
        for k in n_shots:
            # Initialize lists to store AUCs for each repeat
            aucs = []
            val_aucs = []
            final_test_aucs = []

            for repeat in range(1, num_repeats + 1):
                # 创建实验目录
                experiment_name = f"{source}->{target}_{k}-shot_run{repeat}"
                experiment_dir = os.path.join("experiments", experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)

                # 初始化日志
                logger = setup_logging(experiment_dir)

                logger.info(f"\n=== Transfer from {source} to {target}, {k}-shot, Run {repeat} ===")

                # Update data configuration
                data_args.source_dataset_name = source
                data_args.target_dataset_name = target
                data_args.num_shots = k

                # Set different seeds for each repeat to ensure different initializations
                seed = 42 + repeat  # Example: 43, 44, 45
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Initialize history
                history = {
                    'train_loss': [],
                    'train_auc': [],
                    'val_loss': [],
                    'val_auc': [],
                    'test_loss': [],
                    'test_auc': []
                }

                # Train the model and get AUCs
                try:
                    best_val_auc, test_auc, history = train_GC(
                        'cont' if model_args.cont else 'var',
                        experiment_dir,
                        history,
                        logger
                    )
                    val_aucs.append(best_val_auc)
                    final_test_aucs.append(test_auc)
                    logger.info(f"Run {repeat} | Best Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f}")
                except Exception as e:
                    logger.error(f"Error during training: {e}")
                    continue  # Skip to the next repeat

                # Plot curves
                try:
                    plot_curves(history, experiment_dir, logger)
                except Exception as e:
                    logger.error(f"Error while plotting curves: {e}")

                # Optionally save individual run results
                try:
                    run_results = {
                        'best_val_auc': best_val_auc,
                        'test_auc': test_auc,
                        'history': history
                    }
                    run_results_path = os.path.join(experiment_dir, 'run_results.json')
                    with open(run_results_path, 'w') as f:
                        json.dump(run_results, f, indent=4)
                    logger.info(f"Saved run results to {run_results_path}")
                except Exception as e:
                    logger.error(f"Error while saving run results: {e}")

            # After all repeats, compute mean and std
            if final_test_aucs:
                mean_val_auc = np.mean(val_aucs)
                std_val_auc = np.std(val_aucs)
                mean_test_auc = np.mean(final_test_aucs)
                std_test_auc = np.std(final_test_aucs)
                # Format as a±b%
                val_auc_str = f"{mean_val_auc:.4f}±{std_val_auc:.4f}"
                test_auc_str = f"{mean_test_auc:.4f}±{std_test_auc:.4f}"
            else:
                val_auc_str = "N/A"
                test_auc_str = "N/A"

            # Save aggregated results
            try:
                task_name = f"{source}->{target}"
                if task_name not in results:
                    results[task_name] = {}
                results[task_name][f"{k}-shot"] = {
                    'val_aucs': val_aucs,
                    'test_aucs': final_test_aucs,
                    'val_auc_mean_std': val_auc_str,
                    'test_auc_mean_std': test_auc_str
                }
            except Exception as e:
                logger.error(f"Error while saving aggregated results: {e}")

            # Log aggregated AUCs
            logger.info(f"Aggregated AUCs for {source}->{target}, {k}-shot:")
            logger.info(f"Validation AUC: {val_auc_str}")
            logger.info(f"Test AUC: {test_auc_str}")

    # 保存汇总结果到JSON
    try:
        summary_dir = "summary_results"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n=== Summary of Results saved to {summary_path} ===")
    except Exception as e:
        print(f"Error while saving summary results: {e}")

    # 打印汇总结果
    print("\n=== Summary of Results ===")
    for task, scores in results.items():
        print(f"\n{task}:")
        for shot, acc in scores.items():
            print(f"{shot}: Best Val AUC: {acc['val_auc_mean_std']}, Test AUC: {acc['test_auc_mean_std']}")

