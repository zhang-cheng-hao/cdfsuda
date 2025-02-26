import argparse
import os
import json
import logging
import random
import shutil
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd

from meta_model import Meta_model


def setup_logging(log_dir, log_file='train.log'):
    """
    Sets up logging to file and console.

    Args:
        log_dir (str): Directory to save the log file.
        log_file (str): Name of the log file.

    Returns:
        logger (logging.Logger): Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the logger
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seeds(seed):
    """
    Sets the random seeds for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_results(auc_list):
    """
    Calculates the mean and standard deviation of AUCs.

    Args:
        auc_list (list): List of AUC scores.

    Returns:
        mean_auc (float): Mean AUC.
        std_auc (float): Standard deviation of AUC.
    """
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    return mean_auc, std_auc


def save_json(data, path):
    """
    Saves a dictionary as a JSON file.

    Args:
        data (dict): Data to save.
        path (str): File path.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def run_experiment(args, experiment_dir, logger):
    """
    Runs a single experiment based on provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        experiment_dir (str): Directory to save experiment logs and results.
        logger (logging.Logger): Logger for the experiment.

    Returns:
        best_val_auc (float): Best validation AUC achieved.
        final_test_auc (float): Final test AUC achieved.
    """
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Adjust number of tasks based on dataset
    if args.dataset == "tox21":
        args.num_tasks = 12
        args.num_train_tasks = 9
        args.num_test_tasks = 3
    elif args.dataset in ['BZR', 'COX2', 'MUTAG', 'PROTEINS']:
        args.num_tasks = 4
        args.num_train_tasks = 3
        args.num_test_tasks = 1
    elif args.dataset == "sider":
        args.num_tasks = 27
        args.num_train_tasks = 21
        args.num_test_tasks = 6
    else:
        raise ValueError("Invalid dataset name.")

    if args.tar_test_dataset in ['BZR', 'COX2', 'MUTAG', 'PROTEINS']:
        args.num_tar_tasks = 4
        args.num_tar_test_tasks = 1

    # Initialize model
    model = Meta_model(args).to(device)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target Test Dataset: {args.tar_test_dataset}")
    logger.info(f"Number of Support Shots: {args.m_support}")
    logger.info(f"GNN Type: {args.gnn_type}")
    logger.info(f"Add Similarity: {args.add_similarity}")
    logger.info(f"Add Self-Supervise: {args.add_selfsupervise}")
    logger.info(f"Add Masking: {args.add_masking}")
    logger.info(f"Add Weight: {args.add_weight}")
    logger.info(f"Update Steps: {args.update_step}")

    best_val_aucs = []
    best_val_accs = []
    final_test_aucs = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"----------------- Epoch: {epoch} -----------------")
        support_grads = model(epoch)

        if epoch % 1 == 0:
            logger.info(f"Evaluation on epoch {epoch}")
            accs,aucs = model.test(support_grads)
            if best_val_accs != []:
                for acc_num in range(len(best_val_accs)):
                    if best_val_accs[acc_num] < accs[acc_num]:
                        best_val_accs[acc_num] = accs[acc_num]
            else:
                best_val_accs = accs
            if best_val_aucs != []:
                for auc_num in range(len(best_val_aucs)):
                    if best_val_aucs[auc_num] < accs[auc_num]:
                        best_val_aucs[auc_num] = accs[auc_num]
            else:
                best_val_aucs = aucs
        # if epoch % args.save_steps == 0:
        #     model.save_model()
        #     logger.info(f"Model saved at epoch {epoch}")

        # Log training progress
        # logger.info(f"Epoch {epoch} completed. Current Best Val AUC: {best_val_auc:.4f}")

    # Final Test
    final_test_accs,final_test_aucs = model.final_test(support_grads)
    logger.info('Training completed.')
    # logger.info(f'Best Avg AUC: {best_val_auc:.4f}')
    # logger.info(f'Final Test AUC: {final_test_auc:.4f}')

    # model.conclude()

    if args.filename:
        result_path = os.path.join(experiment_dir, f"{args.filename}.txt")
    else:
        result_path = os.path.join(experiment_dir, "results.txt")

    # Save results
    # with open(result_path, "a") as fw:
    #     fw.write("test: " + "\t")
    #     for acc in best_val_accs:
    #         fw.write(f"{acc:.4f}\t")
    #     fw.write("\n")
    # logger.info(f"Saved results to {result_path}")

    return best_val_aucs, final_test_aucs


def main():
    # Define Grid Search Configuration
    grid_search_config = {   
        'dataset': ['BZR', 'COX2', 'MUTAG', 'PROTEINS', 'tox21'],  
        'm_support': [1, 2, 3, 6, 9],  
    }  
    
    # Generate Valid Combinations
    valid_combinations = [  
        {  
            'dataset': dataset,  
            'tar_test_dataset': tar_test_dataset,  
            'm_support': m_support,  
        }  
        for dataset in grid_search_config['dataset']  
        for tar_test_dataset in grid_search_config['dataset']   
        if tar_test_dataset != dataset  
        for m_support in grid_search_config['m_support']  
    ]

    # Number of repetitions per experiment
    num_repeats = 2

    # Directory to save all experiments
    experiments_root = "experiments"
    os.makedirs(experiments_root, exist_ok=True)

    # Summary of all results
    summary_results = {}

    for combo in valid_combinations:
        dataset = combo['dataset']
        tar_test_dataset = combo['tar_test_dataset']
        m_support = combo['m_support']

        # Initialize lists to store AUCs for each repetition
        best_val_aucs = []
        final_test_aucs = []

        for repeat in range(1, num_repeats + 1):
            # Define experiment name and directory
            experiment_name = f"{dataset}_train{m_support}_tar_test{tar_test_dataset}_run{repeat}"
            experiment_dir = os.path.join(experiments_root, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Set up logging
            logger = setup_logging(experiment_dir)
            logger.info(f"\n=== Experiment: {experiment_name} ===")

            # Initialize arguments
            parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
            parser.add_argument('--device', type=int, default=0,
                                help='which gpu to use if any (default: 0)')
            parser.add_argument('--batch_size', type=int, default=5,
                                help='input batch size for training (default: 32)') 
            parser.add_argument('--epochs', type=int, default=30,
                                help='number of epochs to train (default: 100)')
            parser.add_argument('--lr', type=float, default=0.001,
                                help='learning rate (default: 0.001)')
            parser.add_argument('--lr_scale', type=float, default=1,
                                help='relative learning rate for the feature extraction layer (default: 1)')
            parser.add_argument('--decay', type=float, default=0,
                                help='weight decay (default: 0)')
            parser.add_argument('--num_layer', type=int, default=5,
                                help='number of GNN message passing layers (default: 5).')
            parser.add_argument('--emb_dim', type=int, default=300,
                                help='embedding dimensions (default: 300)')
            parser.add_argument('--dropout_ratio', type=float, default=0.5,
                                help='dropout ratio (default: 0.5)')
            parser.add_argument('--graph_pooling', type=str, default="mean",
                                help='graph level pooling (sum, mean, max, set2set, attention)')
            parser.add_argument('--JK', type=str, default="last",
                                help='how the node features across layers are combined. last, sum, max or concat')
            parser.add_argument('--gnn_type', type=str, default="graphsage")
            parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
            parser.add_argument('--tar-test-dataset', type=str, default = 'sider', help='')

            parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
            parser.add_argument('--filename', type=str, default = '', help='output filename')
            parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
            parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
            parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
            parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
            parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')

            parser.add_argument('--num_tasks', type=int, default=12, help = "# of tasks")
            parser.add_argument('--num_train_tasks', type=int, default=9, help = "# of training tasks")
            parser.add_argument('--num_test_tasks', type=int, default=3, help = "# of testing tasks")
            parser.add_argument('--n_way', type=int, default=2, help = "n_way of dataset")
            parser.add_argument('--m_support', type=int, default=5, help = "size of the support dataset")
            parser.add_argument('--k_query', type = int, default=128, help = "size of querry datasets")
            parser.add_argument('--meta_lr', type=float, default=0.001) 
            parser.add_argument('--update_lr', type=float, default=0.4) #0.4
            parser.add_argument('--update_step', type=int, default=3) #5
            parser.add_argument('--update_step_test', type=int, default=5) #10
            parser.add_argument('--add_similarity', type=bool, default=False)
            parser.add_argument('--add_selfsupervise', type=bool, default=False)
            parser.add_argument('--interact', type=bool, default=False)
            parser.add_argument('--add_weight', type=float, default=0.1)

            # Parse known arguments to avoid errors
            args,unknown= parser.parse_known_args()

            # Override arguments based on grid search combination
            args.dataset = dataset
            args.tar_test_dataset = tar_test_dataset
            args.input_model_file = input_model_file = "model_gin/supervised_contextpred.pth"  # Set appropriately if needed
            args.gnn_type = gnn_type = "gin"  # Modify as per your requirement
            args.add_similarity = add_similarity = True  # Set based on your grid
            args.add_selfsupervise = add_selfsupervise = True  # Set based on your grid
            args.add_masking = add_masking = True  # Assuming this is a new argument; handle accordingly
            args.add_weight = add_weight = 0.1  # Set based on your grid
            args.m_support = m_support  # Set based on your grid

            # Set different seeds for each run to ensure different initializations
            seed = 42 + repeat  # Example: 43, 44, 45
            args.runseed = seed
            set_seeds(seed)
            logger.info(f"Set random seed to {seed}")

            # Run the experiment
            start_time = time()
            try:
                val_aucs, test_aucs = run_experiment(args, experiment_dir, logger)
                # logger.info(f"Run {repeat} completed. Best Val AUC: {best_val_auc:.4f}, Final Test AUC: {final_test_auc:.4f}")
                # best_val_aucs.append(best_val_auc)
                # final_test_aucs.append(final_test_auc)
                best_val_aucs.append(val_aucs)
                final_test_aucs.append(test_aucs)
            except Exception as e:
                logger.error(f"Error during run {repeat}: {e}")
                continue  # Skip to the next run

            end_time = time()
            elapsed_time = round((end_time - start_time) / 60, 3)
            logger.info(f"Run {repeat} Time cost (min): {elapsed_time}")

        # After all repetitions, aggregate the results
        if best_val_aucs and final_test_aucs:
            mean_val_auc, std_val_auc = aggregate_results(best_val_aucs)
            mean_test_auc, std_test_auc = aggregate_results(final_test_aucs)

            # Format as "mean±std"
            val_auc_str = f"{mean_val_auc:.4f}±{std_val_auc:.4f}"
            test_auc_str = f"{mean_test_auc:.4f}±{std_test_auc:.4f}"

            # Save aggregated results
            task_name = f"{dataset}_train{m_support}_tar_test{tar_test_dataset}"
            summary_results[task_name] = {
                'best_val_aucs': best_val_aucs,
                'final_test_aucs': final_test_aucs,
                'best_val_auc_mean_std': val_auc_str,
                'final_test_auc_mean_std': test_auc_str
            }

            # Log aggregated results
            logger.info(f"Aggregated AUCs for {task_name}:")
            logger.info(f"Validation AUC: {val_auc_str}")
            logger.info(f"Test AUC: {test_auc_str}")

            print(f"\nAggregated AUCs for {task_name}:")
            print(f"Validation AUC: {val_auc_str}")
            print(f"Test AUC: {test_auc_str}")
        else:
            logger.warning(f"No successful runs for {dataset}_train{m_support}_tar_test{tar_test_dataset}")

    # Save summary results to JSON
    if summary_results:
        summary_dir = "summary_results"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'summary.json')
        save_json(summary_results, summary_path)
        print(f"\n=== Summary of Results saved to {summary_path} ===")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
