import os
import json
import logging
import random
import numpy as np
import torch
from time import time
from itertools import product

from parser import get_args
from chem_lib.models import ContextAwareRelationNet, Meta_Trainer
from chem_lib.utils import count_model_params
from sklearn.metrics import roc_auc_score
from parser import get_parser
from chem_lib.models.cdtc_coordinator import MetaTaskSelector
from chem_lib.models.cdtc_trainer import CDTC_Trainer


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

def main():
    # Define Grid Search Configuration
    grid_search_config = {   
        'dataset': ['tox21', 'BZR', 'COX2', 'MUTAG', 'PROTEINS' ],  
        'n_shot_train': [1, 2, 3, 5],  
    }  
    
    # Generate Valid Combinations
    valid_combinations = [  
        {  
            'dataset': dataset,  
            'test_dataset': dataset,  
            'tar_test_dataset': tar_test_dataset,  
            'n_shot_train': n_shot,  
            'n_shot_test': n_shot  
        }  
        for dataset in grid_search_config['dataset']  
        for tar_test_dataset in grid_search_config['dataset']   
        if tar_test_dataset != dataset  
        for n_shot in grid_search_config['n_shot_train']  
    ]

    # Number of repetitions per experiment
    num_repeats = 3

    # Directory to save all experiments
    experiments_root = "experiments-cdtc"
    os.makedirs(experiments_root, exist_ok=True)

    # Summary of all results
    summary_results = {}

    for combo in valid_combinations:
        dataset = combo['dataset']
        test_dataset = combo['test_dataset']
        tar_test_dataset = combo['tar_test_dataset']
        n_shot_train = combo['n_shot_train']
        n_shot_test = combo['n_shot_test']

        # Initialize lists to store AUCs for each repetition
        best_val_aucs = []
        final_test_aucs = []
        # Initialize arguments
        parser = get_parser(root_dir='.')
        args = parser.parse_args()

        # Iterate over repetitions
        for repeat in range(1, num_repeats + 1):
            # Define experiment name and directory
            experiment_name = f"{dataset}_train{n_shot_train}_test{test_dataset}_tar_test{tar_test_dataset}_run{repeat}"
            experiment_dir = os.path.join(experiments_root, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Set up logging
            logger = setup_logging(experiment_dir)

            logger.info(f"\n=== Experiment: {experiment_name} ===")


            # Override arguments based on the combination
            args.dataset = dataset
            args.test_dataset = test_dataset
            args.tar_test_dataset = tar_test_dataset
            args.n_shot_train = n_shot_train
            args.n_shot_test = n_shot_test

            args = get_args(root_dir='.', args=args)  # Adjust root_dir as needed



            # Set different seeds for each run
            seed = 42 + repeat  # Example: 43, 44, 45
            set_seeds(seed)
            logger.info(f"Set random seed to {seed}")
            # PROTEINS 3
            # MUTAG 7
            # BZR 53
            # COX2 35
            # tox21 2
            if args.dataset == 'PROTEINS':
                args.input_emb_dim = 3
            elif args.dataset == 'MUTAG':
                args.input_emb_dim = 7
            elif args.dataset == 'BZR':
                args.input_emb_dim = 53
            elif args.dataset == 'COX2':
                args.input_emb_dim = 35
            elif args.dataset == 'tox21':
                    args.input_emb_dim = 2
            # Initialize and count model parameters
            meta_task_selector = MetaTaskSelector(
                node_attr_dim= args.input_emb_dim,
                hidden_dim=args.cdtc_hidden_dim,
                embedding_dim=args.cdtc_embedding_dim,
                proto_dim=args.cdtc_proto_dim,
                refined_dim=args.cdtc_refined_dim,
                source_num_meta_tasks = len(args.train_tasks),
                target_num_meta_tasks = len(args.test_tasks),
                num_classes=2,
                delta=args.delta,
                num_tasks_to_select=args.num_tasks_to_select
            )
            model = ContextAwareRelationNet(args) 
            count_model_params(model)
            model = model.to(args.device)

            # Initialize trainer
            trainer = CDTC_Trainer(args, model, meta_task_selector)

            # Training loop
            t1 = time()
            logger.info('Initial Evaluation')
            best_avg_auc = 0

            for epoch in range(1, args.epochs + 1):
                logger.info(f"----------------- Epoch: {epoch} -----------------")
                trainer.train_step()

                if epoch % args.eval_steps == 0 or epoch == 1 or epoch == args.epochs:
                    logger.info(f"Evaluation on epoch {epoch}")
                    current_val_auc = trainer.test_step()
                    if current_val_auc > best_avg_auc:
                        best_avg_auc = current_val_auc

                if epoch % args.save_steps == 0:
                    trainer.save_model()
                    logger.info(f"Model saved at epoch {epoch}")

                time_cost = round((time() - t1) / 60, 3)
                logger.info(f"Time cost (min): {time_cost}")
                t1 = time()

            # Final Test
            final_test_auc = trainer.tar_test_step()
            logger.info('Train done.')
            logger.info(f'Best Avg AUC: {best_avg_auc}')
            logger.info(f'Final AUC: {final_test_auc}')

            trainer.conclude()

            if args.save_logs:
                trainer.save_result_log()
                logger.info("Saved result logs.")

            # Append AUCs to lists
            best_val_aucs.append(best_avg_auc)
            final_test_aucs.append(final_test_auc)

            # Save individual run results
            run_results = {
                'best_val_auc': best_avg_auc,
                'final_test_auc': final_test_auc
            }
            run_results_path = os.path.join(experiment_dir, 'run_results.json')
            save_json(run_results, run_results_path)
            logger.info(f"Saved run results to {run_results_path}")

        # After all repetitions, aggregate the results
        mean_val_auc, std_val_auc = aggregate_results(best_val_aucs)
        mean_test_auc, std_test_auc = aggregate_results(final_test_aucs)

        # Format as "mean±std"
        val_auc_str = f"{mean_val_auc:.4f}±{std_val_auc:.4f}"
        test_auc_str = f"{mean_test_auc:.4f}±{std_test_auc:.4f}"

        # Save aggregated results
        task_name = f"{dataset}_train{n_shot_train}_test{test_dataset}_tar_test{tar_test_dataset}"
        summary_results[task_name] = {
            'best_val_aucs': best_val_aucs,
            'final_test_aucs': final_test_aucs,
            'best_val_auc_mean_std': val_auc_str,
            'final_test_auc_mean_std': test_auc_str
        }

        # Log aggregated results
        aggregated_log = f"Aggregated AUCs for {task_name}:\nValidation AUC: {val_auc_str}\nTest AUC: {test_auc_str}"
        print(aggregated_log)

    # Save summary results to JSON
    summary_dir = "summary_results"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'summary.json')
    save_json(summary_results, summary_path)
    print(f"\n=== Summary of Results saved to {summary_path} ===")

    # Optionally, print a detailed summary
    print("\n=== Detailed Summary of Results ===")
    for task, metrics in summary_results.items():
        print(f"\nTask: {task}")
        print(f"  Best Val AUCs: {metrics['best_val_aucs']}")
        print(f"  Final Test AUCs: {metrics['final_test_aucs']}")
        print(f"  Best Val AUC: {metrics['best_val_auc_mean_std']}")
        print(f"  Final Test AUC: {metrics['final_test_auc_mean_std']}")

if __name__ == "__main__":
    main()
