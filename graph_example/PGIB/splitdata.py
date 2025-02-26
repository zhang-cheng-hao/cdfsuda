import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import os

# 设定数据集名称和保存路径
dataset_name = 'PROTEINS'  # 使用 MUTAG 数据集
root_path = '/root/PGIB/datasets' 
dataset_path = os.path.join(root_path, dataset_name)

# 创建保存子集的目录
train_path = os.path.join(root_path, f"{dataset_name}_training")
eval_path = os.path.join(root_path, f"{dataset_name}_evaluation")
test_path = os.path.join(root_path, f"{dataset_name}_testing")

for path in [train_path, eval_path, test_path]:
    os.makedirs(path, exist_ok=True)

# 加载 TUDataset
dataset = TUDataset(root=root_path, name=dataset_name)

# 计算拆分比例
total_size = len(dataset)
train_ratio, eval_ratio, test_ratio = 90, 7, 3  # 按照 90:7:3 的比例
total_ratio = train_ratio + eval_ratio + test_ratio

train_size = int((train_ratio / total_ratio) * total_size)
eval_size = int((eval_ratio / total_ratio) * total_size)
test_size = total_size - train_size - eval_size  # 确保总和为 total_size

print(f"Total samples: {total_size}")
print(f"Training samples: {train_size} ({(train_size / total_size) * 100:.2f}%)")
print(f"Evaluation samples: {eval_size} ({(eval_size / total_size) * 100:.2f}%)")
print(f"Testing samples: {test_size} ({(test_size / total_size) * 100:.2f}%)")

# 使用 random_split 进行拆分
generator = torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size], generator=generator)

# 定义保存子集的函数
def save_subset(subset, subset_name, save_dir):
    """
    将子集保存为一个 .pt 文件。
    
    参数:
    - subset (torch.utils.data.Subset): 拆分后的子集
    - subset_name (str): 子集名称，如 'training'
    - save_dir (str): 保存目录路径
    """
    subset_path = os.path.join(save_dir, f"{subset_name}.pt")
    torch.save(subset, subset_path)
    print(f"Saved {subset_name} subset with {len(subset)} samples to {subset_path}")

# 保存训练集、验证集和测试集
save_subset(train_dataset, 'training', train_path)
save_subset(eval_dataset, 'evaluation', eval_path)
save_subset(test_dataset, 'testing', test_path)

def load_subset(save_dir, subset_name):
    """
    加载保存的子集。
    
    参数:
    - save_dir (str): 子集保存目录路径
    - subset_name (str): 子集名称，如 'training'
    
    返回:
    - subset (torch.utils.data.Subset): 加载的子集
    """
    subset_path = os.path.join(save_dir, f"{subset_name}.pt")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    subset = torch.load(subset_path)
    print(f"Loaded {subset_name} subset with {len(subset)} samples from {subset_path}")
    return subset

# 加载训练集、验证集和测试集
train_subset = load_subset(train_path, 'training')
eval_subset = load_subset(eval_path, 'evaluation')
test_subset = load_subset(test_path, 'testing')