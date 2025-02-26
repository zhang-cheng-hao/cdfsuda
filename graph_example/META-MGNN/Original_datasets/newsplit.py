import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import os
import json

# 设定数据集名称和保存路径
dataset_name = 'PROTEINS'  # 使用 BZR 数据集
root_path = '/root/Meta-MGNN/Original_datasets'
dataset_path = os.path.join(root_path, dataset_name)

# 定义子集比例或大小
split_ratios = [0.25, 0.25, 0.25, 0.25]  # 用户可以根据需要调整
# 或者直接定义每个子集的大小，例如：[1000, 600, 300, 100]

# 检查 split_ratios 是否为比例，且总和为 1
if isinstance(split_ratios, list) and all(0 < r < 1 for r in split_ratios):
    total_ratio = sum(split_ratios)
    split_ratios = [r / total_ratio for r in split_ratios]  # 归一化
    split_mode = 'ratio'
else:
    split_mode = 'size'

# 创建保存子集的目录
subset_dirs = []
for i in range(len(split_ratios)):
    subset_name = f"new/{i + 1}"
    subset_path = os.path.join(root_path, f"{dataset_name}/{subset_name}/raw")
    os.makedirs(subset_path, exist_ok=True)
    subset_path2 = os.path.join(root_path, f"{dataset_name}/{subset_name}/processed")
    os.makedirs(subset_path2, exist_ok=True)
    subset_dirs.append(subset_path)

# 加载 TUDataset
dataset = TUDataset(root=root_path, name=dataset_name)

# 计算拆分大小
total_size = len(dataset)
if split_mode == 'ratio':
    split_sizes = [int(r * total_size) for r in split_ratios]
    # 调整最后一个子集的大小以确保总和等于 total_size
    split_sizes[-1] = total_size - sum(split_sizes[:-1])
else:
    split_sizes = split_ratios  # 直接使用定义的大小

print(f"Total samples: {total_size}")
for i, size in enumerate(split_sizes):
    percentage = (size / total_size) * 100
    print(f"Subset {i + 1} samples: {size} ({percentage:.2f}%)")

# 使用 random_split 进行拆分
generator = torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
subsets = random_split(dataset, split_sizes, generator=generator)


# 定义保存子集的函数
def save_subset(subset, subset_name, save_dir):
    """
    将子集保存为一个 .pt 文件。

    参数:
    - subset (torch.utils.data.Subset): 拆分后的子集
    - subset_name (str): 子集名称，如 'subset_1'
    - save_dir (str): 保存目录路径
    """
    subset_path = os.path.join(save_dir, f"data.pt")
    torch.save(subset, subset_path)
    print(f"Saved {subset_name} subset with {len(subset)} samples to {subset_path}")


# 保存所有子集
for i, subset in enumerate(subsets):
    subset_name = f"subset_{i + 1}"
    save_subset(subset, subset_name, subset_dirs[i])


# 定义加载子集的函数
def load_subset(save_dir, subset_name):
    """
    加载保存的子集。

    参数:
    - save_dir (str): 子集保存目录路径
    - subset_name (str): 子集名称，如 'subset_1'

    返回:
    - subset (torch.utils.data.Subset): 加载的子集
    """
    subset_path = os.path.join(save_dir, f"data.pt")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    subset = torch.load(subset_path)
    print(f"Loaded {subset_name} subset with {len(subset)} samples from {subset_path}")
    return subset


# 加载所有子集（示例）
loaded_subsets = []
for i in range(len(split_ratios)):
    subset_name = f"subset_{i + 1}"
    loaded_subset = load_subset(subset_dirs[i], subset_name)
    loaded_subsets.append(loaded_subset)

# 统计每个子集的正负样本分布
cnt_tasks = []
for i, subset in enumerate(loaded_subsets):
    positive = 0
    negative = 0
    for data in subset:
        label = data.y.item()
        # 假设正类标签为1，负类标签为0
        if label == 1:
            positive += 1
        else:
            negative += 1
    cnt_tasks.append([negative, positive])
    print(f"Subset {i + 1} - Positive: {positive}, Negative: {negative}")

print("各子集的正负样本分布:", cnt_tasks)
