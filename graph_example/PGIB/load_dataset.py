import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
import pdb
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader, download_url, extract_zip
from torch_geometric.io import read_tu_data
from typing import Callable, List, Optional
import shutil

def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


def read_syn_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

    x = torch.from_numpy(features).float()
    y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    y = torch.from_numpy(np.where(y)[1])
    edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    return data


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list


# def get_dataset(dataset_dir, dataset_name, task=None):
#     sync_dataset_dict = {
#         'BA_2Motifs'.lower(): 'BA_2Motifs',
#         'BA_Shapes'.lower(): 'BA_shapes',
#         'BA_Community'.lower(): 'BA_Community',
#         'Tree_Cycle'.lower(): 'Tree_Cycle',
#         'Tree_Grids'.lower(): 'Tree_Grids',
#     }
#     sentigraph_names = ['Graph-SST2', 'Graph-Twitter', 'Graph-SST5']
#     # sentigraph_names = ['Graph_SST2', 'Graph_Twitter', 'Graph_SST5']
#     sentigraph_names = [name.lower() for name in sentigraph_names]
#     molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

#     if dataset_name.lower() == 'MUTAG'.lower():
#         return load_MUTAG(dataset_dir, 'MUTAG')
#     elif dataset_name.lower() in sync_dataset_dict.keys():
#         sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
#         return load_syn_data(dataset_dir, sync_dataset_filename)
#     elif dataset_name.lower() in molecule_net_dataset_names:
#         return load_MolecueNet(dataset_dir, dataset_name, task)
#     elif dataset_name.lower() in sentigraph_names:
#         return load_SeniGraph(dataset_dir, dataset_name)
#     else:
#         return load_TUDataset(f"{dataset_dir}/{dataset_name}", dataset_name) 
#         # f"./checkpoint/{data_args.dataset_name}/" 
#         # dataset_dir =  ./datasets
#         # f"{dataset_dir}/{dataset_name}"
        
#         # raise NotImplementedError
def adjust_feature_dim(data, target_dim=53):
    """
    调整 data.x 的特征维度为 target_dim。

    如果原始特征维度小于 target_dim，则在后面填充零。
    如果原始特征维度大于 target_dim，则截断到 target_dim。

    Args:
        data (torch_geometric.data.Data): 数据对象。
        target_dim (int): 目标特征维度。

    Returns:
        torch_geometric.data.Data: 修改后的数据对象。
    """
    current_dim = data.x.size(1)
    if current_dim < target_dim:
        # 填充零
        padding = torch.zeros((data.num_nodes, target_dim - current_dim), dtype=data.x.dtype)
        data.x = torch.cat([data.x, padding], dim=1)
    elif current_dim > target_dim:
        # 截断到 target_dim
        data.x = data.x[:, :target_dim]
    # 如果当前维度等于目标维度，则不做任何修改
    return data

def get_dataset(dataset_dir, dataset_name, task=None, domain='source'):
    """
    加载指定域（源域或目标域）的数据集。

    Args:
        dataset_dir (str): 数据集根目录。
        dataset_name (str): 数据集名称。
        domain (str): 域名，'source' 或 'target'。
        task (str, optional): 任务类型（如分类任务）。

    Returns:
        dataset: 加载后的数据集对象。若 domain='target'，则数据集中不包含标签。
    """
    sync_dataset_dict = {
        'BA_2Motifs'.lower(): 'BA_2Motifs',
        'BA_Shapes'.lower(): 'BA_shapes',
        'BA_Community'.lower(): 'BA_Community',
        'Tree_Cycle'.lower(): 'Tree_Cycle',
        'Tree_Grids'.lower(): 'Tree_Grids',
    }
    sentigraph_names = ['Graph-SST2', 'Graph-Twitter', 'Graph-SST5']
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if dataset_name.lower() == 'MUTAG'.lower():
        dataset = load_MUTAG(dataset_dir, 'MUTAG')
    elif dataset_name.lower() in sync_dataset_dict.keys():
        sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
        dataset = load_syn_data(dataset_dir, sync_dataset_filename)
    elif dataset_name.lower() in molecule_net_dataset_names:
        dataset = load_MolecueNet(dataset_dir, dataset_name, task)
    elif dataset_name.lower() in sentigraph_names:
        dataset = load_SeniGraph(dataset_dir, dataset_name)
    else:
        dataset = load_TUDataset(osp.join(dataset_dir, dataset_name), dataset_name)

    target_dim = 53
    # 调整每个数据对象的 .x 属性
    if isinstance(dataset, InMemoryDataset):
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'x'):
            dataset.data = adjust_feature_dim(dataset.data, target_dim)
        if hasattr(dataset, 'slices'):
            # 如果是 InMemoryDataset，可能需要调整 slices 中的 x
            for key in dataset.slices.keys():
                if key == 'x':
                    # 假设 dataset.data.x 已经被调整
                    continue
    elif isinstance(dataset, list):
        for data in dataset:
            data = adjust_feature_dim(data, target_dim)
    elif isinstance(dataset, Subset):
        for data in dataset:
            data = adjust_feature_dim(data, target_dim)
    else:
        # 处理其他可能的数据集类型
        for data in dataset:
            data = adjust_feature_dim(data, target_dim)
    # if domain == 'target':
    #     # 删除目标域数据集中的所有标签
    #     if isinstance(dataset, InMemoryDataset):
    #         # 对于 InMemoryDataset 类型的数据集
    #         if hasattr(dataset, 'data') and hasattr(dataset.data, 'y'):
    #             dataset.data.y = None
    #     elif isinstance(dataset, list):
    #         # 对于数据列表类型的数据集
    #         for data in dataset:
    #             data.y = None
    #     elif isinstance(dataset, Subset):
    #         # 对于 Subset 类型的数据集
    #         for data in dataset:
    #             data.y = None
    #     # 根据需要处理其他类型的数据集

    return dataset

class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.name = name
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0

        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i

        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0

        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['MUTAG_A', 'MUTAG_graph_labels', 'MUTAG_graph_indicator', 'MUTAG_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(', ')
            l = int(lr[0])
            r = int(lr[1])
            adj_all[l - 1, r - 1] = 1

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                y=label)
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = read_syn_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])

def load_TUDataset(dataset_dir, dataset_name):
    dataset = TUDataset(root=dataset_dir, name=dataset_name)
    return dataset

def load_MUTAG(dataset_dir, dataset_name):
    """ 188 molecules where label = 1 denotes mutagenic effect """
    dataset = MUTAGDataset(root=dataset_dir, name=dataset_name)
    return dataset


def load_syn_data(dataset_dir, dataset_name):
    """ The synthetic dataset """
    if dataset_name.lower() == 'BA_2Motifs'.lower():
        dataset = BA2MotifDataset(root=dataset_dir, name=dataset_name)
    else:
        dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


def load_MolecueNet(dataset_dir, dataset_name, task=None):
    """ Attention the multi-task problems not solved yet """
    molecule_net_dataset_names = {name.lower(): name for name in MoleculeNet.names.keys()}
    dataset = MoleculeNet(root=dataset_dir, name=molecule_net_dataset_names[dataset_name.lower()])
    dataset.data.x = dataset.data.x.float()
    if task is None:
        dataset.data.y = dataset.data.y.squeeze().long()
    else:
        dataset.data.y = dataset.data.y[:, 0].long()
    dataset.node_type_dict = None
    dataset.node_color = None
    return dataset


def load_SeniGraph(dataset_dir, dataset_name):
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name)
    return dataset


def get_dataloader(dataset, dataset_name, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: boolepochs
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
        

    elif dataset_name.lower() in [name.lower() for name in MoleculeNet.names.keys()]:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train = torch.utils.data.Subset(dataset, range(num_train))
        eval = torch.utils.data.Subset(dataset, range(num_train, num_train + num_eval))
        test = torch.utils.data.Subset(dataset, range(num_train + num_eval, num_train + num_eval + num_test))

        dataloader = dict()
        dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
        dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
        dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
        

        return dataloader

    
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))
        # # Created using indices from 0 to train_size.


        # print(train)
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)

    return dataloader

# def get_dataloaders(source_dataset, target_dataset, batch_size, num_shots=5, data_split_ratio=None, seed=42):
#     """
#     创建源域和目标域的 DataLoader，支持少样本设置。
#     Args:
#         source_dataset: 源域数据集。
#         target_dataset: 目标域数据集。
#         batch_size: int，批量大小。
#         fewshot: bool，是否启用少样本设置。
#         num_shots: int，每个类别的少量样本数量。
#         data_split_ratio: list，训练、验证和测试的比例。
#         seed: int，随机种子。
#     Returns:
#         dict: 包含源域和目标域的 DataLoader 字典。
#     """
#     dataloaders = {}
#     generator = torch.Generator().manual_seed(seed)
#     # 源域数据集划分

#     # 获取每个类别的少量样本索引
#     fewshot_indices, fewshot_labels = get_fewshot_indices(source_dataset, num_shots=num_shots)

#     # 创建少量标注的源域数据集
#     source_labeled_subset = Subset(source_dataset, fewshot_indices)
#     dataloaders['source_labeled'] = DataLoader(source_labeled_subset, batch_size=batch_size, shuffle=True)
#     # 创建源域的无标签数据集（排除少量标注样本）
#     all_indices = set(range(len(source_dataset)))
#     unlabeled_indices = list(all_indices - set(fewshot_indices))
#     source_unlabeled_subset = Subset(source_dataset, unlabeled_indices)
#     dataloaders['source_unlabeled'] = DataLoader(source_unlabeled_subset, batch_size=batch_size, shuffle=True)

#     # 目标域数据集（全部无标签）
#     target_subset = target_dataset  # 假设目标域数据集已被标注为无标签
#     dataloaders['target_unlabeled'] = DataLoader(target_subset, batch_size=batch_size, shuffle=True)
#     return dataloaders

# def get_fewshot_indices(dataset, num_shots=5, num_classes=None):
#     """
#     获取每个类别的少量样本索引。
#     Args:
#         dataset: 数据集对象。
#         num_shots: 每个类别的样本数量。
#         num_classes: 类别数量，默认为数据集中的类别数量。
#     Returns:
#         list: 选择的索引列表。
#         list: 对应的标签列表。
#     """
#     if num_classes is None:
#         num_classes = len(set([data.y.item() for data in dataset]))
#     class_indices = {c: [] for c in range(num_classes)}
#     for idx, data in enumerate(dataset):
#         label = data.y.item()
#         if len(class_indices[label]) < num_shots:
#             class_indices[label].append(idx)
#         if all(len(indices) == num_shots for indices in class_indices.values()):
#             break
#     selected_indices = [idx for indices in class_indices.values() for idx in indices]
#     selected_labels = [dataset[idx].y.item() for idx in selected_indices]
#     return selected_indices, selected_labels

def get_fewshot_indices(dataset, num_shots=5, num_classes=None, seed=42):
    """
    获取每个类别的少量样本索引。

    Args:
        dataset: 数据集对象，必须具有 `.y` 属性表示标签。
        num_shots: 每个类别的样本数量。
        num_classes: 类别数量，默认为数据集中的类别数量。
        seed: 随机种子，确保可重复性。

    Returns:
        list: 选择的少量样本索引列表。
    """
    if num_classes is None:
        num_classes = len(set([data.y.item() for data in dataset]))

    class_indices = {c: [] for c in range(num_classes)}
    rng = np.random.default_rng(seed)

    for idx, data in enumerate(dataset):
        label = data.y.item()
        if len(class_indices[label]) < num_shots:
            class_indices[label].append(idx)
        if all(len(indices) == num_shots for indices in class_indices.values()):
            break

    selected_indices = [idx for indices in class_indices.values() for idx in indices]
    return selected_indices

def get_dataloaders(source_dataset, target_dataset, batch_size, num_shots=5, 
                   data_split_ratio={'source_val': 0.1, 'target_test': 0.2}, seed=42):
    """
    创建源域和目标域的 DataLoader，支持少样本设置。

    Args:
        source_dataset: 源域数据集对象，完全标注。
        target_dataset: 目标域数据集对象，完全标注。
        batch_size: int，批量大小。
        num_shots: int，每个类别的少量样本数量。
        data_split_ratio: dict，包含 'source_val' 和 'target_test' 的比例。
        seed: int，随机种子。

    Returns:
        dict: 包含训练、验证和测试的 DataLoader 字典。
            源域（Source Domain）：
                source_labeled_train：包含少量有标签的数据（每类 num_shots 个样本）。
                source_val：独立的验证集，有标签。
                source_unlabeled_train：包含剩余的无标签数据，用于训练。
            目标域（Target Domain）：
                target_unlabeled_train：全部无标签数据，用于训练。
                target_test：独立的测试集，有标签，用于评估模型性能。
    """
    dataloaders = {}
    generator = torch.Generator().manual_seed(seed)

    # 获取少量有标签数据的索引
    fewshot_indices = get_fewshot_indices(source_dataset, num_shots=num_shots, seed=seed)
    
    # 创建少量标注的源域数据集
    source_labeled_subset = Subset(source_dataset, fewshot_indices)
    dataloaders['source_labeled_train'] = DataLoader(
        source_labeled_subset, batch_size=batch_size, shuffle=True, generator=generator
    )

    # 创建源域的验证集（独立的有标签数据）
    # 剔除少量标注数据后的剩余数据
    all_source_indices = set(range(len(source_dataset)))
    remaining_source_indices = list(all_source_indices - set(fewshot_indices))
    
    # 计算验证集大小
    num_source_val = int(data_split_ratio.get('source_val', 0.1) * len(source_dataset))
    
    # 随机选择验证集索引
    rng = np.random.default_rng(seed)
    source_val_indices = rng.choice(remaining_source_indices, size=num_source_val, replace=False).tolist()
    
    source_val_subset = Subset(source_dataset, source_val_indices)
    dataloaders['source_val'] = DataLoader(
        source_val_subset, batch_size=batch_size, shuffle=False
    )
    
    # 源域的无标签数据集（排除少量标注和验证集）
    unlabeled_source_indices = list(set(remaining_source_indices) - set(source_val_indices))
    source_unlabeled_subset = Subset(source_dataset, unlabeled_source_indices)
    dataloaders['source_unlabeled_train'] = DataLoader(
        source_unlabeled_subset, batch_size=batch_size, shuffle=True, generator=generator
    )

    # 目标域数据集划分
    # 创建目标域的测试集
    num_target_test = int(data_split_ratio.get('target_test', 0.2) * len(target_dataset))
    target_indices = list(range(len(target_dataset)))
    rng = np.random.default_rng(seed)
    target_test_indices = rng.choice(target_indices, size=num_target_test, replace=False).tolist()
    
    target_test_subset = Subset(target_dataset, target_test_indices)
    dataloaders['target_test'] = DataLoader(
        target_test_subset, batch_size=batch_size, shuffle=False
    )
    
    # 创建目标域的训练集（全部无标签，剔除测试集）
    target_train_indices = list(set(target_indices) - set(target_test_indices))
    target_unlabeled_subset = Subset(target_dataset, target_train_indices)
    dataloaders['target_unlabeled_train'] = DataLoader(
        target_unlabeled_subset, batch_size=batch_size, shuffle=True, generator=generator
    )
    
    return dataloaders