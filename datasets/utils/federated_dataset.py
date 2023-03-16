from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class FederatedDataset:
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list) -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def partition_label_skew_loaders(train_dataset: datasets, test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num
    n_class_sample = setting.N_SAMPLES_PER_Class
    min_size = 0
    min_require_size = 10
    y_train = train_dataset.targets
    N = len(y_train)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_participants)]
        for k in range(n_class):
            idx_k = [i for i, j in enumerate(y_train) if j == k]
            np.random.shuffle(idx_k)
            if n_class_sample != None:
                idx_k = idx_k[0:n_class_sample * n_participants]
            beta = setting.args.beta
            if beta == 0:
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_participants))]
            else:
                proportions = np.random.dirichlet(np.repeat(a=beta, repeats=n_participants))
                proportions = np.array([p * (len(idx_j) < N / n_participants) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_participants):
        np.random.shuffle(idx_batch[j])
        if n_class_sample != None:
            idx_batch[j] = idx_batch[j][0:n_class_sample * n_class]
        net_dataidx_map[j] = idx_batch[j]
    net_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    for j in range(n_participants):
        train_sampler = SubsetRandomSampler(net_dataidx_map[j])
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=4, drop_last=True)
        setting.train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.local_batch_size, shuffle=False, num_workers=4)
    setting.test_loader = test_loader

    return setting.train_loaders, setting.test_loader, net_cls_counts


def partition_digits_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         setting: FederatedDataset) -> Tuple[list, list]:
    ini_len_dict = {}
    not_used_index_dict = {}
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        if name not in not_used_index_dict:
            if name == 'svhn':
                train_dataset = train_datasets[i].dataset
                y_train = train_dataset.labels
            elif name == 'syn':
                train_dataset = train_datasets[i].imagefolder_obj
                y_train = train_dataset.targets
            else:
                train_dataset = train_datasets[i].dataset
                y_train = train_dataset.targets

            not_used_index_dict[name] = np.arange(len(y_train))
            ini_len_dict[name] = len(y_train)

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name

        if name == 'syn':
            train_dataset = train_datasets[index].imagefolder_obj
        else:
            train_dataset = train_datasets[index].dataset

        idxs = np.random.permutation(not_used_index_dict[name])

        percent = setting.percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]

        not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]):]

        train_sampler = SubsetRandomSampler(selected_idx)
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler)
        setting.train_loaders.append(train_loader)

    for index in range(len(test_datasets)):
        name = test_datasets[index].data_name
        if name == 'syn':
            test_dataset = test_datasets[index].imagefolder_obj
        else:
            test_dataset = test_datasets[index].dataset

        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.local_batch_size, shuffle=False)
        setting.test_loader.append(test_loader)

    return setting.train_loaders, setting.test_loader


def partition_office_domain_skew_loaders_new(train_datasets: list, test_datasets: list,
                                             setting: FederatedDataset) -> Tuple[list, list]:
    ini_len_dict = {}
    not_used_index_dict = {}
    all_labels_list = []
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        all_train_index = np.array(train_datasets[i].train_index_list)
        if name not in not_used_index_dict:
            not_used_index_dict[name] = np.arange(len(all_train_index))
            ini_len_dict[name] = len(all_train_index)

        all_labels_list.append(np.unique(np.array(train_datasets[i].imagefolder_obj.targets)[all_train_index]))

    all_labels_array = np.array(all_labels_list)

    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index]
        test_loader = DataLoader(test_dataset, batch_size=setting.args.local_batch_size)
        setting.test_loader.append(test_loader)

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name

        train_dataset = train_datasets[index]

        idxs = np.random.permutation(not_used_index_dict[name])
        percent = setting.percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]

        all_train_index = np.array(train_datasets[index].train_index_list)
        train_labels = np.array(train_datasets[index].imagefolder_obj.targets)[all_train_index]
        selected_labels = train_labels[selected_idx]

        show_up_num = np.zeros(len(all_labels_array[index]))
        for i in range(len(selected_labels)):
            label = selected_labels[i]
            show_up_num[label] += 1

        not_used_labels = np.where(show_up_num == 0)[0]
        for i in range(len(not_used_labels)):
            not_used_label = not_used_labels[i]
            not_used_label_idx = np.where(train_labels == not_used_label)[0]
            add_index = not_used_label_idx[np.random.randint(len(not_used_label_idx))]

            used_label = train_labels[selected_idx]
            prob_del_place = np.where(show_up_num >= 2)[0]
            del_index = np.random.randint(len(prob_del_place))
            del_label = prob_del_place[del_index]

            prob_del_selected = np.where(used_label == del_label)[0]
            del_index_selected = prob_del_selected[np.random.randint(len(prob_del_selected))]
            selected_idx = selected_idx[selected_idx != selected_idx[del_index_selected]]
            selected_idx = np.append(selected_idx, add_index)

            show_up_num[del_label] -= 1
            show_up_num[not_used_label] += 1

        not_select_index = np.array(idxs)
        for i in range(len(selected_idx)):
            not_select_index = not_select_index[not_select_index != selected_idx[i]]

        not_used_index_dict[name] = not_select_index

        train_sampler = SubsetRandomSampler(selected_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler)
        setting.train_loaders.append(train_loader)

    return setting.train_loaders, setting.test_loader


def partition_office_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         setting: FederatedDataset) -> Tuple[list, list]:
    ini_len_dict = {}
    not_used_index_dict = {}
    # all_labels = []
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        if name not in not_used_index_dict:
            all_train_index = np.array(train_datasets[i].train_index_list)
            not_used_index_dict[name] = np.arange(len(all_train_index))
            ini_len_dict[name] = len(all_train_index)


    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index]
        test_loader = DataLoader(test_dataset, batch_size=setting.args.local_batch_size)
        setting.test_loader.append(test_loader)

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name
        train_dataset = train_datasets[index]

        idxs = np.random.permutation(not_used_index_dict[name])

        # use_labels = np.unique(train_dataset.labels[selected_idx])

        percent = setting.percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]
        not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]):]

        train_sampler = SubsetRandomSampler(selected_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler)
        setting.train_loaders.append(train_loader)

    return setting.train_loaders, setting.test_loader


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts
