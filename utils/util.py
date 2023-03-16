import os
import numpy as np

import torch


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def save_networks(model, communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'para')
    create_if_not_exists(model_para_path)
    for net_idx, network in enumerate(nets_list):
        each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
        torch.save(network.state_dict(), each_network_path)


def save_protos(model, communication_idx):
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'protos')
    create_if_not_exists(model_para_path)

    for i in range(len(model.global_protos_all)):
        label = i
        protos = torch.cat(model.global_protos_all[i], dim=0).cpu().numpy()
        save_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(label) + '.npy')
        np.save(save_path, protos)
