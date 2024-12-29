import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedProc.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos



class FedProc(FederatedModel):
    NAME = 'fedproc'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProc, self).__init__(nets_list, args, transform)
        self.global_protos = []
        self.local_protos = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self,local_protos_list):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label


    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_protos=self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        Epoch = self.args.communication_epoch-1
        alpha = 1 - self.epoch_index/Epoch

        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)
                lossCE = criterion(outputs, labels)

                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None
                    for label in labels:
                        if label.item() in self.global_protos.keys():

                            f_pos = np.array(all_f)[all_global_protos_keys==label.item()][0].to(self.device)

                            f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(
                                self.device)

                            f_now = f[i].unsqueeze(0)

                            embedding_len = f_pos.shape
                            f_neg = f_neg.unsqueeze(1).view(-1, embedding_len[0])
                            f_pos = f_pos.view(-1, embedding_len[0])
                            f_proto = torch.cat((f_pos, f_neg), dim=0)
                            l = torch.cosine_similarity(f_now, f_proto, dim=1)
                            l = l

                            exp_l = torch.exp(l)
                            exp_l = exp_l.view(1, -1)
                            # l = torch.einsum('nc,ck->nk', [f_now, f_proto.T])
                            # l = l /self.T
                            # exp_l = torch.exp(l)
                            # exp_l = l
                            pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
                            pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
                            pos_mask = pos_mask.view(1, -1)
                            # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
                            pos_l = exp_l * pos_mask
                            sum_pos_l = pos_l.sum(1)
                            sum_exp_l = exp_l.sum(1)
                            loss_instance = -torch.log(sum_pos_l / sum_exp_l)
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE +=loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = alpha * loss_InfoNCE + (1-alpha) * lossCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.local_epoch-1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i,:])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i,:]]

        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
