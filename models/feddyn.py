import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel


# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedDyn(FederatedModel):
    NAME = 'feddyn'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDyn, self).__init__(nets_list, args, transform)
        self.client_grads = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for i in range(len(self.nets_list)):
            self.client_grads[i] = self.build_grad_dict(self.global_net)

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params)
        return grad_dict

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        local_grad = copy.deepcopy(self.client_grads[index])

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)

                reg_loss = 0.0
                cnt = 0.0
                for name, param in self.global_net.named_parameters():
                    term1 = (param * (
                            local_grad[name] - self.global_net.state_dict()[name]
                    )).sum()
                    term2 = (param * param).sum()

                    reg_loss += self.args.reg_lamb * (term1 + term2)
                    cnt += 1.0

                loss_ce = criterion(outputs, labels)
                loss=loss_ce+reg_loss/cnt
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        for name, param in net.named_parameters():
            local_grad[name] += (
                    net.state_dict()[name] - self.global_net.state_dict()[name]
            )
        self.client_grads[index] = local_grad
