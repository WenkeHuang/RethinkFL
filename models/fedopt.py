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
    parser = ArgumentParser(description='Federated learning via FedOpt.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedOpt(FederatedModel):
    NAME = 'fedopt'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedOpt, self).__init__(nets_list, args, transform)

        self.global_lr = args.global_lr  # 0.5 0.25 0.1

        self.global_optimizer=None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        self.global_optimizer = torch.optim.SGD(
            self.global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def update_global(self):
        mean_state_dict = {}

        for name, param in self.global_net.state_dict().items():
            vs = []
            for client in self.nets_list:
                vs.append(client.state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.global_net)
        new_model.load_state_dict(mean_state_dict, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                    self.global_net.parameters(), new_model.parameters()
            ):
                param.grad = param.data - new_param.data

        # replace some non-parameters's state dict
        state_dict = self.global_net.state_dict()
        for name in dict(self.global_net.named_parameters()).keys():
            mean_state_dict[name] = state_dict[name]
        self.global_net.load_state_dict(mean_state_dict, strict=True)

        # optimization
        self.global_optimizer = torch.optim.SGD(
            self.global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

        for _, net in enumerate(self.nets_list):
            net.load_state_dict(self.global_net.state_dict())

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        # self.aggregate_nets(None)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.update_global()
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
