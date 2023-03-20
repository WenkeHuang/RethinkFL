import os
import sys
import socket
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle

import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=1, help='The Device Id for Experiment')

    parser.add_argument('--communication_epoch',type=int,default=100,help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch',type=int,default=10,help='The Local Epoch for each Participant')
    parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants')

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--rand_dataset', type=bool, default=True, help='The random seed.')

    parser.add_argument('--model', type=str, default='fedopt',  # moon fedinfonce
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='fl_digits',  #fl_officecaltech fl_digits
                        choices=DATASET_NAMES, help='Which scenario to perform experiments on.')

    parser.add_argument('--pri_aug', type=str, default='weak', # weak strong
                        help='Augmentation for Private Data')
    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')
    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--averaing', type=str, default='weight', help='The Option for averaging strategy')

    parser.add_argument('--infoNCET',type=float,default=0.02,help='The InfoNCE temperature')
    parser.add_argument('--T',type=float,default=0.05,help='The Knowledge distillation temperature')
    parser.add_argument('--weight',type=int,default=1,help='The Wegith for the distillation loss')

    parser.add_argument('--reserv_ratio',type=float,default=0.1,help='Reserve ratio for prototypes')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)

    backbones_list = priv_dataset.get_backbone(args.parti_num, None)

    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name

    print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
