'''Train CIFAR10/100 with PyTorch using standard Contrastive Learning. This script tunes evaluates the invariance of
the learned representations.'''
import torch
import torch.backends.cudnn as cudnn

import math
import os
import argparse
from collections import defaultdict

from augmentation import ColourDistortion, TensorNormalise, ModuleCompose
from models import *
from configs import get_datasets, get_mean_std
from evaluate import train_reg, test_reg, test_reg_component, encode_train_set_spirograph
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--reg", type=float, default=1e-8, help='Regularization parameter')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
args = parser.parse_args()



def get_loss(fname):
    # Load checkpoint.
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', fname)
    checkpoint = torch.load(resume_from)
    args.dataset = checkpoint['args']['dataset']
    args.arch = checkpoint['args']['arch']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    trainset, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_datasets(args.dataset, train_proportion=1.0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

    # Model
    ##############################################################
    # Encoder
    ##############################################################
    if args.arch == 'resnet18':
        net = ResNet18(stem=stem)
    elif args.arch == 'resnet34':
        net = ResNet34(stem=stem)
    elif args.arch == 'resnet50':
        net = ResNet50(stem=stem)
    else:
        raise ValueError("Bad architecture specification")
    net = net.to(device)

    if device == 'cuda':
        repr_dim = net.representation_dim
        net = torch.nn.DataParallel(net)
        net.representation_dim = repr_dim
        cudnn.benchmark = True

    net.load_state_dict(checkpoint['net'])

    batch_transform = batch_transform.to(device)

    X, y = encode_train_set_spirograph(clftrainloader, device, net, col_distort, batch_transform)
    X_test, y_test = encode_train_set_spirograph(testloader, device, net, col_distort, batch_transform)
    clf = train_reg(X, y, device, reg_weight=args.reg)
    loss = test_reg_component(X_test, y_test, clf)
    print('Loss for component [m, b, sigma, foreground_R]')
    print(loss)
    return loss



loss = get_loss(args.load_from)
