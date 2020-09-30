'''This script tunes the L2 reg weight of the final classifier.'''
import argparse
import os
import math

import torch
import torch.backends.cudnn as cudnn

from configs import get_datasets
from evaluate import encode_train_set, train_clf, test
from models import *

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--reg", type=float, default=1e-5, help='L2 reg')
parser.add_argument("--num-steps", type=int, default=8, help='Number of spaced reg parameters to try')
parser.add_argument("--proportion-upper", type=float, default=1., help='Proportion of train data to use upper')
parser.add_argument("--proportion-lower", type=float, default=.05, help='Proportion of train data to use lower')
args = parser.parse_args()

# Load checkpoint.
print('==> Loading settings from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
resume_from = os.path.join('./checkpoint', args.load_from)
checkpoint = torch.load(resume_from)
args.dataset = checkpoint['args']['dataset']
args.arch = checkpoint['args']['arch']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

results = []
for proportion in torch.linspace(args.proportion_lower, args.proportion_upper, args.num_steps):
    # Data
    print('==> Preparing data..')
    _, testset, clftrainset, num_classes, stem, _, _ = get_datasets(args.dataset, train_proportion=proportion)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True)

    # Model
    print('==> Building model..')
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

    print('==> Loading encoder from checkpoint..')
    net.load_state_dict(checkpoint['net'])


    X, y = encode_train_set(clftrainloader, device, net)
    clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=args.reg)
    acc, loss = test(testloader, device, net, clf)
    results.append((acc, loss))
print(results)
