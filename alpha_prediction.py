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
from evaluate import train_reg, test_reg

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--reg", type=float, default=1e-5, help='Regularization parameter')
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
    _, testset, clftrainset, _, stem, col_distort, batch_transform = get_datasets(
        args.dataset, augment_clf_train=True, augment_test=True)

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


    def create_dataset(clftrainloader, device, net, target=None):
        if target is None:
            target = device

        net.eval()

        with torch.no_grad():
            store = []
            for batch_idx, (inputs, _) in enumerate(clftrainloader):
                inputs = inputs.to(device)
                rn = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                inputs = batch_transform(inputs, rn)
                representation = net(inputs)
                representation = representation.to(target)
                store.append((representation, rn))

            Xi, y = zip(*store)
            Xi, y = torch.cat(Xi, dim=0), torch.cat(y, dim=0)

        return Xi, y
        

    X, y = create_dataset(clftrainloader, device, net)
    X_test, y_test = create_dataset(testloader, device, net)
    clf = train_reg(X, y, device, reg_weight=args.reg)
    loss = test_reg(X_test, y_test, clf)
    print('Loss for augmentation parameter prediction')
    print(loss)
    return loss



loss = get_loss(args.load_from)

