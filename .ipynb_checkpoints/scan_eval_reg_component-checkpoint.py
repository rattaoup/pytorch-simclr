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
from evaluate import train_reg, test_reg,test_reg_component
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--baselines", type=str, default='ckpt', help='File series to load for baseline')
parser.add_argument("--ours", type=str, default='invgpn', help='File series to load for our method')
parser.add_argument("--reg", type=float, default=1e-8, help='Regularization parameter')
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


    def encode_train_set_spirograph(clftrainloader, device, net, col_distort, batch_transform):
        net.eval()

        store = []
        with torch.no_grad():
            t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),
                     bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, targets) in t:
                inputs, targets = inputs.to(device), targets.to(device)
                shape = (inputs.shape[0] * 100, *inputs.shape[1:])
                rn1 = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                inputs = batch_transform(inputs, rn1)
                representation = net(inputs)
                store.append((representation, targets))

                t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

        X, y = zip(*store)
        X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
        return X, y
        

    X, y = encode_train_set_spirograph(clftrainloader, device, net, col_distort, batch_transform)
    X_test, y_test = encode_train_set_spirograph(testloader, device, net, col_distort, batch_transform)
    clf = train_reg(X, y, device, reg_weight=args.reg)
    loss = test_reg_component(X_test, y_test, clf)
    return loss



baselines = args.baselines.split(",")
ours = args.ours.split(",")
results = defaultdict(list)
for stem in baselines+ours:
    for epoch in range(49, 50, 10):
        print(epoch)
        fname = stem + '_epoch{:03d}.pth'.format(epoch)
        loss = get_loss(fname)
        results[stem].append(loss)

print(results)
