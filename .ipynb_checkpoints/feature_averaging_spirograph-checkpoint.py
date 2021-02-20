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
from evaluate import train_reg, test_reg, test_reg_component
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--reg", type=float, default=1e-8, help='Regularization parameter')
parser.add_argument("--max-num-passes", type=int, default=20, help='Max number of passes to average')
parser.add_argument("--min-num-passes", type=int, default=10, help='Min number of passes to average')
parser.add_argument("--step-num-passes", type=int, default=2, help='Number of distinct M values to try')
parser.add_argument("--load-from", type=str, default='eiei', help='File series to load')
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


    def encode_train_set_spirograph_multi(clftrainloader, device, net, col_distort, batch_transform,  num_passes, target=None,):
        if target is None:
            target = device
        net.eval()
        
        X, y = [], None
  
        with torch.no_grad():
            for i in tqdm(range(num_passes)):
                store = []
                for batch_idx, (inputs, targets) in enumerate(clftrainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    shape = (inputs.shape[0] * 100, *inputs.shape[1:])
                    rn1 = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                    inputs = batch_transform(inputs, rn1)
                    representation = net(inputs)
                    store.append((representation, targets))

                    
                Xi, y = zip(*store)
                Xi, y = torch.cat(Xi, dim=0), torch.cat(y, dim=0)
                X.append(Xi)
                
        X = torch.stack(X, dim=0)
        return X, y
        

    X, y = encode_train_set_spirograph_multi(clftrainloader, device, net, col_distort, batch_transform, num_passes=args.max_num_passes,target='cpu')
    X_test, y_test = encode_train_set_spirograph_multi(testloader, device, net, col_distort, batch_transform, num_passes=args.max_num_passes,target='cpu')
    
    m_list = []
    loss_list = []
    for m in torch.linspace(args.min_num_passes, args.max_num_passes, args.step_num_passes):
        m = int(m)
        print("FA with M =", m)
        best_loss = float('inf')
        X_this = X[:m, ...].mean(0)
        X_test_this = X_test[:m, ...].mean(0)
        clf = train_reg(X_this, y, device, reg_weight=args.reg)
        loss = test_reg(X_test_this, y_test, clf)
        loss = round(loss, 5)
        m_list.append(m)
        loss_list.append(loss)

    return m_list, loss_list








m_list, loss_list = get_loss(args.load_from)
print('Number of passes')
print(m_list)
print('Loss')
print(loss_list)

