'''This script tunes the L2 reg weight of the final classifier.'''
import argparse
import os
import math

import torch
import torch.backends.cudnn as cudnn

from configs import get_datasets
from evaluate import encode_train_set, train_clf, test, train_reg, test_reg
from models import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--reg-lower", type=float, default=-8, help='Minimum log regularization parameter (base 10)')
parser.add_argument("--reg-upper", type=float, default=-1, help='Maximum log regularization parameter (base 10)')
parser.add_argument("--num-steps", type=int, default=8, help='Number of log-linearly spaced reg parameters to try')
parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
args = parser.parse_args()

# Load checkpoint.
print('==> Loading settings from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
resume_from = os.path.join('./checkpoint', args.load_from)
checkpoint = torch.load(resume_from)
args.dataset = checkpoint['args']['dataset']
args.arch = checkpoint['args']['arch']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_datasets(args.dataset, train_proportion=args.proportion)

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

best_acc = 0
best_loss = float('inf')
if args.dataset == 'spirograph':
    X,y = encode_train_set_spirograph(clftrainloader, device, net, col_distort, batch_transform)
    X_test, y_test = encode_train_set_spirograph(testloader, device, net, col_distort, batch_transform)
else:
    X, y = encode_train_set(clftrainloader, device, net)
for reg_weight in torch.exp(math.log(10) * torch.linspace(args.reg_lower, args.reg_upper, args.num_steps,
                                                              dtype=torch.float, device=device)):
    if args.dataset == 'spirograph':
        clf = train_reg(X, y, device, reg_weight=reg_weight)
        loss = test_reg(X_test, y_test, clf)
        if loss < best_loss:
            best_loss = loss
        print("Best test accuracy", best_loss)
    else:
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=reg_weight)
        acc, loss = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
        print("Best test accuracy", best_acc, "%")
