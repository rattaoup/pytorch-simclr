'''Train CIFAR10/100 with PyTorch using standard Contrastive Learning. This script tunes the L2 reg weight of the
final classifier.'''
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn as nn

import os
import argparse
from tqdm import tqdm

from models import *
from configs import get_datasets, get_root, get_datasets_from_transform
from evaluate import train_clf, test_matrix
import torch.optim as optim

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--max-num-passes", type=int, default=20, help='Max number of passes to average')
parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
parser.add_argument("--no-crop", action='store_true', help="Don't use crops, only feature averaging with colour "
                                                           "distortion")
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
if args.no_crop:
    transform = transforms.ToTensor()
    root = get_root(args.dataset)
    _, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_datasets_from_transform(
        args.dataset, root, transform, transform, transform, train_proportion=args.proportion
    )
else:
    _, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_datasets(
        args.dataset, augment_clf_train=True, augment_test=True, train_proportion=args.proportion)

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

batch_transform = batch_transform.to(device)


def encode_feature_averaging(clftrainloader, device, net, target=None, num_passes=10):
    if target is None:
        target = device

    net.eval()

    X, y = [], None
    with torch.no_grad():
        for i in tqdm(range(num_passes)):
            store = []
            for batch_idx, (inputs, targets) in enumerate(clftrainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                rn = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                inputs = batch_transform(inputs, rn)
                representation = net(inputs)
                representation, targets = representation.to(target), targets.to(target)
                store.append((representation, targets))

            Xi, y = zip(*store)
            Xi, y = torch.cat(Xi, dim=0), torch.cat(y, dim=0)
            X.append(Xi)

    X = torch.stack(X, dim=0)

    return X, y


X, y = encode_feature_averaging(clftrainloader, device, net, num_passes=args.max_num_passes, target='cpu')
X_test, y_test = encode_feature_averaging(testloader, device, net, num_passes=args.max_num_passes, target='cpu')
torch.save([X, y, X_test, y_test], './checkpoint'+args.load_from[:-4]+'_encode100.pth')
