'''Train CIFAR10/100 with PyTorch using standard Contrastive Learning. This script tunes the L2 reg weight of the
final classifier.'''
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import os
import argparse
from tqdm import tqdm

from augmentation import TensorColorJitter, TensorNormalise, ModuleCompose
from models import *
from configs import get_datasets, get_mean_std, get_root, get_datasets_from_transform
from evaluate import train_clf, test_matrix

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--num-passes", type=int, default=1, help='Number of passes to average')
parser.add_argument("--reg-weight", type=float, default=1e-5, help='Regularization parameter')
parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
parser.add_argument("--max-s", type=float, default=0.9, help='Max colour distortion strength to use')
parser.add_argument("--min-s", type=float, default=1e-5, help='Min colour distortion strength to use')
parser.add_argument("--step-s", type=int, default=8, help='Number of steps of colour distortion')
parser.add_argument("--mean-shift", action="store_true", help='Apply mean shift, rather than variance shift')
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
root = get_root(args.dataset)
transform = transforms.ToTensor()
_, testset, clftrainset, num_classes, stem, _, _ = get_datasets_from_transform(
    args.dataset, root, transform, transform, transform, train_proportion=args.proportion
)


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


def encode_feature_averaging(clftrainloader, device, net, target=None, num_passes=10, s=0.5):
    if target is None:
        target = device

    if args.mean_shift:
        col_distort = TensorColorJitter(1e-5, 1e-5, 1e-5, 1e-5,
                                        1 + 0.8*s, 1 + 0.8*s, 1 + 0.8*s, 0.5*s)
    else:
        col_distort = TensorColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    batch_transform = ModuleCompose([
        col_distort,
        TensorNormalise(*get_mean_std(args.dataset))
    ]).to(device)

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

    X = torch.stack(X, dim=0).mean(0)

    return X, y


results = []
for s in torch.linspace(args.min_s, args.max_s, args.step_s):
    s = s.item()
    print("\nStrength =", s)
    X, y = encode_feature_averaging(clftrainloader, device, net, num_passes=args.num_passes, target='cpu', s=s)
    X_test, y_test = encode_feature_averaging(testloader, device, net, num_passes=args.num_passes, target='cpu', s=s)
    clf = train_clf(X.to(device), y.to(device), net.representation_dim, num_classes, device, reg_weight=args.reg_weight)
    acc, loss = test_matrix(X_test.to(device), y_test.to(device), clf)
    results.append((acc,loss))
print(results)
