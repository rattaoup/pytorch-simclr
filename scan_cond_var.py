'''Train CIFAR10/100 with PyTorch using standard Contrastive Learning. This script tunes evaluates the invariance of
the learned representations.'''
import torch
import torch.backends.cudnn as cudnn

import math
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

from augmentation import ColourDistortion, TensorNormalise, ModuleCompose
from models import *
from configs import get_datasets, get_mean_std
from evaluate import train_reg, test_reg

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--baselines", type=str, default='ckpt', help='File series to load for baseline')
parser.add_argument("--ours", type=str, default='invgpn', help='File series to load for our method')
parser.add_argument("--num-passes", default=10, type=int, help='Num samples to compute')
parser.add_argument("--min-epoch", default=99, type=int, help='First epoch checkpoint to take')
parser.add_argument("--max-epoch", default=1000, type=int, help='Last epoch checkpoint to take')
parser.add_argument("--step-epoch", default=100, type=int, help='Spacing between checkpoints')
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

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=100, shuffle=False, num_workers=args.num_workers,
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

    def compute_cond_var(clftrainloader, device, net, target=None):
        if target is None:
            target = device

        net.eval()

        with torch.no_grad():
            store = []
            for batch_idx, (inputs, _) in tqdm(enumerate(clftrainloader), total=len(clftrainloader)):
                inputs = inputs.to(device)
                B = inputs.shape[0]
                rn = col_distort.sample_random_numbers((args.num_passes * B, *inputs.shape[1:]), inputs.device)
                inputs = torch.repeat_interleave(inputs, args.num_passes, dim=0)
                inputs = batch_transform(inputs, rn)
                representation = net(inputs)
                representation = representation / representation.norm(p=2, dim=-1, keepdim=True)
                representation = representation.reshape(args.num_passes, B, representation.shape[-1])
                representation_proj = (representation * \
                        torch.bernoulli(.5 * torch.ones(*representation.shape, device=device)) * 2 - 1).sum(-1)
                cond_var = representation_proj.var(0)
                store.append(cond_var)

            trcov_all = torch.cat(store, dim=0)

        return trcov_all.mean()


    return compute_cond_var(testloader, device, net)



baselines = args.baselines.split(",")
ours = args.ours.split(",")
results = defaultdict(list)
for stem in baselines+ours:
    for epoch in range(args.min_epoch, args.max_epoch, args.step_epoch):
        fname = stem + '_epoch{:03d}.pth'.format(epoch)
        loss = get_loss(fname)
        results[stem].append(loss)

print(results)
