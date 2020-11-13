'''Train an encoder using Contrastive Learning.'''
import argparse
import os
import subprocess
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchlars import LARS
from tqdm import tqdm

from augmentation import ColourDistortion, TensorNormalise, ModuleCompose
from configs import get_datasets, get_mean_std
from critic import MoCoTwoLayerCritic
from evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--base-lr', default=0.03, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint with this filename')
parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset',
                    choices=['cifar10', 'cifar100', 'stl10', 'imagenet'])
parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=100, help='Number of training epochs')
parser.add_argument("--cosine-anneal", action='store_true', help="Use cosine annealing on the learning rate")
parser.add_argument("--arch", type=str, default='resnet50', help='Encoder architecture',
                    choices=['resnet18', 'resnet34', 'resnet50'])
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--save-freq", type=int, default=100, help='Frequency to save checkpoints.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
parser.add_argument("--cut-off", type=int, default=1000, help='last epoch')
parser.add_argument("--moco_k", type=int, default=2048, help='Cache length for MoCo')
parser.add_argument("--moco-m", type=float, default=0.99, help='Momentum parameter for MoCo')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = defaultdict(list)
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

# drop_last so that the queue works correctly
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
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
    net_k = ResNet18(stem=stem)
elif args.arch == 'resnet34':
    net = ResNet34(stem=stem)
    net_k = ResNet34(stem=stem)
elif args.arch == 'resnet50':
    net = ResNet50(stem=stem)
    net_k = ResNet50(stem=stem)
else:
    raise ValueError("Bad architecture specification")
for param_q, param_k in zip(net.parameters(), net_k.parameters()):
    param_k.data.copy_(param_q.data)  # initialize
    param_k.requires_grad = False  # not update by gradient

net = net.to(device)
net_k = net_k.to(device)
queue = torch.randn(args.moco_k, 128)
queue = nn.functional.normalize(queue, dim=1)
queue = queue.to(device)
ptr = 0

##############################################################
# Critic
##############################################################
critic = MoCoTwoLayerCritic(net.representation_dim, temperature=args.temperature)
critic_k = MoCoTwoLayerCritic(net.representation_dim, temperature=args.temperature)
for param_q, param_k in zip(critic.parameters(), critic_k.parameters()):
    param_k.data.copy_(param_q.data)  # initialize
    param_k.requires_grad = False  # not update by gradient
critic = critic.to(device)
critic_k = critic_k.to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net_k = torch.nn.DataParallel(net_k)
    net.representation_dim = repr_dim
    net_k.representation_dim = repr_dim
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6,
                       momentum=args.momentum)
if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(optimizer, args.num_epochs)
# MoCo does not use LARS
# encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)


if args.resume:
    print("Resuming not support for MoCo")
    raise SystemExit


col_distort = ColourDistortion(s=0.5)
batch_transform = ModuleCompose([
        col_distort,
        TensorNormalise(*get_mean_std(args.dataset))
    ]).to(device)

@torch.no_grad()
def momentum_update_key_encoder():
    for param_q, param_k in zip(net.parameters(), net_k.parameters()):
        param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
    for param_q, param_k in zip(critic.parameters(), critic_k.parameters()):
        param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)


@torch.no_grad()
def dequeue_and_enqueue(keys):
    global ptr
    batch_size = keys.shape[0]

    assert args.moco_k % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    queue[ptr:ptr + batch_size, :] = keys.detach()
    ptr = (ptr + batch_size) % args.moco_k  # move pointer


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_total_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        rn1, rn2 = col_distort.sample_random_numbers(x1.shape, x1.device), col_distort.sample_random_numbers(x2.shape, x2.device)
        x1, x2 = batch_transform(x1, rn1), batch_transform(x2, rn2).detach()

        # Encode k
        with torch.no_grad():
            momentum_update_key_encoder()
            k = critic_k.project(net_k((x2)))

        optimizer.zero_grad()
        q = critic.project(net(x1))

        raw_scores, pseudotargets = critic_k(q, k, queue)
        dequeue_and_enqueue(k)
        contrastive_loss = criterion(raw_scores, pseudotargets)

        loss_gp = contrastive_loss

        loss_gp.backward()
        optimizer.step()

        train_total_loss += loss_gp.item()

        t.set_description('Loss: %.3f' % (train_total_loss / (batch_idx + 1)))

    return train_total_loss / len(trainloader), 


def update_results(train_total_loss, test_loss, test_acc):
    results['train_total_loss'].append(train_total_loss)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)


for epoch in range(start_epoch, min(args.cut_off,start_epoch + args.num_epochs)):
    outputs = train(epoch)
    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
        acc, test_loss = test(testloader, device, net, clf)
        update_results(*outputs, test_loss, acc)
    if (epoch % args.save_freq == (args.save_freq - 1)):
        save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__), results)
    if args.cosine_anneal:
        scheduler.step()