'''Train an encoder using Contrastive Learning.'''
import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchlars import LARS
from tqdm import tqdm

from configs import get_datasets
from critic import LinearCritic
from evaluate import save_checkpoint,save_checkpoint2, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR
from augmentation import ManualNormalise, DifferentiableColourDistortionByTorch_manual, gen_lambda
from torchvision import transforms

import torch.autograd as autograd

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--base-lr', default=0.25, type=float, help='base learning rate, rescaled by batch_size/256')
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
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

# args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
# args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
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

##############################################################
# Critic
##############################################################
critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)



#differentiable augmentation parameter
brightness_bound = [0.2, 1.8]
contrast_bound = [0.2, 1.8]
saturation_bound = [0.2, 1.8]
hue_bound = [-0.2, 0.2]

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6,
                           momentum=args.momentum)
if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', args.resume)
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint['net'])
    critic.load_state_dict(checkpoint['critic'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    encoder_optimizer.load_state_dict(checkpoint['encoder_optim'])
    base_optimizer.load_state_dict(checkpoint['base_optim'])




# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)

        #colour augmentation
        B = x1.size()[0]
        brightness_list1, saturation_list1, contrast_list1, hue_list1 = gen_lambda(B, brightness_bound,
                                                                                   contrast_bound,
                                                                                   saturation_bound,
                                                                                   hue_bound)
        aug_manual1 = DifferentiableColourDistortionByTorch_manual(brightness = brightness_list1,
                                                                  contrast = contrast_list1,
                                                                  saturation = saturation_list1,
                                                                  hue = hue_list1)

        brightness_list2, saturation_list2, contrast_list2, hue_list2 = gen_lambda(B, brightness_bound,
                                                                                   contrast_bound,
                                                                                   saturation_bound,
                                                                                   hue_bound)

        aug_manual2 = DifferentiableColourDistortionByTorch_manual(brightness = brightness_list2,
                                                                  contrast = contrast_list2,
                                                                  saturation = saturation_list2,
                                                                  hue = hue_list2)

        x1, x2 = aug_manual1(x1), aug_manual2(x2)
        x1, x2 = ManualNormalise(x1, args.dataset), ManualNormalise(x2, args.dataset)

        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)
        #gradient penalty
        gradient_lambda = []
        lambda_list = [brightness_list1, saturation_list1, contrast_list1, hue_list1,
                       brightness_list2, saturation_list2, contrast_list2, hue_list2 ]

        for aug_variable in lambda_list:
            gradients_augvar =  autograd.grad(outputs = loss,
                                 inputs = aug_variable,
                                 retain_graph = True,
                                 grad_outputs = torch.ones_like(aug_variable).to(device))[0]
            gradient_lambda.append(gradients_augvar)

        gradient_lambda = torch.stack(gradient_lambda, dim = 0)

        # currently gradient_lamba.sum([0]).sqrt() has nan element? so I use norm instead
        gradient_penalty = gradient_lambda.sum([0]).norm(2)

        loss_gp = loss + 0.0001*gradient_penalty
        loss_gp.backward()
        encoder_optimizer.step()

        train_loss += loss_gp.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))


for epoch in range(start_epoch, start_epoch + args.num_epochs):
    train(epoch)
    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
        acc = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
        save_checkpoint2(net, clf, critic, epoch, args, os.path.basename(__file__), base_optimizer, encoder_optimizer)
    elif args.test_freq == 0:
        save_checkpoint2(net, clf, critic, epoch, args, os.path.basename(__file__), base_optimizer, encoder_optimizer)
    if args.cosine_anneal:
        scheduler.step()
