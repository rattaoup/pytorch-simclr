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
from evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR
from aam_tools import get_datasets_aam, DifferentiableColourDistortionByTorch3
from torchvision import transforms
from gp_tools import gen_lambda, DifferentiableColourDistortionByTorch_manual
import torch.autograd as autograd
from update_checkpoint import update_checkpoint


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
parser.add_argument("--norm", type=float, default=2, help='Norm for gradient penalty')
parser.add_argument("--lambda-gp", type=float, default=0.001, help='lambda_gp for gradient penalty')
parser.add_argument("--cut-off", type=int, default=1000, help='cut off epochs')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

# args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
# args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
last_epoch = args.num_epochs #last epoch or from last epoch from checkpoint
clf = None
current_checkpoint = {}
acc = 0

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets_aam(args.dataset)

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
#     best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    encoder_optimizer.load_state_dict(checkpoint['encoder_optim'])
    base_optimizer.load_state_dict(checkpoint['base_optim'])
    scheduler.step(start_epoch)
    if 'num_epochs' in checkpoint:
        last_epoch = checkpoint['num_epochs']
    else:
        last_epoch = 1000 #default value for num epochs = 1000
    current_checkpoint = checkpoint





# Augmentation setup
s = 0.5
brightness_bound = [1-0.8*s, 1+0.8*s]
contrast_bound = [1-0.8*s, 1+0.8*s]
saturation_bound = [1-0.8*s, 1+0.8*s]
hue_bound = [-0.2*s, 0.2*s]

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0
    sum_gradient = 0
    contrastive_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)

        #### Differentiable Colour Augmentation
        B = x1.size()[0]

        brightness_list1, saturation_list1, contrast_list1, hue_list1 = gen_lambda(B, brightness_bound,
                                                                                   contrast_bound,
                                                                                   saturation_bound,
                                                                                   hue_bound)
        brightness_list2, saturation_list2, contrast_list2, hue_list2 = gen_lambda(B, brightness_bound,
                                                                                   contrast_bound,
                                                                                   saturation_bound,
                                                                                   hue_bound)
        lambda_ = torch.stack([brightness_list1, saturation_list1, contrast_list1, hue_list1,
                               brightness_list2, saturation_list2, contrast_list2, hue_list2], dim=1)
        aug_manual1 = DifferentiableColourDistortionByTorch_manual(brightness = lambda_[:,0],
                                                                   contrast = lambda_[:,1],
                                                                   saturation = lambda_[:,2],
                                                                   hue = lambda_[:,3])
        aug_manual2 = DifferentiableColourDistortionByTorch_manual(brightness = lambda_[:,4],
                                                                   contrast = lambda_[:,5],
                                                                   saturation = lambda_[:,6],
                                                                   hue = lambda_[:,7])
        x1, x2 = aug_manual1(x1), aug_manual2(x2)
        #####
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)

        ##### Gradient Penalty
        sum_rep1 = representation1.sum()
        sum_rep2 = representation2.sum()
        sum_rep = sum_rep1 + sum_rep2
        gradient_lambda =  autograd.grad(outputs = sum_rep,
                             inputs = lambda_,
                             create_graph = True,
                             retain_graph = True,
                             only_inputs = True)[0]
        gradient_penalty = (gradient_lambda +1e-16).norm(p=args.norm, dim=1).mean()
        loss_gp = loss + args.lambda_gp * gradient_penalty.to(device)
        loss_gp.backward()
        encoder_optimizer.step()

        train_loss += loss_gp.item()
        sum_gradient += gradient_penalty
        contrastive_loss += loss

        t.set_description('gp: {:0.5f} ,  loss: {:0.4f},  final_loss: {:0.4f}'.format((sum_gradient / (batch_idx + 1)), (contrastive_loss / (batch_idx + 1)), (train_loss / (batch_idx + 1))))

    return (sum_gradient / (batch_idx + 1)), (contrastive_loss / (batch_idx + 1)), (train_loss / (batch_idx + 1))




for epoch in range(start_epoch, min(last_epoch,args.cut_off)):
    gradient_penalty, contrastive_loss, final_loss = train(epoch)
    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
        acc = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
        current_checkpoint = update_checkpoint(current_checkpoint, net, clf, critic, epoch, args, os.path.basename(__file__), base_optimizer, encoder_optimizer, args.num_epochs, gradient_penalty, contrastive_loss, final_loss, acc)
    elif args.test_freq == 0:
        current_checkpoint = update_checkpoint(current_checkpoint, net, clf, critic, epoch, args, os.path.basename(__file__), base_optimizer, encoder_optimizer, args.num_epochs, gradient_penalty, contrastive_loss, final_loss, acc)
    if args.cosine_anneal:
        scheduler.step()
