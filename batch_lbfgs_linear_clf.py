import argparse
import os
import math

import torch
import torch.backends.cudnn as cudnn

from configs import get_datasets
from evaluate import encode_train_set, train_clf, test
from models import *

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--start-epochs", type=int, default=99, help='Start epoch (99,199, ....)')
parser.add_argument("--last-epochs", type=int, default=99, help='Last epoch (99,199, ...)')
parser.add_argument("--load-from", type=str, default='resnet18_new.pth', help='File to load from')
args = parser.parse_args()





# Load checkpoint.
epoch_to_evaluate = [args.start_epochs + 100*i for i in range(int((args.last_epochs - args.start_epochs)/100)+1)]
resume_from_list = [args.load_from + str(epoch_to_evaluate[i]) for i in range(len(epoch_to_evaluate))]
best_acc_dict ={}
for load_from in resume_from_list:
    print('==> Loading settings from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', load_from)
    checkpoint = torch.load(resume_from)
    args.dataset = checkpoint['args']['dataset']
    args.arch = checkpoint['args']['arch']

    # Data
    print('==> Preparing data..')
    _, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


    best_acc = 0
    X, y = encode_train_set(clftrainloader, device, net)
    for reg_weight in torch.exp(math.log(10) * torch.linspace(-7, -3, 16, dtype=torch.float, device=device)):
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=reg_weight)
        acc = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
    print("Best test accuracy for model {} is {} %".format(load_from, best_acc))
    best_acc_dict[load_from] = best_acc_dict

for file in best_acc_dict:
    print(file)
    print(best_acc_dict[file])
