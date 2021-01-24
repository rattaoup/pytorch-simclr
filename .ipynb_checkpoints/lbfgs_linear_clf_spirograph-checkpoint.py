'''This script tunes the L2 reg weight of the final classifier.'''
import argparse
import os
import math

import torch
import torch.backends.cudnn as cudnn

from configs import get_datasets, get_spirograph_dataset
from evaluate import encode_train_set, train_clf, test, train_reg, test_reg
from models import *
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--reg-lower", type=float, default=-8, help='Minimum log regularization parameter (base 10)')
parser.add_argument("--reg-upper", type=float, default=-8, help='Maximum log regularization parameter (base 10)')
parser.add_argument("--num-steps", type=int, default=1, help='Number of log-linearly spaced reg parameters to try')
parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
parser.add_argument("--fore-lower", type = float, default=0.4, help = 'Lower bound for fore rgb augmentation')
parser.add_argument("--fore-upper", type = float, default=1.0, help = 'Upper bound for fore rgb augmentation')
parser.add_argument("--back-lower", type = float, default=0, help = 'Lower bound for back rgb augmentation')
parser.add_argument("--back-upper", type = float, default=0.6, help = 'Upper bound for back rgb augmentation')
parser.add_argument("--fore-shift", type =float, default = 0.0, help ='Shift mean of foreground aug')
parser.add_argument("--back-shift", type = float, default = 0.0, help = 'Shift mean of background aug')
parser.add_argument("--fore-shift-upper",type = float, default = 0, help = 'Upper shift mean of foreground aug')
parser.add_argument("--fore-shift-lower",type = float, default = 0, help = 'Lower shift mean of foreground aug')
parser.add_argument("--back-shift-upper",type = float, default = 0, help = 'Upper shift mean of background aug')
parser.add_argument("--back-shift-lower",type = float, default = 0, help = 'Lower shift mean of background aug')
parser.add_argument("--h-shift-upper",type = float, default = 0, help = 'Upper shift mean of h aug')
parser.add_argument("--h-shift-lower",type = float, default = 0, help = 'Lower shift mean of h aug')
parser.add_argument("--fore-num-passes", type=int, default=1, help='Number of shift param to try')
parser.add_argument("--back-num-passes", type=int, default=1, help='Number of shift param to try')
parser.add_argument("--h-num-passes", type=int, default=1, help='Number of shift param to try')
parser.add_argument("--norm-rep", action = 'store_true' ,help = 'Normalise representation h or not')

parser.add_argument("--fore-var-upper",type = float, default = 0, help = 'Upper  var of foreground aug')
parser.add_argument("--fore-var-lower",type = float, default = 0, help = 'Lower  var of foreground aug')
parser.add_argument("--fore-var-num-passes", type=int, default=1, help='Number of var fore param to try')
parser.add_argument("--back-var-upper",type = float, default = 0, help = 'Upper  var of foreground aug')
parser.add_argument("--back-var-lower",type = float, default = 0, help = 'Lower  var of foreground aug')
parser.add_argument("--back-var-num-passes", type=int, default=1, help='Number of var fore param to try')
parser.add_argument("--h-var-upper",type = float, default = 0, help = 'Upper shift var of h aug')
parser.add_argument("--h-var-lower",type = float, default = 0, help = 'Lower shift var of h aug')
parser.add_argument("--h-var-num-passes", type=int, default=1, help='Number of var h param to try')



args = parser.parse_args()

# Load checkpoint.
# print('==> Loading settings from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
resume_from = os.path.join('./checkpoint', args.load_from)
checkpoint = torch.load(resume_from)
args.dataset = checkpoint['args']['dataset']
args.arch = checkpoint['args']['arch']

device = 'cuda' if torch.cuda.is_available() else 'cpu'




# Data
# print('==> Preparing data..')
def get_loss(fore_shift, back_shift, h_shift, fore_var, h_var, back_var):
    trainset, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_spirograph_dataset(rgb_fore_bounds = (.4 + fore_shift-fore_var , 1 + fore_shift+fore_var), rgb_back_bounds=(0 + back_shift - back_var, .6 + back_shift+ back_var), h_bounds = (.5 + h_shift-h_var, 2.5+ h_shift+h_var), train_proportion=args.proportion)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True)

    # Model
    # print('==> Building model..')
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

    # print('==> Loading encoder from checkpoint..')
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
        if args.norm_rep:
            X = X / X.norm(p=2, dim=-1, keepdim=True)
            X_test = X_test / X_test.norm(p=2, dim=-1, keepdim=True)
        
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
    return loss

results = defaultdict(list)
fore_list = []
back_list = []
loss_list = []
h_list = []
for fore_shift in torch.linspace(args.fore_shift_lower, args.fore_shift_upper, args.fore_num_passes):
    for back_shift in torch.linspace(args.back_shift_lower, args.back_shift_upper, args.back_num_passes):
        for h_shift in torch.linspace(args.h_shift_lower, args.h_shift_upper, args.h_num_passes):
            for fore_var in torch.linspace(args.fore_var_lower, args.fore_var_upper, args.fore_var_num_passes):
                for h_var in torch.linspace(args.h_var_lower, args.h_var_upper, args.h_var_num_passes):
                    for back_var in torch.linspace(args.back_var_lower, args.back_var_upper, args.back_var_num_passes):
                        fname = 'fore_shift ' + str(fore_shift) + ', back_shift ' + str(back_shift) + ' h_shift ' + str(h_shift) + ' fore_var '+ str(fore_var) + ' h_var ' + str(h_var) + ' back_var ' + str(back_var)
                        loss = get_loss(fore_shift, back_shift, h_shift, fore_var, h_var, back_var)
                        print(fname)
                        results[fname].append(loss)
                        loss_list.append(loss)

print(results)
print('loss_list')
print(loss_list)

