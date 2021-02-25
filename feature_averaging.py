import torch
import torch.backends.cudnn as cudnn

import os
import argparse
from tqdm import tqdm

from models import *
from configs import get_datasets
from evaluate import train_clf, test_matrix, train_reg, test_reg


def encode_feature_averaging(clftrainloader, device, net, col_distort, batch_transform, target=None, num_passes=10):
    if target is None:
        target = device

    net.eval()

    X, y = [], None
    with torch.no_grad():
        for _ in tqdm(range(num_passes)):
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


def main(args):
    # Load checkpoint.
    print('==> Loading settings from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', args.load_from)
    checkpoint = torch.load(resume_from)
    args.dataset = checkpoint['args']['dataset']
    args.arch = checkpoint['args']['arch']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = 'reg' if args.dataset == 'spirograph' else 'clf'

    # Data
    print('==> Preparing data..')
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

    results = {}
    X, y = encode_feature_averaging(clftrainloader, device, net, col_distort, batch_transform,
                                    num_passes=args.max_num_passes, target='cpu')
    X_test, y_test = encode_feature_averaging(testloader, device, net, col_distort, batch_transform,
                                              num_passes=args.max_num_passes, target='cpu')
    for m in torch.linspace(args.min_num_passes, args.max_num_passes, args.step_num_passes):
        m = int(m)
        print("FA with M =", m)
        X_this = X[:m, ...].mean(0)
        X_test_this = X_test[:m, ...].mean(0)
        if task == "clf":
            clf = train_clf(X_this.to(device), y.to(device), net.representation_dim, num_classes, device, reg_weight=args.reg_weight)
            acc, loss = test_matrix(X_test_this.to(device), y_test.to(device), clf)
        elif task == "reg":
            clf = train_reg(X_this, y, device, reg_weight=args.reg_weight)
            loss = test_reg(X_test_this, y_test, clf)
            acc = None
        else:
            raise ValueError("Unexpected task type: %s" % task)
        results[m] = (acc,loss)
    print(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
    parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
    parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
    parser.add_argument("--max-num-passes", type=int, default=20, help='Max number of passes to average')
    parser.add_argument("--min-num-passes", type=int, default=10, help='Min number of passes to average')
    parser.add_argument("--step-num-passes", type=int, default=2, help='Number of distinct M values to try')
    parser.add_argument("--reg-weight", type=float, default=1e-5, help='Regularization parameter')
    parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
    args = parser.parse_args()
    main(args)



