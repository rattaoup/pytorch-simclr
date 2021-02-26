import torch
import torch.backends.cudnn as cudnn

import os
import argparse
from tqdm import tqdm

from models import *
from data.configs import get_datasets
from evaluate import train_clf, test_matrix, train_reg, test_reg, encode_train_set


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
    if args.untransformed and args.num_passes > 1:
        raise ValueError("Cannot perform feature averaging with untransformed inputs")
    elif args.untransformed and args.dataset == 'spirograph':
        raise ValueError("Spirograph does not support untransformed inputs")
    aug = not args.untransformed

    # Data
    print('==> Preparing data..')
    _, testset, clftrainset, num_classes, stem, col_distort, batch_transform = get_datasets(
        args.dataset, augment_clf_train=aug, augment_test=aug, train_proportion=args.proportion)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)

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

    if args.untransformed:
        X, y = encode_train_set(clftrainloader, device, net)
        X_test, y_test = encode_train_set(testloader, device, net)
    else:
        X, y = encode_feature_averaging(clftrainloader, device, net, col_distort, batch_transform,
                                        num_passes=args.num_passes, target='cpu')
        X_test, y_test = encode_feature_averaging(testloader, device, net, col_distort, batch_transform,
                                                  num_passes=args.num_passes, target='cpu')
        X = X.mean(0)
        X_test = X_test.mean(0)

    if args.num_passes > 1:
        print("Feature averaging with M =", args.num_passes)

    if task == "clf":
        clf = train_clf(X.to(device), y.to(device), net.representation_dim, num_classes, device,
                        reg_weight=args.reg_weight)
        acc, loss = test_matrix(X_test.to(device), y_test.to(device), clf)
    elif task == "reg":
        regressor = train_reg(X, y, device, reg_weight=args.reg_weight)
        loss = test_reg(X_test, y_test, regressor)
        acc = None
    else:
        raise ValueError("Unexpected task type: %s" % task)
    results = (acc, loss)
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
    parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
    parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
    parser.add_argument("--num-passes", type=int, default=1, help='Number of features to average')
    parser.add_argument("--reg-weight", type=float, default=1e-5, help='Regularization parameter')
    parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
    parser.add_argument("--untransformed", action="store_true", default=False,
                        help="Use untransformed inputs")
    args = parser.parse_args()
    print(main(args))
