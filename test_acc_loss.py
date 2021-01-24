import torch
import numpy as np
import matplotlib.pyplot as plt


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    # baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
    # invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
    baselines = ['aam-cifar100-1', 'aam-cifar100-2', 'aam-cifar100-3']
    invs = ['invgpn-cifar100-1e-1-1', 'invgpn-cifar100-1e-1-2', 'invgpn-cifar100-1e-1-3']
    baseline_acc = []
    baseline_loss = []
    for bname in baselines:
        checkpoint = torch.load('checkpoint/resnet50/'+bname+'_epoch999.pth', map_location='cpu')
        baseline_acc.append(checkpoint['results']['test_acc'][-1])
        baseline_loss.append(checkpoint['results']['test_loss'][-1])
    inv_acc = []
    inv_loss = []
    for bname in invs:
        checkpoint = torch.load('checkpoint/resnet50/' + bname + '_epoch999.pth', map_location='cpu')
        inv_acc.append(checkpoint['results']['test_acc'][-1])
        inv_loss.append(checkpoint['results']['test_loss'][-1])

    baseline_acc = np.stack([np.array(x) for x in baseline_acc])
    baseline_loss = np.stack([np.array(x) for x in baseline_loss])
    inv_acc = np.stack([np.array(x) for x in inv_acc])
    inv_loss = np.stack([np.array(x) for x in inv_loss])

    base_mean, base_se = baseline_acc.mean(0), se(baseline_acc, 0)
    print(base_mean, base_se)
    base_mean, base_se = baseline_loss.mean(0), se(baseline_loss, 0)
    print(base_mean, base_se)

    inv_mean, inv_se = inv_acc.mean(0), se(inv_acc, 0)
    print(inv_mean, inv_se)
    inv_mean, inv_se = inv_loss.mean(0), se(inv_loss, 0)
    print(inv_mean, inv_se)