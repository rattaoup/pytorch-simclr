import torch
import numpy as np
import matplotlib.pyplot as plt


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
    invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2']
    baseline_acc = []
    baseline_loss = []
    for bname in baselines:
        checkpoint = torch.load('checkpoint/resnet50/'+bname+'_epoch999.pth', map_location='cpu')
        baseline_acc.append(checkpoint['results']['test_acc'])
        baseline_loss.append(checkpoint['results']['test_loss'])
    inv_acc = []
    inv_loss = []
    for bname in invs:
        checkpoint = torch.load('checkpoint/resnet50/' + bname + '_epoch999.pth', map_location='cpu')
        inv_acc.append(checkpoint['results']['test_acc'])
        inv_loss.append(checkpoint['results']['test_loss'])

    baseline_acc = np.stack([np.array(x) for x in baseline_acc])
    baseline_loss = np.stack([np.array(x) for x in baseline_loss])
    inv_acc = np.stack([np.array(x) for x in inv_acc])
    inv_loss = np.stack([np.array(x) for x in inv_loss])

    base_mean, base_se = baseline_acc.mean(0), se(baseline_acc, 0)
    print(base_mean[-1], base_se[-1])
    base_mean, base_se = baseline_loss.mean(0), se(baseline_loss, 0)
    print(base_mean[-1], base_se[-1])

    inv_mean, inv_se = inv_acc.mean(0), se(inv_acc, 0)
    print(inv_mean[-1], inv_se[-1])
    inv_mean, inv_se = inv_loss.mean(0), se(inv_loss, 0)
    print(inv_mean[-1], inv_se[-1])