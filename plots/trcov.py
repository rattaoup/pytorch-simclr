import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    data = {'aam-baseline-1.pth': [np.array(0.5232), np.array(0.5707), np.array(0.6010), np.array(0.6210), np.array(0.6581),
                            np.array(0.6681), np.array(0.6873), np.array(0.6977), np.array(0.6931), np.array(0.6887)],
     'aam-baseline-2.pth': [np.array(0.5241), np.array(0.5644), np.array(0.5901), np.array(0.6197), np.array(0.6442),
                            np.array(0.6668), np.array(0.6926), np.array(0.7116), np.array(0.7015), np.array(0.6979)],
     'aam-baseline-3.pth': [np.array(0.5346), np.array(0.5652), np.array(0.5903), np.array(0.6247), np.array(0.6345),
                            np.array(0.6698), np.array(0.6859), np.array(0.6927), np.array(0.6935), np.array(0.6954)],
     'invgpn-1e-1-1': [np.array(0.0368), np.array(0.0263), np.array(0.0198), np.array(0.0141), np.array(0.0108),
                       np.array(0.0076), np.array(0.0058), np.array(0.0047), np.array(0.0042), np.array(0.0041)],
     'invgpn-1e-1-2': [np.array(0.0406), np.array(0.0273), np.array(0.0181), np.array(0.0135), np.array(0.0103),
                       np.array(0.0077), np.array(0.0058), np.array(0.0047), np.array(0.0043), np.array(0.0042)]}

    baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
    baseline = np.stack([np.stack(data[x]) for x in baselines])
    invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2']
    inv = np.stack([np.array(data[x]) for x in invs])

    plt.figure(figsize=(5.5, 3.5))

    epochs = np.array(range(100, 1001, 100))
    base_mean, base_se = baseline.mean(0), se(baseline, 0)
    #col = '#2ba0c4'
    col = '#377e82'
    plt.plot(epochs, base_mean, color=col, marker='x', markersize=7)
    plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv.mean(0), se(inv, 0)
    col2 = '#375e82'
    plt.plot(epochs, inv_mean, color=col2, marker='.', markersize=7)
    plt.fill_between(epochs, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.xlabel('Epoch')
    plt.ylabel('Conditional variance')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='center right', fontsize=12, frameon=False)

    plt.show()