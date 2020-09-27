import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return 2 * array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    # data = {'aam-baseline-1.pth': [np.array(0.5232), np.array(0.5707), np.array(0.6010), np.array(0.6210), np.array(0.6581),
    #                         np.array(0.6681), np.array(0.6873), np.array(0.6977), np.array(0.6931), np.array(0.6887)],
    #  'aam-baseline-2.pth': [np.array(0.5241), np.array(0.5644), np.array(0.5901), np.array(0.6197), np.array(0.6442),
    #                         np.array(0.6668), np.array(0.6926), np.array(0.7116), np.array(0.7015), np.array(0.6979)],
    #  'aam-baseline-3.pth': [np.array(0.5346), np.array(0.5652), np.array(0.5903), np.array(0.6247), np.array(0.6345),
    #                         np.array(0.6698), np.array(0.6859), np.array(0.6927), np.array(0.6935), np.array(0.6954)],
    #  'invgpn-1e-1-1': [np.array(0.0368), np.array(0.0263), np.array(0.0198), np.array(0.0141), np.array(0.0108),
    #                    np.array(0.0076), np.array(0.0058), np.array(0.0047), np.array(0.0042), np.array(0.0041)],
    #  'invgpn-1e-1-2': [np.array(0.0406), np.array(0.0273), np.array(0.0181), np.array(0.0135), np.array(0.0103),
    #                    np.array(0.0077), np.array(0.0058), np.array(0.0047), np.array(0.0043), np.array(0.0042)]}
    data = {'aam-baseline-1.pth': [6.9421, 7.3924, 7.0669, 7.3108, 7.3018, 6.0960, 6.0507, 5.9493, 6.0909, 6.2383],
     'aam-baseline-2.pth': [7.9082, 7.5181, 7.4649, 6.6028, 6.2907, 5.9757, 5.9316, 5.9842, 5.9969, 6.2311],
     'aam-baseline-3.pth': [7.0199, 7.8414, 6.9805, 6.9545, 7.2471, 6.1085, 6.0561, 6.1005, 6.2059, 6.1712],
     'invgpn-1e-1-1': [1.2956, 1.2907, 1.3168, 1.2430, 1.1442, 1.1071, 1.0785, 1.0752, 1.0788, 1.0848],
     'invgpn-1e-1-2': [1.2576, 1.3683, 1.1904, 1.1973, 1.1432, 1.1063, 1.0802, 1.0819, 1.0904, 1.0832],
     'invgpn-1e-1-4': [1.2231, 1.3864, 1.3539, 1.2192, 1.1478, 1.1128, 1.0843, 1.0725, 1.0836, 1.0851]}

    baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
    baseline = np.stack([np.stack(data[x]) for x in baselines])
    invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
    inv = np.stack([np.array(data[x]) for x in invs])

    plt.figure(figsize=(4.5, 3.5))

    epochs = np.array(range(100, 1001, 100))
    base_mean, base_se = baseline.mean(0), se(baseline, 0)
    #col = '#2ba0c4'
    col = '#1f77b4'
    plt.plot(epochs, base_mean, color=col, marker='o', markersize=7)
    plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv.mean(0), se(inv, 0)
    col2 = '#ff7f0e'
    plt.plot(epochs, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(epochs, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.xlabel('Epoch')
    plt.ylabel('Conditional variance')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='center right', fontsize=12, frameon=False)

    plt.show()