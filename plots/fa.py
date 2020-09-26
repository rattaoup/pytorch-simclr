import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    baseline = [(86.61, 0.4277), (93.99, 0.1732), (94.28, 0.1658), (94.24, 0.1633), (94.41, 0.1618), (94.43, 0.1611),
     (94.43, 0.1604), (94.47, 0.1599), (94.54, 0.1591), (94.62, 0.1586)]

    inv = [(86.94, 0.4224), (94.45, 0.1678), (94.58, 0.1602), (94.62, 0.1569), (94.65, 0.1553), (94.76, 0.1546),
     (94.8, 0.1542), (94.85, 0.1537), (94.91, 0.1531), (94.87, 0.1533)]

    m = [int(x) for x in np.linspace(1, 100, 10)]

    plt.figure(figsize=(5.5, 3.5))
    # col = '#2ba0c4'
    col = '#377e82'
    plt.plot(m, [x[0] for x in baseline], color=col, marker='x', markersize=7)
    # plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    col2 = '#375e82'
    plt.plot(m, [x[0] for x in inv], color=col2, marker='.', markersize=7)
    # plt.fill_between(epochs, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.hlines(y=93., color='#377e82', xmin=1, xmax=100)
    plt.hlines(y=93.44, color='#375e82', xmin=1, xmax=100)

    plt.xlabel('$M$')
    plt.ylabel('Test loss')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='upper right', fontsize=12, frameon=False)

    plt.show()
