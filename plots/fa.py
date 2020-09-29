import matplotlib.pyplot as plt
import numpy as np

def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])

if __name__ == '__main__':
    baseline = [(86.61, 0.4277), (93.99, 0.1732), (94.28, 0.1658), (94.24, 0.1633), (94.41, 0.1618), (94.43, 0.1611),
     (94.43, 0.1604), (94.47, 0.1599), (94.54, 0.1591), (94.62, 0.1586)]

    # inv = [(86.94, 0.4224), (94.45, 0.1678), (94.58, 0.1602), (94.62, 0.1569), (94.65, 0.1553), (94.76, 0.1546),
    #  (94.8, 0.1542), (94.85, 0.1537), (94.91, 0.1531), (94.87, 0.1533)]
    invs = [[(93.4, 0.19084855914115906), (94.52, 0.16708481311798096), (94.71, 0.16115325689315796),
             (94.73, 0.15988996624946594), (94.69, 0.15886880457401276), (94.76, 0.15774832665920258),
             (94.79, 0.15716587007045746), (94.75, 0.15680773556232452), (94.77, 0.15641529858112335),
             (94.84, 0.1561741977930069)],
             [(93.58, 0.18735511600971222), (94.48, 0.16065604984760284), (94.66, 0.15743771195411682),
              (94.75, 0.15562307834625244), (94.74, 0.1553688794374466), (94.7, 0.1548636257648468),
              (94.72, 0.15395040810108185), (94.75, 0.1541283130645752), (94.81, 0.1539543867111206),
              (94.76, 0.15337489545345306)],
             [(93.55, 0.18709571659564972), (94.52, 0.16025693714618683), (94.84, 0.15642397105693817),
              (94.77, 0.1557120978832245), (94.79, 0.1545531451702118), (94.83, 0.15411685407161713),
              (94.78, 0.1541195511817932), (94.8, 0.15403395891189575), (94.8, 0.15365758538246155),
              (94.89, 0.15360258519649506)]
            ]

    m = [int(x) for x in np.linspace(5, 100, 10)]
    inv_acc = np.stack([np.array([x[0] for x in y]) for y in invs])

    plt.figure(figsize=(4.5, 3.5))
    # # col = '#2ba0c4'
    # col = '#377e82'
    # plt.plot(m, [x[0] for x in baseline], color=col, marker='x', markersize=7)
    # # plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv_acc.mean(0), se(inv_acc, 0)
    col2 = '#ff7f0e'
    plt.plot(m, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(m, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    col3='#8c564b'
    base_mean, base_se = 93.34 * np.ones(len(m)), 0.04 * np.ones(len(m))
    plt.plot(m, base_mean, color=col3, marker='v', markersize=7)
    plt.fill_between(m, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col3)

    plt.xlabel('$M$')
    plt.ylabel('Test accuracy')
    plt.legend(['Training transformations', 'No transformations'], loc='center right', fontsize=12, frameon=False)

    plt.show()
