import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return 2 * array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    data = {
        'aam-baseline-1.pth': [0.029353315010666847, 0.030347226187586784, 0.029843715950846672, 0.030111605301499367,
                               0.031037887558341026, 0.03176790103316307, 0.032418858259916306, 0.03323134779930115,
                               0.03435888513922691, 0.034906335175037384],
        'aam-baseline-2.pth': [0.029789388179779053, 0.029735427349805832, 0.030534764751791954, 0.0313221700489521,
                               0.031091803684830666, 0.03171573951840401, 0.03318208456039429, 0.03418804332613945,
                               0.03562922403216362, 0.035608433187007904],
        'aam-baseline-3.pth': [0.03013007529079914, 0.030110040679574013, 0.030412763357162476, 0.030614838004112244,
                               0.03095303289592266, 0.03225488215684891, 0.03280729800462723, 0.03426627442240715,
                               0.034937694668769836, 0.03530283644795418],
        'invgpn-1e-1-1': [0.029118286445736885, 0.028729351237416267, 0.02924526296555996, 0.029064301401376724,
                          0.029236460104584694, 0.030397707596421242, 0.04180039092898369, 0.041689690202474594,
                          0.04145955666899681, 0.04144816845655441],
        'invgpn-1e-1-2': [0.02895859070122242, 0.028751229867339134, 0.02951500006020069, 0.02950734831392765,
                          0.029529279097914696, 0.030279986560344696, 0.04158830642700195, 0.04161111265420914,
                          0.04190617427229881, 0.041664112359285355],
        'invgpn-1e-1-4': [0.028958627954125404, 0.02879471704363823, 0.029150495305657387, 0.029677407816052437,
                          0.029619388282299042, 0.02968589775264263, 0.041663799434900284, 0.04157743975520134,
                          0.041384629905223846, 0.04147062823176384]
    }
    baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
    baseline = np.stack([np.array(data[x]) for x in baselines])
    invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
    inv = np.stack([np.array(data[x]) for x in invs])

    plt.figure(figsize=(4.5, 3.5))

    epochs = np.array(range(100, 1001, 100))
    base_mean, base_se = baseline.mean(0), se(baseline, 0)
    col = '#1f77b4'
    plt.plot(epochs, base_mean, color=col, marker='o', markersize=7, markeredgewidth=0.)
    plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv.mean(0), se(inv, 0)
    col2 = '#ff7f0e'
    plt.plot(epochs, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(epochs, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.hlines(y=0.04083, xmin=100, xmax=1000, color='k', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Test squared error loss')
    plt.legend(['No gradient\npenalty', 'Gradient penalty'], loc='center left', fontsize=12, frameon=False)
    plt.xticks(epochs[1::2])
    plt.show()