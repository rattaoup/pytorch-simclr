import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    dataset = 'cifar100'
    if dataset == 'cifar10':
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
        invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
    elif dataset == 'cifar100':
        data = {
            'aam-cifar100-1': [0.02916673570871353, 0.029527196660637856, 0.029921937733888626, 0.030482295900583267,
                            0.030714601278305054, 0.03162180259823799, 0.0327644944190979, 0.034309227019548416,
                            0.03571349009871483, 0.03619527071714401],
         'aam-cifar100-2': [0.02950606681406498, 0.02979712374508381, 0.029878517612814903, 0.03037528693675995,
                            0.031002849340438843, 0.032608773559331894, 0.03334202617406845, 0.03530240058898926,
                            0.03622185438871384, 0.036089811474084854],
         'aam-cifar100-3': [0.02942727878689766, 0.029479006305336952, 0.029664967209100723, 0.029968803748488426,
                            0.03020596317946911, 0.03192892298102379, 0.03229393810033798, 0.03338206559419632,
                            0.03453229367733002, 0.034868620336055756],
         'invgpn-cifar100-1e-1-1': [0.02854905277490616, 0.029010863974690437, 0.028957854956388474,
                                    0.029180070385336876, 0.02922966144979, 0.029484417289495468, 0.0419076606631279,
                                    0.04209518805146217, 0.04239267483353615, 0.04182949289679527],
         'invgpn-cifar100-1e-1-2': [0.028616808354854584, 0.028902586549520493, 0.02931550145149231,
                                    0.028793876990675926, 0.029554763808846474, 0.02984314225614071,
                                    0.041957102715969086, 0.04197246581315994, 0.04200359433889389,
                                    0.04155195876955986],
         'invgpn-cifar100-1e-1-3': [0.029155461117625237, 0.028519075363874435, 0.0286313034594059,
                                    0.029758000746369362, 0.02872277982532978, 0.029967468231916428,
                                    0.04186294600367546, 0.04166246950626373, 0.041909728199243546,
                                    0.041758887469768524]}
        baselines = ['aam-cifar100-1', 'aam-cifar100-2', 'aam-cifar100-3']
        invs = ['invgpn-cifar100-1e-1-1', 'invgpn-cifar100-1e-1-2', 'invgpn-cifar100-1e-1-3']
    else:
        raise ValueError
    baseline = np.stack([np.array(data[x]) for x in baselines])

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