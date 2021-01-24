import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    dataset = 'cifar10'
    if dataset == 'cifar10':
        data = {
            'aam-baseline-1.pth': [(93.14, 0.19921961426734924), (93.09, 0.19942986965179443), (92.94, 0.20143158733844757),
                                   (92.93, 0.20748338103294373), (92.89, 0.21201463043689728), (92.16, 0.22669745981693268),
                                   (91.61, 0.24323879182338715), (90.02, 0.29178616404533386), (86.7, 0.3894282877445221)],
            'aam-baseline-2.pth': [(92.94, 0.20307712256908417), (93.07, 0.20482158660888672), (92.99, 0.20563456416130066),
                                   (92.86, 0.20908500254154205), (92.26, 0.2238365262746811), (92.1, 0.23319609463214874),
                                   (91.63, 0.25010067224502563), (90.08, 0.28537026047706604), (86.53, 0.3937372863292694)],
            'aam-baseline-3.pth': [(92.99, 0.1998269259929657), (92.95, 0.2009839117527008), (93.07, 0.201896071434021),
                                   (92.79, 0.2087673395872116), (92.61, 0.21413397789001465), (92.13, 0.22855685651302338),
                                   (91.15, 0.2502360939979553), (89.92, 0.2890027165412903), (86.62, 0.3932151198387146)],
            'invgpn-1e-1-1': [(93.32, 0.19491587579250336), (93.26, 0.19535113871097565), (93.23, 0.19840554893016815),
                              (93.01, 0.20450791716575623), (92.71, 0.21252326667308807), (92.26, 0.22726796567440033),
                              (91.77, 0.2406376451253891), (90.51, 0.2782331705093384), (87.1, 0.3828711211681366)],
            'invgpn-1e-1-2': [(93.44, 0.18910697102546692), (93.38, 0.19106343388557434), (93.29, 0.19115133583545685),
                              (93.32, 0.19630995392799377), (92.99, 0.19873647391796112), (92.8, 0.20453882217407227),
                              (92.2, 0.22287660837173462), (91.24, 0.2558348774909973), (87.48, 0.3652328848838806)],
            'invgpn-1e-1-4': [(93.33, 0.19323566555976868), (93.33, 0.1926955282688141), (93.29, 0.19446511566638947),
                              (92.97, 0.1977396160364151), (93.01, 0.20649589598178864), (92.8, 0.21055801212787628),
                              (92.4, 0.22229953110218048), (91.14, 0.256561279296875), (87.3, 0.3795110285282135)]
        }
        baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
        invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
        baseline_acc = np.stack([np.array([y[0] for y in data[x][:-1]]) for x in baselines])
        baseline_loss = np.stack([np.array([y[1] for y in data[x][:-1]]) for x in baselines])
        inv_acc = np.stack([np.array([y[0] for y in data[x][:-1]]) for x in invs])
        inv_loss = np.stack([np.array([y[1] for y in data[x][:-1]]) for x in invs])
        params = np.linspace(1e-5, 1., 9)[:-1]

    elif dataset == 'cifar100':
        data = {'aam-1': [(71.07, 1.0187190771102905), (71.2, 1.0203205347061157), (70.98, 1.0249942541122437),
                      (70.69, 1.0383248329162598), (70.52, 1.0528647899627686), (70.16, 1.0644025802612305),
                      (69.34, 1.0923759937286377), (68.39, 1.1360427141189575)],
                 'aam-2': [(71.16, 1.0146222114562988), (70.96, 1.016646146774292), (71.0, 1.0194787979125977),
                      (70.92, 1.0259077548980713), (70.41, 1.0434194803237915), (69.86, 1.0646246671676636),
                      (68.9, 1.0894519090652466), (67.72, 1.138643503189087)],
                  'aam-3': [(71.64, 0.9990115165710449), (71.49, 0.9988921880722046), (71.33, 1.0064200162887573),
                      (71.47, 1.0107405185699463), (71.06, 1.028464436531067), (70.54, 1.0458329916000366),
                      (69.57, 1.07221257686615), (68.69, 1.109005093574524)],
                  'gp1':  [(72.26, 0.970640242099762), (72.27, 0.9719348549842834), (72.46, 0.975175678730011),
                 (71.81, 0.9847431778907776), (71.67, 0.9957839846611023), (70.97, 1.0201725959777832),
                 (70.55, 1.0359083414077759), (69.33, 1.0770869255065918)],
                'gp2': [(72.09, 0.9748682379722595), (72.14, 0.9774567484855652), (71.74, 0.9833812713623047),
                 (71.64, 0.9938498735427856), (71.41, 1.0012744665145874), (71.05, 1.0259361267089844),
                 (70.61, 1.0476417541503906), (69.08, 1.0906877517700195)],
                'gp3': [(72.2, 0.9724085927009583), (72.05, 0.9739083051681519), (71.95, 0.9791049957275391),
                 (71.79, 0.9867172837257385), (71.44, 0.9976739287376404), (71.66, 1.0105066299438477),
                 (70.18, 1.0398199558258057), (68.96, 1.0800659656524658)]
        }
        baselines = ['aam-1', 'aam-2', 'aam-3']
        invs = ['gp1', 'gp2', 'gp3']
        baseline_acc = np.stack([np.array([y[0] for y in data[x]]) for x in baselines])
        baseline_loss = np.stack([np.array([y[1] for y in data[x]]) for x in baselines])
        inv_acc = np.stack([np.array([y[0] for y in data[x]]) for x in invs])
        inv_loss = np.stack([np.array([y[1] for y in data[x]]) for x in invs])
        params = np.linspace(1e-5, .9, 8)
    else:
        raise ValueError

    plt.figure(figsize=(4.5, 3.5))

    base_mean, base_se = baseline_acc.mean(0), se(baseline_acc, 0)
    col = '#1f77b4'
    plt.plot(params, base_mean, color=col, marker='o', markersize=7, markeredgewidth=0.)
    plt.fill_between(params, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv_acc.mean(0), se(inv_acc, 0)
    col2 = '#ff7f0e'
    plt.plot(params, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(params, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.xlabel('Transformation strength $s$')
    plt.ylabel('Test accuracy')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='lower left', fontsize=12, frameon=False)

    plt.show()

    plt.figure(figsize=(4.5, 3.5))

    base_mean, base_se = baseline_loss.mean(0), se(baseline_loss, 0)
    col = '#1f77b4'
    plt.plot(params, base_mean, color=col, marker='o', markersize=7, markeredgewidth=0.)
    plt.fill_between(params, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv_loss.mean(0), se(inv_loss, 0)
    col2 = '#ff7f0e'
    plt.plot(params, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(params, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.xlabel('Transformation strength $s$')
    plt.ylabel('Test loss')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='upper left', fontsize=12, frameon=False)

    plt.show()