import matplotlib.pyplot as plt
import numpy as np


def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])


if __name__ == '__main__':
    dataset = 'cifar100'
    if dataset == 'cifar10':
        data = {'aam-baseline-1.pth': [0.5222747921943665, 0.5687380433082581, 0.599071204662323, 0.6186872124671936,
                                       0.6597009301185608, 0.6647329926490784, 0.6905269622802734, 0.6950846910476685,
                                       0.696789562702179, 0.69144207239151],
                'aam-baseline-2.pth': [0.5265793800354004, 0.5665996074676514, 0.5878450870513916, 0.6193771362304688,
                                       0.6423370838165283, 0.6661521196365356, 0.6950228214263916, 0.7105359435081482,
                                       0.701993465423584, 0.6994907259941101],
                'aam-baseline-3.pth': [0.5326544046401978, 0.5703436136245728, 0.5899834036827087, 0.623084545135498,
                                       0.6333988308906555, 0.6716654300689697, 0.6828789114952087, 0.6886543035507202,
                                       0.6939307451248169, 0.6870090365409851],
                'invgpn-1e-1-1': [0.03685380145907402, 0.026371929794549942, 0.01984754577279091, 0.01411467231810093,
                                  0.010772784240543842, 0.007608495187014341, 0.005808610934764147, 0.004754021298140287,
                                  0.004194346722215414, 0.004133099224418402],
                'invgpn-1e-1-2': [0.0405542366206646, 0.027279173955321312, 0.01815769635140896, 0.013478664681315422,
                                  0.010221606120467186, 0.007686822209507227, 0.005794198252260685,
                                  0.0047878362238407135, 0.004235611762851477, 0.004159882199019194],
                'invgpn-1e-1-4': [0.03755423054099083, 0.02606799826025963, 0.017826175317168236, 0.013781581073999405,
                                  0.010779437609016895, 0.007888314314186573, 0.006086194422096014, 0.004768269136548042,
                                  0.004205300472676754, 0.004163868725299835]}

        baselines = ['aam-baseline-1.pth', 'aam-baseline-2.pth', 'aam-baseline-3.pth']
        invs = ['invgpn-1e-1-1', 'invgpn-1e-1-2', 'invgpn-1e-1-4']
    elif dataset == 'cifar100':

        data = {'aam-cifar100-1': [0.5189045071601868, 0.544103741645813, 0.5717862844467163, 0.6082149744033813,
                                   0.622480571269989, 0.6570854187011719, 0.6890130639076233, 0.6866220235824585,
                                   0.6920819282531738, 0.6804587841033936],
                'aam-cifar100-2': [0.5266953110694885, 0.5575991868972778, 0.558264434337616, 0.5950544476509094,
                                   0.626933217048645, 0.6605053544044495, 0.6684527397155762, 0.6750876307487488,
                                   0.6790212988853455, 0.6681933999061584],
                'aam-cifar100-3': [0.5030343532562256, 0.5266659259796143, 0.5579777956008911, 0.5745605230331421,
                                   0.6025350689888, 0.640057384967804, 0.6377320885658264, 0.6600807309150696,
                                   0.6305169463157654, 0.6285222768783569],
                'invgpn-cifar100-1e-1-1': [0.032917022705078125, 0.02532925456762314, 0.018341146409511566,
                                           0.013414815999567509, 0.010561604052782059, 0.007618218194693327,
                                           0.005861249286681414, 0.004877534229308367, 0.0041760848835110664,
                                           0.004234471824020147],
                'invgpn-cifar100-1e-1-2': [0.03051426261663437, 0.02587311901152134, 0.018510356545448303,
                                           0.01331283524632454, 0.010870205238461494, 0.007882832549512386,
                                           0.0057779536582529545, 0.004726470913738012, 0.004234625957906246,
                                           0.004188129212707281],
                'invgpn-cifar100-1e-1-3': [0.03326355665922165, 0.022807210683822632, 0.016274694353342056,
                                           0.012832744978368282, 0.010492820292711258, 0.00774725154042244,
                                           0.00584620563313365, 0.004763239063322544, 0.004266665317118168,
                                           0.004178598523139954]}
        baselines = ['aam-cifar100-1', 'aam-cifar100-2', 'aam-cifar100-3']
        invs = ['invgpn-cifar100-1e-1-1', 'invgpn-cifar100-1e-1-2', 'invgpn-cifar100-1e-1-3']
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
        # data = {'aam-baseline-1.pth': [6.9421, 7.3924, 7.0669, 7.3108, 7.3018, 6.0960, 6.0507, 5.9493, 6.0909, 6.2383],
        #  'aam-baseline-2.pth': [7.9082, 7.5181, 7.4649, 6.6028, 6.2907, 5.9757, 5.9316, 5.9842, 5.9969, 6.2311],
        #  'aam-baseline-3.pth': [7.0199, 7.8414, 6.9805, 6.9545, 7.2471, 6.1085, 6.0561, 6.1005, 6.2059, 6.1712],
        #  'invgpn-1e-1-1': [1.2956, 1.2907, 1.3168, 1.2430, 1.1442, 1.1071, 1.0785, 1.0752, 1.0788, 1.0848],
        #  'invgpn-1e-1-2': [1.2576, 1.3683, 1.1904, 1.1973, 1.1432, 1.1063, 1.0802, 1.0819, 1.0904, 1.0832],
        #  'invgpn-1e-1-4': [1.2231, 1.3864, 1.3539, 1.2192, 1.1478, 1.1128, 1.0843, 1.0725, 1.0836, 1.0851]}
    else:
        raise ValueError


    baseline = np.stack([np.stack(data[x]) for x in baselines])

    inv = np.stack([np.array(data[x]) for x in invs])

    plt.figure(figsize=(4., 3.5))

    epochs = np.array(range(100, 1001, 100))
    base_mean, base_se = baseline.mean(0), se(baseline, 0)
    #col = '#2ba0c4'
    col = '#1f77b4'
    plt.plot(epochs, base_mean, color=col, marker='o', markersize=7, markeredgewidth=0.)
    plt.fill_between(epochs, base_mean + base_se, base_mean - base_se, alpha=0.15, color=col)

    inv_mean, inv_se = inv.mean(0), se(inv, 0)
    col2 = '#ff7f0e'
    plt.plot(epochs, inv_mean, color=col2, marker='x', markersize=7)
    plt.fill_between(epochs, inv_mean + inv_se, inv_mean - inv_se, alpha=0.15, color=col2)

    plt.xlabel('Epoch')
    plt.ylabel('Conditional variance')
    plt.legend(['No gradient penalty', 'Gradient penalty'], loc='center right', fontsize=12, frameon=False)
    plt.xticks(epochs[1::2])
    plt.show()