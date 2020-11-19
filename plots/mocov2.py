import numpy as np

def se(array, axis=0):
    return array.std(axis=axis) / np.sqrt(array.shape[axis])

if __name__ == '__main__':
    resnet18_baseline = np.array([(53.03, 1.841), (51.710, 1.888), (52.330, 1.878)])
    print('Resnet18 baseline')
    print(resnet18_baseline.mean(0), se(resnet18_baseline, axis=0))

    print('Resnet18 gradient')
    resnet18_gp = np.array([(54.190, 1.692), (54.040, 1.696), (53.960, 1.712)])
    print(resnet18_gp.mean(0), se(resnet18_gp))

    print('Resnet18 gradient and feature averaging')
    resnet18_all = np.array([(60.53, 1.4116665124893188), (60.87, 1.4029300212860107), (60.37, 1.4243007898330688)])
    print(resnet18_all.mean(0), se(resnet18_all))

    resnet50_baseline = np.array([(57.830, 2.051), (58.120, 2.032)])
    print('Resnet50 baseline')
    print(resnet50_baseline.mean(0), se(resnet50_baseline, axis=0))

    print('Resnet50 gradient')
    resnet50_gp = np.array([(58.650, 1.876),(59.230, 1.850)])
    print(resnet50_gp.mean(0), se(resnet50_gp))

    print('Resnet50 gradient and feature averaging')
    resnet50_all = np.array([(64.14, 1.4295225143432617), (64.67, 1.4164313077926636)])
    print(resnet50_all.mean(0), se(resnet50_all))
