import json

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from collections import defaultdict

from augmentation import ColourDistortion, TensorNormalise, ModuleCompose, DrawSpirograph
from dataset import *
from models import *


def get_mean_std(dataset):
    CACHED_MEAN_STD = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }
    return CACHED_MEAN_STD[dataset]


def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=None,
                 augment_test=False, train_proportion=1.):
    if dataset == 'spirograph':
        return get_spirograph_dataset()
    else:
        return get_img_datasets(dataset=dataset, augment_clf_train=augment_clf_train,
                                add_indices_to_data=add_indices_to_data, num_positive=num_positive,
                                augment_test=augment_test, train_proportion=train_proportion)


def get_spirograph_dataset(augment_clf_train=False, add_indices_to_data=False, num_positive=None,
                           augment_test=False, train_proportion=1., rgb_fore_bounds = (.4, 1), rgb_back_bounds=(0, .6)):

    spirograph = DrawSpirograph(['m', 'b', 'sigma', 'rfore'], ['h', 'rback', 'gfore', 'gback', 'bfore', 'bback'],
                                rgb_fore_bounds= rgb_fore_bounds, rgb_back_bounds=rgb_back_bounds)
    stem = StemCIFAR
    trainset, clftrainset, testset = spirograph.dataset()
    num_classes = 3
    return trainset, testset, clftrainset, num_classes, stem, spirograph, spirograph


def get_root(dataset):
    PATHS = {
        'cifar10': '/data/cifar10/',
        'cifar100': '/data/cifar100/',
        'stl10': '/data/stl10/',
        'imagenet': '/data/imagenet/2012/'
    }
    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]
    return root


def get_img_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=None,
                     augment_test=False, train_proportion=1.):

    root = get_root(dataset)

    # Data
    if dataset == 'stl10':
        img_size = 96
    elif dataset == 'imagenet':
        img_size = 224
    else:
        img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ColourDistortion(s=0.5),
        # transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if augment_test:
        transform_test = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(*get_mean_std(dataset)),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_mean_std(dataset)),
        ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(*get_mean_std(dataset)),
        ])
    else:
        transform_clftrain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_mean_std(dataset)),
        ])

        return get_datasets_from_transform(dataset, root, transform_train, transform_test, transform_clftrain,
                                           add_indices_to_data=add_indices_to_data, num_positive=num_positive,
                                           train_proportion=train_proportion)


def get_datasets_from_transform(dataset, root, transform_train, transform_test, transform_clftrain,
                                add_indices_to_data=False, num_positive=None, train_proportion=1.):

    if dataset == 'cifar100':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR100)
        else:
            dset = torchvision.datasets.CIFAR100
        if num_positive is None:
            trainset = CIFAR100Biaugment(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = CIFAR100Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                            n_augmentations=num_positive)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 100
        stem = StemCIFAR
    elif dataset == 'cifar10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10
        if num_positive is None:
            trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = CIFAR10Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                           n_augmentations=num_positive)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == 'stl10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STl10
        if num_positive is None:
            trainset = STL10Biaugment(root=root, split='unlabeled', download=True, transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=root, split='train', download=True, transform=transform_test)
        clftrainset = dset(root=root, split='test', download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemSTL
    elif dataset == 'imagenet':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.ImageNet)
        else:
            dset = torchvision.datasets.ImageNet
        if num_positive is None:
            trainset = ImageNetBiaugment(root=root, split='train', transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=root, split='val', transform=transform_test)
        clftrainset = dset(root=root, split='train', transform=transform_clftrain)
        num_classes = len(testset.classes)
        stem = StemImageNet
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    if train_proportion < 1.:
        trainset = make_stratified_subset(trainset, train_proportion)
        clftrainset = make_stratified_subset(clftrainset, train_proportion)

    col_distort = ColourDistortion(s=0.5)
    batch_transform = ModuleCompose([
        col_distort,
        TensorNormalise(*get_mean_std(dataset))
    ])

    return trainset, testset, clftrainset, num_classes, stem, col_distort, batch_transform


def make_stratified_subset(trainset, train_proportion):

    target_n_per_task = int(len(trainset) * train_proportion / len(trainset.classes))
    target_length = target_n_per_task * len(trainset.classes)
    indices = []
    counts = defaultdict(lambda: 0)
    for i in torch.randperm(len(trainset)):
        y = trainset.targets[i]
        if counts[y] < target_n_per_task:
            indices.append(i)
            counts[y] += 1
        if len(indices) >= target_length:
            break
    return Subset(trainset, indices)
