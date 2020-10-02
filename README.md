# Reproducing SimCLR in PyTorch

## Introduction
This is a[PyTorch](https://github.com/pytorch/pytorch) implementation of Improving Transformation Invariance in Contrastive Representation Learning  
 
### Data
We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and our a novel Spirograph dataset. This code also supports [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/) datasets and [STL-10](https://cs.stanford.edu/~acoates/stl10/).

### Augmentation
The following augmentations are used on the training set
 - Random crop and resize. We use `RandomResizedCrop` in PyTorch with cubic interpolation for the resizing.
   CIFAR images are resized to 32×32, STL-10 images to 96×96 and ILSVRC2012 images to 224×224.
 - Random horizontal flip
 - Colour distortion. We use the code provided in the paper (Appendix A)
 - Gaussian blur is not yet included
 
### Encoder
We use a ResNet50 as our base architecture. We use the ResNet50 implementation included in `torchvision` with the
following changes:
 - Stem adapted to the dataset, for details see `models/resnet.py`. We adapt the stem for CIFAR in the same way as
   the original paper: replacing the first 7×7 Conv of stride 2 with 3×3 Conv of stride 1, and also removing the 
   first max pooling operation. For STL, we use the 3×3 Conv of stride 1 but include a max pool.
 - We do not make special adjustments to sync the batch norm means and variances across GPU nodes.
 - Remove the final fully connected layer, giving a representation of dimension 2048.
 
### Projection head
The projection head consists of the following:
 - MLP projection with one hidden layer with dimensions 2048 -> 2048 -> 128
 - Following the tensorflow code, we also include batch norm in the projection head
 
### Loss
We use the NT-Xent loss of the original paper. Specifically, we calculate the `CosineSimilarity` (using PyTorch's
implementation) between each of the `2N` projected representations `z`. We rescale these similarities by temperature.
We set the diagonal similarities to `-inf` and treat the one remaining positive example as the correct category in a
`2N`-way classification task. The scores are fed directly into `CrossEntropyLoss`.

### Optimizer
We use the LARS optimizer with `trust_coef=1e-3` to match the tensorflow code. We set the weight decay to `1e-6`.
The 10 epoch linear ramp and cosine annealing of the original paper are implemented and can be activated using
 `--cosine-anneal`, otherwise a constant learning rate is used.

### Evaluation
On CIFAR-10, we fitted the downstream classifier using L-BFGS with no augmentation on the training set. This is the
approach used in the original paper for transfer learning (and is substantially faster for small datasets).
For ImageNet, we use SGD with the same random resized crop and random flip as for the original training, but no
colour distortion or other augmentations. This is as in the original paper.



## Running the code

### Requirements
See `requirements.txt`. Note we require the [torchlars](https://github.com/kakaobrain/torchlars) package.

### Dataset locations
The dataset file locations should be specified in a JSON file of the following form
```
dataset-paths.json
{
    "cifar10": "/data/cifar10/",
    "cifar100": "/data/cifar100/",
    "stl10": "/data/stl10/",
    "imagenet": "/data/imagenet/2012/"
}
```

### Training an encoder
Use the following command to train an encoder from scratch on CIFAR-10
```
$python3 invariance_by_gradient.py --base-lr 1.5 --num-epochs 1000 --cosine-anneal --arch resnet50 --dataset cifar10 --lambda-gp 1e-2 --filename output
```
to train an encoder on Spirograph dataset
```
$ python3 invariance_by_gradient.py --base-lr 1.5 --num-epochs 50 --cosine-anneal --test-freq 0 --save-freq 10 --cut-off 50 --arch resnet18 --dataset spirograph --lambda-gp 1e-2 --filename output --gp-upper-limit 1000
```
we can set `--lambda-gp 0` to train an encoder with no gradient penalty.

### Evaluating an encoder
Use the following command to evaluate the trained encoder on a classification task
```
$ python3 lbfgs_linear_clf.py --load-from output.pth 
```
For the spirograph dataset, use the following to evaluate on generative parameters regression.
```
$ python3 lbfgs_linear_clf_spirograph.py --load-from output.pth 
```



## Outstanding differences with the original paper
 - We do not synchronize the batch norm between multiple GPUs. To use PyTorch's `SyncBatchNorm`, we would need to
   change from using `DataParallel` to `DistributedDataParallel`.
 - We not use Gaussian blur for any datasets, including ILSVRC2012.
 - We are not aware of any other discrepancies with the original work, but any correction is more than welcome and 
   should be suggested by opening an Issue in this repo.



## Reproduction results
### CIFAR-10 with ResNet50 (1000 epochs)
Method | Test accuracy 
--- | ---
SimCLR quoted | 94.0%
SimCLR reproduced (this repo) | 92.7%


## Acknowledgements
The basis for this repository was [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
We make use of [torchlars](https://github.com/kakaobrain/torchlars).
