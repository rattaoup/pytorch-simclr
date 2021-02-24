# InvCLR: Improving Transformation Invariance in Contrastive Representation Learning
## Introduction
This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of the ICLR paper [Improving Transformation Invariance in Contrastive Representation Learning (InvCLR)](https://arxiv.org/abs/2010.09515):
```
@article{foster2020improving,
  title={Improving Transformation Invariance in Contrastive Representation Learning},
  author={Foster, Adam and Pukdee, Rattana and Rainforth, Tom},
  journal={arXiv preprint arXiv:2010.09515},
  year={2020}
}
```

## Installation and datasets
Install PyTorch following the instructions [here](https://pytorch.org/). Download the the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.
The Spirograph dataset is included in this code base.
To use the Spirograph dataset in an unrelated project, see this [standalone repo](https://github.com/rattaoup/spirograph).

## Training an encoder
We support multi-GPU `DataParallel` training.
Use the following command to train an encoder from scratch on CIFAR-10
```bash
$ python3 invariance_by_gradient.py \
  --base-lr 1.5 \
  --num-epochs 1000 \
  --cosine-anneal \
  --arch resnet50 \
  --dataset cifar10 \
  --lambda-gp 1e-2 \
  --filename output
```
with a similar command for `cifar100`.
To train an encoder on the Spirograph dataset, use
```bash
$ python3 invariance_by_gradient.py \
  --base-lr 1.5 \
  --num-epochs 50 \
  --cosine-anneal \
  --test-freq 0 \
  --save-freq 10 \
  --cut-off 50 \
  --arch resnet18 \
  --dataset spirograph \
  --lambda-gp 1e-2 \
  --filename output \
  --gp-upper-limit 1000
```
You can set `--lambda-gp 0` to train an encoder with no gradient penalty.

## Evaluating an encoder
Use the following command to evaluate the trained encoder on a classification task
```
$ python3 lbfgs_linear_clf.py --load-from output_epoch999.pth
```
and the regression task on generative parameters for the spirograph dataset
```
$ python3 lbfgs_linear_clf_spirograph.py --load-from output_sg_epoch049.pth
```
### Component-wise regression tasks for Spirograph
In addition, we can look at each downstream tasks as in the Figure 3c) in the paper by running
```
$ python3 lbfgs_linear_clf_spirograph_cp.py --load-from output_sg_epoch049.pth
```
### Feature averaging
Use the following command to evaluate classification performance of feature averaging.
```
$ python3 feature_averaging.py --load-from output_epoch999.pth --min-num-passes 10 --max-num-passes 20	--step-num-passes 2
```
for  Spirograph, run the following code
```
$ python3 feature_averaging_spirograph.py --load-from output_sg_epoch049.pth --min-num-passes 10 --max-num-passes 20 --step-num-passes 2
```


TODO: include brief note on expected performance on CIFAR-10 CIFAR-100
