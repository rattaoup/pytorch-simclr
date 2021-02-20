# Improving Transformation Invariance in Contrastive Representation Learning
## Introduction
This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of Improving Transformation Invariance in Contrastive Representation Learning  

### Data
We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and our a novel Spirograph dataset which is introduced to explore our ideas in the context of a differentiable generative process with multiple downstream tasks. This code also supports [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/) datasets and [STL-10](https://cs.stanford.edu/~acoates/stl10/).


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
$ python3 lbfgs_linear_clf.py --load-from output_epoch999.pth
```
and the regression task on generative parameters for the spirograph dataset
```
$ python3 lbfgs_linear_clf_spirograph.py --load-from output_sg_epoch049.pth
```
### Transformation parameters prediction

We can evaluate on predicting transformation parameter alpha by using the following command
```
$ python3 alpha_prediction.py --load-from output_epoch999.pth
```
that works for all datasets.

### Component-wise regression tasks for Spirograph
In addition, we can look at each downstream tasks as in the Figure 3c) in the paper by running
```
$ python3 lbfgs_linear_clf_spirograph_cp.py --load-from output_sg_epoch049.pth
```
#### Feature averaging
Use the following command to evaluate classification performance of feature averaging where we scan over linspace(min_passes, max_passes, num_passes).
```
$ python3 feature_averaging.py --load-from output_epoch999.pth --min-num-passes 10 --max-num-passes 20	--step-num-passes 2
```
We evaluate the regression performance of feature averaging by running the following code
```
$ python3 scan_eval_reg_fa.py --baselines output_base --ours output_gp --min-num-passes 10 --max-num-passes --step-num-passes 3
```
this code will scan over a range of epoch of each checkpointfile, the default range is (100,1000).

#### Robustness
For spirograph robustness evaluation, we can run `lbfgs_linear_clf_spirograph.py` with additional command. For example, if we want to shift the distribution of background colour by s for each s in linspace(-0.5, 0.5, 6) we can use the following command
```
$ python3 lbfgs_linear_clf_spirograph.py --load-from sg-gp-0_run1_epoch049.pth --back-var-lower -0.5 --back-var-upper 0.5 --back-var-num-passes 6
```
available parameter includes mean and variance of h, background and foreground colour.
