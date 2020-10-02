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
For the spirograph dataset, use the following to evaluate on generative parameters regression.
```
$ python3 lbfgs_linear_clf_spirograph.py --load-from output_epoch999.pth 
```
We can evaluate on predicting transformation parameter $\rvalpha$ by using the command
```
$ python3 scan_predict_alpha.py --baselines output_base --ours output_gp
```
this code will scan over a range of epoch of each checkpointfile, the default range is (100,1000).

In addition, we can look at each downstream tasks as in the Figure 3c) in the paper by running
```
$run scan_eval_reg_component.py --baselines output_base --ours output_gp
```
