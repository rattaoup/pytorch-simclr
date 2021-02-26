# InvCLR: Improving Transformation Invariance in Contrastive Representation Learning
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
To use the Spirograph dataset on its own, see this [standalone repo](https://github.com/rattaoup/spirograph).
To install the requisite packages for this project, use `pip install -r requirements.txt`.
Note: to install `torchlars` it is necessary to set the environment variable `CUDA_HOME`

## Training an encoder
We support multi-GPU `DataParallel` training.
Use the following command to train an encoder from scratch on CIFAR-10. We ran this using 8 GPUs.
```bash
$ python3 invclr.py \
  --num-epochs 1000 \
  --cosine-anneal \
  --arch resnet50 \
  --dataset cifar10 \
  --lambda-gp 1e-1 \
  --filename cifar10_run
```
Set `--dataset cifar100` to train on CIFAR-100.
To train an encoder on the Spirograph dataset, use
```bash
$ python3 invclr.py \
  --num-epochs 50 \
  --cosine-anneal \
  --test-freq 0 \
  --save-freq 10 \
  --arch resnet18 \
  --dataset spirograph \
  --lambda-gp 1e-2 \
  --filename spirograph_run \
  --gp-upper-limit 1000
```
You can set `--lambda-gp 0` to train an encoder with no gradient penalty.

## Evaluating an encoder
Use the following command to evaluate the trained CIFAR-10 encoder on untransformed inputs with a fraction 50% of
the labels used at training
```bash
$ python3 eval.py \
  --load-from cifar10_run_epoch999.pth \
  --untransformed \
  --proportion 0.5
```
for Spirograph, we used
```bash
$ python3 eval.py \
  --load-from spirograph_run_epoch049.pth \
  --reg-weight 1e-8 \
  --proportion 0.5
```

### Component-wise regression tasks for Spirograph
TODO: add this to the `eval` script

In addition, we can look at each downstream tasks as in the Figure 3c) in the paper by running
```
$ python3 lbfgs_linear_clf_spirograph_cp.py --load-from output_sg_epoch049.pth
```
### Feature averaging
Use the following command to evaluate classification performance of feature averaging using an average of 100 samples
```bash
$ python3 eval.py \
  --load-from cifar10_run_epoch999.pth \
  --num-passes 100 \
```
for  Spirograph, run the following code
```bash
$ python3 eval.py \
  --load-from spirograph_run_epoch049.pth \
  --num-passes 100 \
  --reg-weight 1e-8
```
We obtained the following
| Dataset    | Loss | Accuracy |
|------------|------|----------|
| CIFAR-10   |      |          |
| CIFAR-100  |      |          |
| Spirograph |      |          |




TODO: Merge `feature_averaging.py` and `feature_averaging_spirograph.py` (done)

TODO: refactor and test `invclr.py` on all datasets. Top priority!

TODO: test `feature_averaging.py` works on spirograph and on cifar

TODO: Merge `lbfgs_linear_clf_spirograph.py` and `lbfgs_linear_clf_spirograph_cp.py` and make this support classification with untransformed inputs for CIFAR

TODO (after reproducing): include brief note on expected performance on CIFAR-10 CIFAR-100 Spirograph


