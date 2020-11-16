import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import math
import os
import argparse
from tqdm import tqdm

from augmentation import ColourDistortion, TensorNormalise, ModuleCompose
from models import *
from configs import get_datasets, get_root, get_mean_std, get_datasets_from_transform
from evaluate import train_clf, test_matrix

import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser(description='Tune regularization coefficient of downstream classifier.')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--load-from", type=str, default='ckpt.pth', help='File to load from')
parser.add_argument("--max-num-passes", type=int, default=20, help='Max number of passes to average')
parser.add_argument("--min-num-passes", type=int, default=10, help='Min number of passes to average')
parser.add_argument("--step-num-passes", type=int, default=2, help='Number of distinct M values to try')
parser.add_argument("--reg-weight", type=float, default=1e-5, help='Regularization parameter')
parser.add_argument("--proportion", type=float, default=1., help='Proportion of train data to use')
parser.add_argument("--no-crop", action='store_true', help="Don't use crops, only feature averaging with colour "
                                                           "distortion")
args = parser.parse_args()

#MoCo
args.data = './datasets/fastai/imagenette2' # change this for imagenet directory
args.pretrained = './moco/moco_v2_800ep_pretrain.pth.tar' #facebook pretrained file
args.arch = 'resnet50'
device = 'cuda'



# create model
print("=> creating model '{}'".format(args.arch))
model = models.__dict__[args.arch]()

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

# load from pre-trained, before DistributedDataParallel constructor
if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

# remove fc layer from the model
net = torch.nn.Sequential(*(list(model.children())[:-1]))
print("=> remove the final fc layer from the model" )


if device == 'cuda':
#     repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
#     net.representation_dim = repr_dim
    cudnn.benchmark = True

# Data loading code - args.data is the location of imagenet folder
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

# Imagenet normalise
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

clftrainset = datasets.ImageFolder(traindir, transform = transform_test)
testset = datasets.ImageFolder(valdir, transform = transform_test)

# Batch transformation
col_distort = ColourDistortion(s=0.5)
batch_transform = ModuleCompose([
    col_distort,
    TensorNormalise(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])


clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=256, shuffle=True, num_workers=args.num_workers,
                                             pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=True)


def encode_feature_averaging(clftrainloader, device, net, target=None, num_passes=10):
    if target is None:
        target = device

    net.eval()

    X, y = [], None
    with torch.no_grad():
        for i in tqdm(range(num_passes)):
            store = []
            for batch_idx, (inputs, targets) in enumerate(clftrainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                rn = col_distort.sample_random_numbers(inputs.shape, inputs.device)
                inputs = batch_transform(inputs, rn)
                representation = net(inputs)
                representation, targets = representation.to(target), targets.to(target)
                store.append((representation, targets))

            Xi, y = zip(*store)
            Xi, y = torch.cat(Xi, dim=0), torch.cat(y, dim=0)
            X.append(Xi)

    X = torch.stack(X, dim=0)

    return X, y


results = []
X, y = encode_feature_averaging(clftrainloader, device, net, num_passes=args.max_num_passes, target='cpu')
X_test, y_test = encode_feature_averaging(testloader, device, net, num_passes=args.max_num_passes, target='cpu')
for m in torch.linspace(args.min_num_passes, args.max_num_passes, args.step_num_passes):
    m = int(m)
    print("FA with M =", m)
    X_this = X[:m, ...].mean(0)
    X_test_this = X_test[:m, ...].mean(0)
    clf = train_clf(X_this.to(device), y.to(device), net.representation_dim, num_classes, device, reg_weight=args.reg_weight)
    acc, loss = test_matrix(X_test_this.to(device), y_test.to(device), clf)
    results.append((acc,loss))
print(results)
