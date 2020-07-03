import torch
from torch import nn

from PIL import ImageFilter
from torchvision import transforms


class DifferentiableColourDistortion(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_dist = torch.distributions.MultivariateNormal(
            torch.tensor([ 0.98,  0.06,  0.04, -0.03,  0.84, -0.04,  0.05,  0.09,  0.99, -0.00, -0.00, -0.00]),
            covariance_matrix=torch.tensor(
        [[ 0.18074650,  0.00123624, -0.02651235, -0.09860805,  0.09091195, -0.07780737, -0.00920981, -0.01918633,
          0.17576791, -0.00834481, -0.00830104, -0.00805414],
        [ 0.00123624,  0.10395385,  0.01573675, -0.02785348, -0.04892306,  0.04504404,  0.03578304, -0.04570618,
         -0.05145603, -0.00118306, -0.00125381, -0.00119490],
        [-0.02651235,  0.01573675,  0.04726495,  0.03685538, -0.03615827,  0.00513693, -0.00279980,  0.02796142,
         -0.04439921, -0.00095502, -0.00097433, -0.00106152],
        [-0.09860805, -0.02785348,  0.03685538,  0.12118191, -0.00126875,  0.06086478, -0.01757026,  0.03399952,
         -0.09201879, -0.00075510, -0.00075904, -0.00088032],
        [ 0.09091195, -0.04892306, -0.03615827, -0.00126875,  0.13618506, -0.00899528, -0.02416800, -0.02172066,
          0.10960858, -0.00748678, -0.00747979, -0.00730574],
        [-0.07780737,  0.04504404,  0.00513693,  0.06086478, -0.00899528,  0.10825535,  0.02111831, -0.03186969,
         -0.10883092, -0.00062423, -0.00066785, -0.00071586],
        [-0.00920981,  0.03578304, -0.00279980, -0.01757026, -0.02416800,  0.02111831,  0.03327096, -0.00517776,
         -0.01198933, -0.00083387, -0.00084015, -0.00080103],
        [-0.01918633, -0.04570618,  0.02796142,  0.03399952, -0.02172066, -0.03186969, -0.00517776,  0.07695913,
          0.01365563, -0.00124058, -0.00119133, -0.00125144],
        [ 0.17576791, -0.05145603, -0.04439921, -0.09201879,  0.10960858, -0.10883092, -0.01198933,  0.01365563,
          0.22387496, -0.00818232, -0.00812927, -0.00789771],
        [-0.00834481, -0.00118306, -0.00095502, -0.00075510, -0.00748678, -0.00062423, -0.00083387, -0.00124058,
         -0.00818232,  0.00251631,  0.00250656,  0.00247063],
        [-0.00830104, -0.00125381, -0.00097433, -0.00075904, -0.00747979, -0.00066785, -0.00084015, -0.00119133,
         -0.00812927,  0.00250656,  0.00251393,  0.00247548],
        [-0.00805414, -0.00119490, -0.00106152, -0.00088032, -0.00730574, -0.00071586, -0.00080103, -0.00125144,
         -0.00789771,  0.00247063,  0.00247548,  0.00245932]])
        )

    def sample(self, device, shape):
        coef = self.linear_dist.sample(shape).to(device)
        return coef

    def forward(self, x):
        shape = (x.shape[0], 1, 1)
        params = self.sample(x.device, shape)
        return self.main(x, params)

    def main(self, x, coef):
        matrix = coef[..., :9].reshape(coef.shape[:-1] + (3, 3))
        intercept = coef[..., 9:]
        y = matrix.matmul(x.transpose(-1, 1).unsqueeze(-1)).squeeze(-1) + intercept
        y = y.transpose(-1, 1).clamp(min=0., max=1.)
        return y


def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def BlurOrSharpen(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.5)
    return full_transform


class ImageFilterTransform(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


class GaussianBlur(ImageFilterTransform):

    def __init__(self, radius=2.):
        self.filter = ImageFilter.GaussianBlur(radius=radius)


class Sharpen(ImageFilterTransform):

    def __init__(self):
        self.filter = ImageFilter.SHARPEN
