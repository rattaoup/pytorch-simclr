import numbers
from PIL import ImageFilter
from torchvision import transforms
import torch
from torch import nn
from math import pi


def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = TensorColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = TensorRandomApply([color_jitter], p=0.8)
    rnd_gray = TensorRandomApply([TensorGrayscale()], p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def BlurOrSharpen(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.5)
    return full_transform


def grayscale(img):
    if len(img.shape) == 3:
        output = 0.2990 * img[0, ...] + 0.5870 * img[1, ...] + 0.1140 * img[2, ...]
        return output.unsqueeze(0)
    elif len(img.shape) == 4:
        output = 0.2990 * img[:, 0, ...] + 0.5870 * img[:, 1, ...] + 0.1140 * img[:, 2, ...]
        return output.unsqueeze(1)
    else:
        raise ValueError("Unexpected number of dimensions for grayscale conversion")


def rmv(matrix, vector):
    return matrix.matmul(vector.unsqueeze(-1)).squeeze(-1)


def adjust_brightness(img, scale):
    y = img * scale
    return y.clamp(min=0, max=1)


def adjust_saturation(img, scale):
    y = img * scale + grayscale(img) * (1 - scale)
    return y.clamp(min=0, max=1)


def adjust_contrast(img, scale):
    y = img * scale + grayscale(img).mean(dim=[-1, -2], keepdim=True) * (1 - scale)
    return y.clamp(min=0, max=1)


def adjust_hue(img, scale):

    scale = 2 * pi * torch.as_tensor(scale, device=img.device)

    T_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]], device=img.device)
    T_rgb = torch.tensor([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]], device=img.device)

    if len(img.shape) == 3:
        T_hue = torch.tensor([1, 0, 0,
                              0, torch.cos(scale), -torch.sin(scale),
                              0, -torch.sin(scale), torch.cos(scale)], device=img.device).reshape(3, 3)
        T_final = torch.matmul(torch.matmul(T_rgb, T_hue), T_yiq)
        return rmv(T_final, img.transpose(0, -1)).transpose(-1, 0)

    elif len(img.shape) == 4:
        B = img.shape[0]
        T_hue = torch.stack([torch.ones(B, device=scale.device), torch.zeros(B, device=scale.device), torch.zeros(B, device=scale.device),
                             torch.zeros(B, device=scale.device), torch.cos(scale), -torch.sin(scale),
                             torch.zeros(B, device=scale.device), -torch.sin(scale), torch.cos(scale)], dim=-1).reshape(B, 3, 3)
        T_final = torch.matmul(torch.matmul(T_rgb, T_hue), T_yiq).unsqueeze(1).unsqueeze(1)
        return rmv(T_final, img.transpose(1, -1)).transpose(-1, 1)

    else:
        raise ValueError("Unexpected number of dimensions for hue adjustment")


class TensorRandomApply(nn.Module):

    def __init__(self, transform_list, p=0.5):
        super().__init__()
        self.transform = transforms.Compose(transform_list)
        self.p = p

    def forward(self, x):
        if len(x.shape) == 3:
            if torch.rand() < self.p:
                return self.transform(x)
            else:
                return x
        elif len(x.shape) == 4:
            B = x.shape[0]
            y = self.transform(x)
            apply_indicator = torch.bernoulli(self.p * torch.ones(B))
            return y * apply_indicator + (1 - apply_indicator) * x


class TensorGrayscale(nn.Module):

    def forward(self, x):
        shape = x.shape
        return grayscale(x).expand(shape)


class TensorColorJitter(nn.Module):

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        if len(img.shape) == 3:
            if self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)

        elif len(img.shape) == 4:
            B = img.shape[0]
            if self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.ones(B, 1, 1, 1).uniform_(brightness[0], brightness[1])
                img = adjust_brightness(img, brightness_factor)

            if self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.ones(B, 1, 1, 1).uniform_(contrast[0], contrast[1])
                img = adjust_contrast(img, contrast_factor)

            if self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.ones(B, 1, 1, 1).uniform_(saturation[0], saturation[1])
                img = adjust_saturation(img, saturation_factor)

            if self.hue is not None:
                hue = self.hue
                hue_factor = torch.ones(B, 1, 1, 1).uniform_(hue[0], hue[1])
                img = adjust_hue(img, hue_factor)

        return img


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
