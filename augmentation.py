import torch
from torch import nn
import numbers
import random
from torch import Tensor

from PIL import ImageFilter
from torchvision import transforms


def gen_lambda(B, brightness_bound, contrast_bound, saturation_bound, hue_bound):
    brightness_list = (torch.torch.rand(B)*(brightness_bound[1] - brightness_bound[0])
                   + brightness_bound[0]).requires_grad_(True)
    saturation_list = (torch.torch.rand(B)*(saturation_bound[1] - saturation_bound[0])
                       + saturation_bound[0]).requires_grad_(True)
    contrast_list   = (torch.torch.rand(B)*(contrast_bound[1] - contrast_bound[0])
                       + contrast_bound[0]).requires_grad_(True)
    hue_list        = ((torch.torch.rand(B)*(hue_bound[1]-hue_bound[0])
                       + hue_bound[0])* 3.1415* 2).requires_grad_(True)
    return brightness_list, saturation_list, contrast_list, hue_list


class DifferentiableColourDistortionByTorch_manual(nn.Module):

    '''
    need to input the parameter of color augmentation
    '''

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue


    #### batch color augmentation forward #####

    def forward(self, x):
        '''
        Args:
            x: Input tensor batchsize x 32 x 32

        Returns:
            x_aug : color jittered image

        apply color jitter with prob 0.8
        apply random grayscale with prob 0.2

        '''
        batch_size = x.size()[0]

        p_jitter = torch.ones(batch_size) * 0.8
        jitter = torch.bernoulli(p_jitter)
        jitter = jitter.reshape(batch_size,1,1,1)

        p_gray = torch.ones(batch_size)* 0.2
        gray = torch.bernoulli(p_gray)
        gray = gray.reshape(batch_size,1,1,1)

        jitter = jitter.to(x.device)
        gray = gray.to(x.device)

        #random color jitter
        x_jitter = self.batch_colourjitter(x)
        x = x_jitter * jitter + x *(1-jitter)

        #random gray scale
        x_gray = self.batch_rgb_to_grayscale(x).unsqueeze(1)
        x = x_gray * gray + x* (1-gray)

        return x


    def batch_colourjitter(self, img: Tensor) -> Tensor:

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness_list = self.brightness
                img = self.batch_adjust_brightness(img, brightness_list)

            if fn_id == 1 and self.contrast is not None:
                contrast_list = self.contrast
                img = self.batch_adjust_contrast(img, contrast_list)

            if fn_id == 2 and self.saturation is not None:
                saturation_list = self.saturation
                img = self.batch_adjust_saturation(img, saturation_list)

            if fn_id == 3 and self.hue is not None:
                hue_list = self.hue
                img = self.batch_adjust_hue(img, hue_list)

        return img


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

    ##### function from pytorch source code #####
    def _is_tensor_a_torch_image(self, x: Tensor) -> bool:
        return x.ndim >= 2

    def _blend(self, img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
        bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
        return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)
    def batch_adjust_brightness(self,img: Tensor, brightness_list: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        brightness_factor = brightness_list.reshape(B,1,1,1).to(img.device)
        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def batch_rgb_to_grayscale(self,img: Tensor) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        if img.shape[1] != 3:
            raise TypeError('Input Image does not contain 3 Channels')

        img_tp = img.transpose(0,1)
        img_gray = (0.2989 * img_tp[0] + 0.5870 * img_tp[1] + 0.1140 * img_tp[2])
        return img_gray

    def batch_adjust_saturation(self,img: Tensor, saturation_list: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''

        B = img.size()[0]
        saturation_factor = saturation_list.reshape(B,1,1,1).to(img.device)

        return self._blend(img, self.batch_rgb_to_grayscale(img).unsqueeze(1), saturation_factor)

    def batch_adjust_contrast(self,img: Tensor, contrast_list: float) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        contrast_list = contrast_list.reshape(B,1,1,1).to(img.device)

        #mean for each pic (over HxW points)
        img_gray = self.batch_rgb_to_grayscale(img)
        mean = torch.mean(img_gray.reshape(img_gray.shape[0],-1), dim = 1)


        return self._blend(img, mean.reshape([mean.size()[0],1,1,1]),  contrast_list)

    def batch_adjust_hue(self, img: Tensor, hue_list: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''

        B = img.size()[0]

        # generate tensor
        one_tensor = torch.ones(B)
        zero_tensor = torch.zeros(B)
        cos_tensor = torch.cos(hue_list)
        sin_tensor = torch.sin(hue_list)

        #stack
        T_hue = torch.stack([one_tensor, zero_tensor, zero_tensor,
                     zero_tensor, cos_tensor, -sin_tensor,
                     zero_tensor, sin_tensor, cos_tensor]).transpose(0,1).reshape(B,3,3)

        T_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
        T_rgb = torch.tensor([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]])
        T_final = torch.matmul(torch.matmul(T_rgb, T_hue), T_yiq)
        T_final = T_final.to(img.device)

        #return T_rgb x T_hue x T_yiq x img

        return torch.matmul(T_final.unsqueeze(1).unsqueeze(1), img.transpose(1,-1).unsqueeze(-1)).squeeze(-1).transpose(1,-1)



class DifferentiableColourDistortionByTorch3(nn.Module):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
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

    #### batch color augmentation forward #####

    def forward(self, x):
        '''
        Args:
            x: Input tensor batchsize x 32 x 32

        Returns:
            x_aug : color jittered image

        apply color jitter with prob 0.8
        apply random grayscale with prob 0.2

        '''
        batch_size = x.size()[0]

        p_jitter = torch.ones(batch_size) * 0.8
        jitter = torch.bernoulli(p_jitter)
        jitter = jitter.reshape(batch_size,1,1,1)

        p_gray = torch.ones(batch_size)* 0.2
        gray = torch.bernoulli(p_gray)
        gray = gray.reshape(batch_size,1,1,1)

        jitter = jitter.to(x.device)
        gray = gray.to(x.device)

        #random color jitter
        x_jitter = self.batch_colourjitter(x)
        x = x_jitter * jitter + x *(1-jitter)

        #random gray scale
        x_gray = self.batch_rgb_to_grayscale(x).unsqueeze(1)
        x = x_gray * gray + x* (1-gray)

        return x


    def batch_colourjitter(self, img: Tensor) -> Tensor:

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness_bound = self.brightness
                img = self.batch_adjust_brightness(img, brightness_bound)

            if fn_id == 1 and self.contrast is not None:
                contrast_bound = self.contrast
                img = self.batch_adjust_contrast(img, contrast_bound)

            if fn_id == 2 and self.saturation is not None:
                saturation_bound = self.saturation
                img = self.batch_adjust_saturation(img, saturation_bound)

            if fn_id == 3 and self.hue is not None:
                hue_bound = self.hue
                img = self.batch_adjust_hue(img, hue_bound)

        return img


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

    ##### function from pytorch source code #####
    def _is_tensor_a_torch_image(self, x: Tensor) -> bool:
        return x.ndim >= 2

    def _blend(self, img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
        bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
        return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)

    # batch adjust brightness
    def batch_adjust_brightness(self,img: Tensor, brightness_bound: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        brightness_list = torch.torch.rand(B)*(brightness_bound[1] - brightness_bound[0]) + brightness_bound[0]
        brightness_list.requires_grad_(True)
        brightness_factor = brightness_list.reshape(B,1,1,1).to(img.device)
        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def batch_rgb_to_grayscale(self,img: Tensor) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        if img.shape[1] != 3:
            raise TypeError('Input Image does not contain 3 Channels')

        img_tp = img.transpose(0,1)
        img_gray = (0.2989 * img_tp[0] + 0.5870 * img_tp[1] + 0.1140 * img_tp[2])
        return img_gray

    def batch_adjust_saturation(self,img: Tensor, saturation_bound: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''

        B = img.size()[0]
        saturation_list = torch.torch.rand(B)*(saturation_bound[1] - saturation_bound[0]) + saturation_bound[0]
        saturation_list.requires_grad_(True)
        saturation_factor = saturation_list.reshape(B,1,1,1).to(img.device)
    #     print(saturation_factor.size())

        return self._blend(img, self.batch_rgb_to_grayscale(img).unsqueeze(1), saturation_factor)

    def batch_adjust_contrast(self,img: Tensor, contrast_bound: float) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        contrast_list = torch.torch.rand(B)*(contrast_bound[1] - contrast_bound[0]) + contrast_bound[0]
        contrast_list.requires_grad_(True)
        contrast_list = contrast_list.reshape(B,1,1,1).to(img.device)

        #mean for each pic (over HxW points)
        img_gray = self.batch_rgb_to_grayscale(img)
        mean = torch.mean(img_gray.reshape(img_gray.shape[0],-1), dim = 1)


        return self._blend(img, mean.reshape([mean.size()[0],1,1,1]),  contrast_list)

    def batch_adjust_hue(self, img: Tensor, hue_bound: list) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''

        B = img.size()[0]
        theta = (torch.torch.rand(B)*(hue_bound[1]-hue_bound[0]) + hue_bound[0])* 3.1415* 2
        theta.requires_grad_(True)

        # generate tensor
        one_tensor = torch.ones(B)
        zero_tensor = torch.zeros(B)
        cos_tensor = torch.cos(theta)
        sin_tensor = torch.sin(theta)

        #stack
        T_hue = torch.stack([one_tensor, zero_tensor, zero_tensor,
                     zero_tensor, cos_tensor, -sin_tensor,
                     zero_tensor, sin_tensor, cos_tensor]).transpose(0,1).reshape(B,3,3)

        T_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
        T_rgb = torch.tensor([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]])
        T_final = torch.matmul(torch.matmul(T_rgb, T_hue), T_yiq)
        T_final = T_final.to(img.device)

        #return T_rgb x T_hue x T_yiq x img

        return torch.matmul(T_final.unsqueeze(1).unsqueeze(1), img.transpose(1,-1).unsqueeze(-1)).squeeze(-1).transpose(1,-1)














def ManualNormalise(x, dataset):
    CACHED_MEAN_STD = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

    mean = CACHED_MEAN_STD[dataset][0]
    std = CACHED_MEAN_STD[dataset][1]

    #batch x channel x height x wide
    B = x.size()[0]
    C = x.size()[1]
    H = x.size()[2]
    W = x.size()[3]

    #create a matrix
    mean_batch = torch.stack([torch.tensor(mean[i]).repeat(H,W) for i in range(C)], dim = 0).repeat(B,1,1,1)
    std_batch = torch.stack([torch.tensor(std[i]).repeat(H,W) for i in range(C)], dim = 0).repeat(B,1,1,1)
    mean_batch = mean_batch.to(x.device)
    std_batch = std_batch.to(x.device)

    norm_x = (x - mean_batch)/std_batch

    return norm_x



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
