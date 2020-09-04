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
