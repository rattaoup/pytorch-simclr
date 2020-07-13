import torch
from torch import nn
import numbers
import random
from torch import Tensor

from PIL import ImageFilter
from torchvision import transforms

class DifferentiableColourDistortionByTorch2(nn.Module):

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
#         print(x_gray.size())
#         print(gray.size())
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
        saturation_factor = saturation_list.reshape(B,1,1,1).to(img.device)
    #     print(saturation_factor.size())

        return self._blend(img, self.batch_rgb_to_grayscale(img).unsqueeze(1), saturation_factor)

    def batch_adjust_contrast(self,img: Tensor, contrast_bound: float) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        contrast_list = torch.torch.rand(B)*(contrast_bound[1] - contrast_bound[0]) + contrast_bound[0]
        contrast_list = contrast_list.reshape(B,1,1,1).to(img.device)

        #mean for each pic (over HxW points)
        img_gray = self.batch_rgb_to_grayscale(img)
        mean = torch.mean(img_gray.view(img_gray.shape[0],-1), dim = 1)


        return self._blend(img, mean.reshape([mean.size()[0],1,1,1]),  contrast_list)

    def batch_adjust_hue(self, img: Tensor, hue_bound: float) -> Tensor:
        '''
        Batch x C x H x  W -> Batch x C x H x W
        '''
        B = img.size()[0]
        hue_list = torch.torch.rand(B)*(hue_bound[1] - hue_bound[0]) + hue_bound[0]
        hue_list = hue_list.to(img.device)

        #unbind
        img_u = torch.unbind(img, dim = 0)


        return torch.stack([self.adjust_hue(img_i, hue_list[i]) for i, img_i in enumerate(img_u)], dim = 0)


    def _rgb2hsv(self, img):
        r, g, b = img.unbind(0)

        maxc = torch.max(img, dim=0).values
        minc = torch.min(img, dim=0).values

        # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
        # from happening in the results, because
        #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
        #   + H channel has division by `(maxc - minc)`.
        #
        # Instead of overwriting NaN afterwards, we just prevent it from occuring so
        # we don't need to deal with it in case we save the NaN in a buffer in
        # backprop, if it is ever supported, but it doesn't hurt to do so.
        eqc = maxc == minc

        cr = maxc - minc
        # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
        s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
        # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
        # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
        # would not matter what values `rc`, `gc`, and `bc` have here, and thus
        # replacing denominator with 1 when `eqc` is fine.
        cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

    #     hr = (maxc == r) * (bc - gc)
    #     hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    #     hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        hr = (maxc == r).float() * (bc - gc)
        hg = ((maxc == g) & (maxc != r)).float() * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)).float() * (4.0 + gc - rc)
        h = (hr + hg + hb)
        h = torch.fmod((h / 6.0 + 1.0), 1.0)
        return torch.stack((h, s, maxc))

    def _hsv2rgb(self, img):
        h, s, v = img.unbind(0)
        i = torch.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.to(dtype=torch.int32)

        p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
        q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
        t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6

    #     mask = i == torch.arange(6)[:, None, None]
        mask = (i == torch.arange(6)[:, None, None].to(img.device).int())
        a1 = torch.stack((v, q, p, p, t, v))
        a2 = torch.stack((t, v, v, q, p, p))
        a3 = torch.stack((p, p, t, v, v, q))
        a4 = torch.stack((a1, a2, a3))

        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4)

    def adjust_hue(self, img, hue_factor):

        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

        if not self._is_tensor_a_torch_image(img):
            raise TypeError('tensor is not a torch image.')

        orig_dtype = img.dtype
        if img.dtype == torch.uint8:
            img = img.to(dtype=torch.float32) / 255.0

        img = self._rgb2hsv(img)
        h, s, v = img.unbind(0)
        h += hue_factor
        h = h % 1.0
        img = torch.stack((h, s, v))
        img_hue_adj = self._hsv2rgb(img)

        if orig_dtype == torch.uint8:
            img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

        return img_hue_adj




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

class DifferentiableColourDistortionByTorch(nn.Module):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

#     @torch.jit.unused
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

        '''
        xu = torch.unbind(x, dim=0)
        aug_x = torch.stack([self.colordistort(x_i) for i,x_i in enumerate(xu)], dim = 0)


        return aug_x

    def colordistort(self, img: Tensor)-> Tensor:
        if (torch.rand(1)<0.8):
            img = self.colorjitter(img)
            #print(x_i.shape)
        if (torch.rand(1)<0.2):
            img = self.rgb_to_grayscale(img)
            img = img.repeat(3,1,1) # reshape to r==g==b
            #print(x_i.shape)
        return img


    def colorjitter(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        #### index
#         brightness_factor = None
#         contrast_factor = None
#         saturation_factor = None
#         hue_factor = None

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = self.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = self.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = self.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = self.adjust_hue(img, hue_factor)

        #also return the coefficient
#         format_string = self.__class__.__name__ + '('
#         format_string += 'brightness={:.3f}'.format(brightness_factor)
#         format_string += ', contrast={:.3f}'.format(contrast_factor)
#         format_string += ', saturation={:.3f}'.format(saturation_factor)
#         format_string += ', hue={:.3f})'.format(hue_factor)
        #print(format_string)

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

    def rgb_to_grayscale(self, img: Tensor) -> Tensor:
        """Convert the given RGB Image Tensor to Grayscale.
        For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
        is L = R * 0.2989 + G * 0.5870 + B * 0.1140

        Args:
            img (Tensor): Image to be converted to Grayscale in the form [C, H, W].

        Returns:
            Tensor: Grayscale image.

        """
        if img.shape[0] != 3:
            raise TypeError('Input Image does not contain 3 Channels')

        return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)

    def _rgb2hsv(self, img):
        r, g, b = img.unbind(0)

        maxc = torch.max(img, dim=0).values
        minc = torch.min(img, dim=0).values

        # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
        # from happening in the results, because
        #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
        #   + H channel has division by `(maxc - minc)`.
        #
        # Instead of overwriting NaN afterwards, we just prevent it from occuring so
        # we don't need to deal with it in case we save the NaN in a buffer in
        # backprop, if it is ever supported, but it doesn't hurt to do so.
        eqc = maxc == minc

        cr = maxc - minc
        # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
        s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
        # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
        # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
        # would not matter what values `rc`, `gc`, and `bc` have here, and thus
        # replacing denominator with 1 when `eqc` is fine.
        cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

    #     hr = (maxc == r) * (bc - gc)
    #     hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    #     hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        hr = (maxc == r).float() * (bc - gc)
        hg = ((maxc == g) & (maxc != r)).float() * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)).float() * (4.0 + gc - rc)
        h = (hr + hg + hb)
        h = torch.fmod((h / 6.0 + 1.0), 1.0)
        return torch.stack((h, s, maxc))

    def _hsv2rgb(self, img):
        h, s, v = img.unbind(0)
        i = torch.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.to(dtype=torch.int32)

        p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
        q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
        t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6

    #     mask = i == torch.arange(6)[:, None, None]
        mask = (i == torch.arange(6)[:, None, None].to(img.device).int())
        a1 = torch.stack((v, q, p, p, t, v))
        a2 = torch.stack((t, v, v, q, p, p))
        a3 = torch.stack((p, p, t, v, v, q))
        a4 = torch.stack((a1, a2, a3))

        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4)

    #### Brightness ####
    def adjust_brightness(self, img: Tensor, brightness_factor: float) -> Tensor:
        """Adjust brightness of an RGB image.

        Args:
            img (Tensor): Image to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.

        Returns:
            Tensor: Brightness adjusted image.
        """
        if brightness_factor < 0:
            raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

        if not self._is_tensor_a_torch_image(img):
            raise TypeError('tensor is not a torch image.')

        return self._blend(img, torch.zeros_like(img), brightness_factor)

    #### Contrast ####
    def adjust_contrast(self, img: Tensor, contrast_factor: float) -> Tensor:
        """Adjust contrast of an RGB image.

        Args:
            img (Tensor): Image to be adjusted.
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.

        Returns:
            Tensor: Contrast adjusted image.
        """
        if contrast_factor < 0:
            raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

        if not self._is_tensor_a_torch_image(img):
            raise TypeError('tensor is not a torch image.')

        mean = torch.mean(self.rgb_to_grayscale(img).to(torch.float))

        return self._blend(img, mean, contrast_factor)

    #### Hue ####
    def adjust_hue(self, img, hue_factor):
        """Adjust hue of an image.

        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.

        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.

        See `Hue`_ for more details.

        .. _Hue: https://en.wikipedia.org/wiki/Hue

        Args:
            img (Tensor): Image to be adjusted. Image type is either uint8 or float.
            hue_factor (float):  How much to shift the hue channel. Should be in
                [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                HSV space in positive and negative direction respectively.
                0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                with complementary colors while 0 gives the original image.

        Returns:
             Tensor: Hue adjusted image.
        """
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

        if not self._is_tensor_a_torch_image(img):
            raise TypeError('tensor is not a torch image.')

        orig_dtype = img.dtype
        if img.dtype == torch.uint8:
            img = img.to(dtype=torch.float32) / 255.0

        img = self._rgb2hsv(img)
        h, s, v = img.unbind(0)
        h += hue_factor
        h = h % 1.0
        img = torch.stack((h, s, v))
        img_hue_adj = self._hsv2rgb(img)

        if orig_dtype == torch.uint8:
            img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

        return img_hue_adj

    def adjust_saturation(self, img: Tensor, saturation_factor: float) -> Tensor:
        """Adjust color saturation of an RGB image.

        Args:
            img (Tensor): Image to be adjusted.
            saturation_factor (float):  How much to adjust the saturation. Can be any
                non negative number. 0 gives a black and white image, 1 gives the
                original image while 2 enhances the saturation by a factor of 2.

        Returns:
            Tensor: Saturation adjusted image.
        """
        if saturation_factor < 0:
            raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

        if not self._is_tensor_a_torch_image(img):
            raise TypeError('tensor is not a torch image.')

        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)


















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
