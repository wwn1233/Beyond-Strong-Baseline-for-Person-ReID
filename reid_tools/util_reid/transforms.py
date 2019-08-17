from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from PIL import Image
from torchvision import transforms as T

import cv2
import numpy as np
import math
import torch

import _init_paths  # pylint: disable=unused-import

from core.config import cfg

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        random_value = np.random.random()  # random.random()
        # print(random_value)
        if random_value > self.probability:
            return img
        # count = 0
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # count +=1
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img


        return img


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        random_value = np.random.random() #random.random()
        if random_value < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(np.random.uniform(0, x_maxrange)))
        y1 = int(round(np.random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        random_value = np.random.random()  # random.random()
        if random_value < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class TrainTransform(object):
    def __init__(self, h, w, type = 0):
        self.h = h
        self.w = w
        self.type = type
        # self.transform_other_1 = T.ColorJitter(0.2,0.2,0.2)

        # self.transform_other_2 = Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec'])




    def __call__(self, x):
        if self.type == 0:
            x = Random2DTranslation(self.h, self.w)(x)
            # x = T.RandomHorizontalFlip()(x)

            x = RandomHorizontalFlip()(x)
            # x = self.transform_other_1(x)
            x = T.ToTensor()(x)
            # x = self.transform_other_2(x)
            x = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(x)
            # print(cfg.REID.RE)
            if cfg.REID.RE:
                x = RandomErasing(mean=[0.0, 0.0, 0.0])(x)
        # elif self.type == 1:
        #     x = Random2DTranslation(self.h, self.w)(x)
        #     # x = T.RandomHorizontalFlip()(x)
        #     x = process_1(image=x)['image']
        #     x -= PIXEL_MEANS
        #     x = x.transpose(2, 0, 1)

        else:
            ValueError('Not existing this type of Transform!')
        return x


class TestTransform(object):
    def __init__(self, h, w, type = 0):
        self.h = h
        self.w = w
        self.type = type
        # self.process_test = A.Compose([A.Resize(h,w),],p=1)

    def __call__(self, x=None):
        if self.type == 0:
            x = T.Resize((self.h, self.w))(x)
            x = T.ToTensor()(x)
            x = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(x)
        # elif self.type == 1:
        #     # x = T.Resize((self.h, self.w))(x)
        #     x = self.process_test(image=x)['image']
        #     x -= PIXEL_MEANS
        #     x = x.transpose(2, 0, 1)
        else:
            ValueError('Not existing this type of Transform!')
        return x

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


if __name__ == '__main__':
    pass