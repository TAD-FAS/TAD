import numpy as np
import random
from PIL import Image

 
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as fun

 
# 使得gt和原图进行相同的transform操作，

class Resize(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target=None):
        image = fun.resize(image, self.size, interpolation=fun.InterpolationMode.NEAREST)
        if target is not None:
            target = fun.resize(target, self.size, interpolation=fun.InterpolationMode.NEAREST)
        return image, target
 
 
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = fun.hflip(image)
            if target is not None:
                target = fun.hflip(target)
        return image, target
 
class RandomCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = fun.crop(image, *crop_params)
        if target is not None:
            target = fun.crop(target, *crop_params)
        return image, target

class RandomRotation(object):
    def __init__(self, degree, interpolation=fun.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None):
        self.degree = degree
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        self.resample = resample

    def __call__(self, img, target=None):
        angle = float(torch.empty(1).uniform_(float(-self.degree), float(self.degree)).item())
        image = fun.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill, self.resample)
        if target is not None:
            target = fun.rotate(target, angle, self.interpolation, self.expand, self.center, self.fill, self.resample)
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        '''(min, max)'''
        super(ColorJitter, self).__init__()
        self.color = T.ColorJitter(brightness, contrast, saturation, hue)
        # self.brightness = brightness
        # self.contrast = contrast
        # self.saturation = saturation
        # self.hue = hue
    def __call__(self, img, target=None):
        img = self.color(img)
        
        # b = None if self.brightness is None else float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        # c = None if self.contrast is None else float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        # s = None if self.saturation is None else float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        # # -0.5 <= min <= max <= 0.5
        # h = None if self.hue is None else float(torch.empty(1).uniform_(self.hue[0], self.hue[1])) 
        # img = F.adjust_brightness(img, b)
        # img = F.adjust_contrast(img, c)
        # img = F.adjust_saturation(img, s)
        # img = F.adjust_hue(img, h)
        return img, target
        
class CenterCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        image = fun.center_crop(image, self.size)
        if target is not None:
            target = fun.center_crop(target, self.size)
        return image, target
 
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target):
        image = fun.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            target = fun.normalize(target, mean=self.mean[0], std=self.std[0])
        return image, target
    
        
 
class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0,padding_mode=' constant'):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value
        self.padding_mode = padding_mode
 
    def __call__(self, image, target):
        image = fun.pad(image, self.padding_n, self.padding_fill_value, padding_mode=self.padding_mode)
        if target is not None:
            target = fun.pad(target, self.padding_n, self.padding_fill_target_value, padding_mode=self.padding_mode)
        return image, target
 
class ToTensor(object):
    def __call__(self, image, target):
        image = fun.to_tensor(image)
        if target is not None:
            target = fun.to_tensor(target)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
 
 
