import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class GaussianBlurSimCLR(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class GaussianBlurBarlowTwins(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    # size和s 用于simclr的random crop
    def __init__(self, size, s=1):
        # barlow twins data aug
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlurBarlowTwins(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            # imagenet的
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.441, 0.427, 0.385],
                                 std=[0.231, 0.226, 0.224])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlurBarlowTwins(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.441, 0.427, 0.385],
                                 std=[0.231, 0.226, 0.224])
        ])
        # # simclr data aug
        # self.transform_simclr = transforms.Compose([
        #     transforms.RandomResizedCrop(size=size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     GaussianBlurSimCLR(kernel_size=int(0.1 * size)),
        #     transforms.ToTensor()
        # ])
        # simclr data aug
        self.transform_simclr = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlurSimCLR(kernel_size=int(0.1 * size)),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_simclr(x)
        y4 = self.transform_simclr(x)
        return y1, y2, y3, y4
