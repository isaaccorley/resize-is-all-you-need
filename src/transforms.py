import kornia.augmentation as K
import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Clip(nn.Module):
    def __init__(self, min=0, max=255):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clip(x, self.min, self.max)


def seco_rgb_transforms():
    _min = torch.tensor([3.0, 2.0, 0.0])
    _max = torch.tensor([88.0, 103.0, 129.0])
    _mean = torch.tensor(IMAGENET_MEAN)
    _std = torch.tensor(IMAGENET_STD)
    return [
        K.Normalize(mean=_min, std=_max - _min),
        K.Normalize(mean=torch.tensor(0), std=1 / torch.tensor(255)),
        Clip(0.0, 255.0),
        K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
        K.Normalize(mean=_mean, std=_std),
    ]


def ssl4eo_transforms():
    return [K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(10000.0))]


def sentinel2_transforms():
    return [K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(10000.0))]


def uint8_transforms():
    return [K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(255.0))]


def imagenet_transforms():
    return [
        K.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD))
    ]
