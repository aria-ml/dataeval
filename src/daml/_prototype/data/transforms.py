# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
#   https://github.com/DeepVoltaire/AutoAugment/blob/master/ops.py
#   https://github.com/mhamilton723/FeatUp/blob/main/featup/datasets/JitteredImage.py

import random
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors as tvt
from torchvision.transforms import v2


class ToTVTensor(v2.ToImage):
    """
    Convert a ``torch.Tensor``, ``PIL Image`` or ``numpy.ndarray``
    to a torchvision tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic: Image to be converted to tensor.
                 (PIL Image, numpy.ndarray or torch.tensor)

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, tvt.Image):
            return pic
        return super().__call__(pic)


class GaussianBlur(v2.RandomApply):
    """
    Randomly apply Gaussian Blur to the image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class RandomTranslate(v2.RandomApply):
    """
    Randomly translate the image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class RandomZoom(v2.RandomApply):
    """
    Randomly apply zoom to the image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def apply_jitter(img, max_pad, transform_params):
    h, w = img.shape[2:]

    padded = F.pad(img, [max_pad] * 4, mode="reflect")

    zoom = transform_params["zoom"].item()
    x = transform_params["x"].item()
    y = transform_params["y"].item()
    flip = transform_params["flip"].item()

    if zoom > 1.0:
        zoomed = F.interpolate(padded, scale_factor=zoom, mode="bilinear")
    else:
        zoomed = padded

    cropped = zoomed[:, :, x : h + x, y : w + y]

    if flip:
        return torch.flip(cropped, [3])
    else:
        return cropped


def sample_transform(use_flips, max_pad, max_zoom, h, w):
    flip = random.random() > 0.5 if use_flips else False

    apply_zoom = random.random() > 0.5
    zoom = random.random() * (max_zoom - 1) + 1 if apply_zoom else 1.0

    valid_area_h = (int((h + max_pad * 2) * zoom) - h) + 1
    valid_area_w = (int((w + max_pad * 2) * zoom) - w) + 1

    return {
        "x": torch.tensor(torch.randint(0, valid_area_h, ()).item()),
        "y": torch.tensor(torch.randint(0, valid_area_w, ()).item()),
        "zoom": torch.tensor(zoom),
        "flip": torch.tensor(flip),
    }


class JitteredImage(Dataset):
    def __init__(self, img, length, use_flips, max_zoom, max_pad):
        self.img = img
        self.length = length
        self.use_flips = use_flips
        self.max_zoom = max_zoom
        self.max_pad = max_pad

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        h, w = self.img.shape[2:]
        transform_params = sample_transform(
            self.use_flips, self.max_pad, self.max_zoom, h, w
        )
        return apply_jitter(self.img, self.max_pad, transform_params).squeeze(
            0
        ), transform_params
