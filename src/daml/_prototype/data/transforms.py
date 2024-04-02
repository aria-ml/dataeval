# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
#   https://github.com/DeepVoltaire/AutoAugment/blob/master/ops.py
#   https://github.com/mhamilton723/FeatUp/blob/main/featup/datasets/JitteredImage.py

from typing import Any, List, Sequence

import torch
from torchvision.transforms import v2

LISTOFTRANSFORMS = """ 
v2.ScaleJitter
Crop
Translate
Rotate
Invert - Normalize pixels to [0-1] and invert (1 - norm_pix)
Equalize - Equalize the image histogram
Solarize - Invert pixels above a given threshold
Posterize - Reduce the number of bits for each pixel to __ bits
Contrast - Find current contrast (max-min)/2**(8 or 12 or 16) depending on the closest max, then randomly readjust
Brightness - Adjust the Brightness of the image, 0 - black image, 1 - original image, 2 - white image
Sharpness -  Amount of blur to the image, 0 - blurry, 1 - orginial, 2 - sharpened image
Color - B/W vs RGB
Color Swapping - adjusting the values of specific colors
Outlines - Running gradient tests on the image to produce an outline



"""


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(v2.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor,
    or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic: Image to be converted to tensor.
                 (PIL Image, numpy.ndarray or torch.tensor)

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list: List[Any] = [
        v2.RandomResizedCrop(crop_size, interpolation=interpolation)
    ]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return v2.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=v2.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [
        v2.Resize(resize_size, interpolation=interpolation),
        v2.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return v2.Compose(transforms_list)


class DataAugmentationDINO:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = v2.Compose(
            [color_jittering, global_transfo1_extra, self.normalize]
        )
        self.global_transfo2 = v2.Compose(
            [color_jittering, global_transfo2_extra, self.normalize]
        )
        self.local_transfo = v2.Compose(
            [color_jittering, local_transfo_extra, self.normalize]
        )


import random

import torch.nn.functional as F
from torch.utils.data import Dataset


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
    if use_flips:
        flip = random.random() > 0.5
    else:
        flip = False

    apply_zoom = random.random() > 0.5
    if apply_zoom:
        zoom = random.random() * (max_zoom - 1) + 1
    else:
        zoom = 1.0

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


from PIL import Image, ImageEnhance, ImageOps


class ShearX:
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class ShearY:
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class TranslateX:
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor,
        )


class TranslateY:
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor,
        )


class Rotate:
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(
            rot, Image.new("RGBA", rot.size, (128,) * 4), rot
        ).convert(x.mode)


class Color:
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize:
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize:
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast:
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness:
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness:
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class AutoContrast:
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize:
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert:
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)
