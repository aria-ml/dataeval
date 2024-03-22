# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT Liscence
# found in the LICENSE file in the root directory of this source tree.
#
# References:
#   https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


def to_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    Construct the patch embeddings from an image.
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size:      Image size, either single int or tuple(int, int).
        patch_size:    Patch token size, either single int or tuple(int, int).
        in_channels:   Number of input image channels, C.
        embed_dim:     Number of linear projection output channels, D.
        norm_layer:    Normalization layer. Default nn.LayerNorm

        Convolution Args:
        stride:        Stride of the convolution. Default 1
        pad:           Amount of padding added to all 4 sides of input. Default 0
        padding_mode:  Type of padding added: zeros, reflect, replicate or circular.
                        Default: zeros
        dilate:        Spacing between kernel elements. Default 1

    Input:
        x:             Input Tensor (forward only)

    Output:
        embeddings:    Flattened Tensor of shape (B, HW, D)
        height:        Height of the embedding, H - int
        width:         Width of the embedding, W - int
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        stride: int = 1,
        pad: Union[int, Tuple[int, int]] = 0,
        padding_mode: str = "zeros",
        dilate: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
        # flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.H = self.img_size[0] // self.patch_size[0]
        self.W = self.img_size[1] // self.patch_size[1]

        self.num_patches = self.H * self.W
        # ^ for reference only
        # The Output Image Size of each stage = [h/R,w/R]
        #   where R is governed by the patch size & stride
        # Output size of in later stages is further reduced
        #   sequentially by patch_size & stride

        self.in_chans = in_channels
        self.embed_dim = embed_dim

        # self.flatten_embedding = flatten_embedding

        padding = to_2tuple(pad)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=stride,
            padding=padding,
            dilation=dilate,
            padding_mode=padding_mode,
        )

        self.layer_norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x = self.proj(x)  # B D H W
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)  # B HW D
        x = self.layer_norm(x)
        # if not self.flatten_embedding:
        #     x = x.reshape(-1, height, width, self.embed_dim)  # B H W D
        return x, height, width
