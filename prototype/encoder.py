# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from daml._prototype.utils.layer import SegformerLayer
from daml._prototype.utils.modeling_outputs import BaseModelOutput
from daml._prototype.utils.patch_embed import PatchEmbed


def _get_size(i, image_size, downsampling) -> Union[int, Tuple[int, int]]:
    """
    Determines the type of image_size and adjusts downsampling accordingly.

    Returns image_size // downsampling
    """
    downsample = downsampling[i]
    # Check if image_size is an int
    if isinstance(image_size, int):
        if isinstance(downsample, int):
            return image_size // downsample
        else:
            return image_size // downsample[0]
    # Check if image_size is a tuple
    elif isinstance(image_size, tuple):
        if isinstance(downsample, int):
            return (image_size[0] // downsample, image_size[1] // downsample)
        else:
            return (image_size[0] // downsample[0], image_size[1] // downsample[1])
    # If it's something else, then there are issues
    else:
        raise TypeError(
            "The variable image_size should be of type int or \
                tuple[int, int], but it's not!!"
        )


class SegformerEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        image_size: Union[int, Tuple[int, int]],
        downsampling: list[Union[int, Tuple[int, int]]],
        patch_size: list[int],
        in_channel: int,
        embed_dims: list[int],
        strides: list[int],
        padding: list[int],
        pad_methods: list[str],
        dilation: list[int],
        num_heads: list[int],
        mlp_ratios: list[int],
        drop_rate: list[float],
        attn_drop_rate: list[float],
        drop_path_rate: float,
        depths: list[int],
        sr_ratios: list[int],
        reshape_last_stage: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        attn_class: Optional[Callable[..., nn.Module]] = None,
        ffn_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.reshape_last_stage = reshape_last_stage
        # Prepare optional parameters
        optional_params = {
            "norm_layer": norm_layer,
            "drop_layer": drop_layer,
            "attn_class": attn_class,
            "ffn_layer": ffn_layer,
            "act_layer": act_layer,
        }
        # Filter out None values
        optional_params = {k: v for k, v in optional_params.items() if v is not None}

        # Getting the channels per block
        in_channels = [in_channel] + [embed_dims[i] for i in range(num_blocks - 1)]

        # patch embeddings
        embeddings = []
        for i in range(num_blocks):
            embeddings.append(
                PatchEmbed(
                    image_size=_get_size(i, image_size, downsampling),
                    patch_size=patch_size[i],
                    in_channel=in_channels[i],
                    embed_dim=embed_dims[i],
                    stride=strides[i],
                    pad=padding[i],
                    padding_mode=pad_methods[i],
                    dilate=dilation[i],
                    norm_layer=norm_layer,
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(num_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += depths[i - 1]
            for j in range(depths[i]):
                layers.append(
                    SegformerLayer(
                        embed_dim=embed_dims[i],
                        num_attn_heads=num_heads[i],
                        drop_path=dpr[cur + j],
                        attn_dropout=attn_drop_rate[i],
                        dropout=drop_rate[i],
                        sr_ratio=sr_ratios[i],
                        mlp_ratio=mlp_ratios[i],
                        **optional_params,
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(embed_dims[i]) for i in range(num_blocks)]
        )

    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = ()
        all_self_attentions = ()

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, (embedding_layer, block_layer, norm_layer) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):  # type: ignore
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]  # context layer
                if output_attentions:
                    # attention probabilities
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to
            # (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or self.reshape_last_stage:
                hidden_states = (
                    hidden_states.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,  # type: ignore
            attentions=all_self_attentions,  # type: ignore
        )
