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

from daml._prototype.utils.layer import TransformerBlock
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
        pyramid_depth: int,
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
        block_depths: list[int],
        sr_ratios: list[int],
        reshape_last_stage: bool = True,
        extra_tokens: int = 0,
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
        in_channels = [in_channel] + [embed_dims[i] for i in range(pyramid_depth - 1)]

        # patch embeddings
        embeddings = []
        for i in range(pyramid_depth):
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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_depths))]

        # Pyramid layers
        layers = []
        cur = 0
        for i in range(pyramid_depth):
            # Each layer consists of a pretty normal transformer block
            blocks = []
            if i != 0:
                cur += block_depths[i - 1]
            for j in range(block_depths[i]):
                blocks.append(
                    TransformerBlock(
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
            layers.append(nn.ModuleList(blocks))

        self.pyramid = nn.ModuleList(layers)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(embed_dims[i]) for i in range(pyramid_depth)]
        )

        # Extra tokens for the model simply for calculations
        self.num_extra_tokens = extra_tokens
        self.register_tokens = (
            nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, extra_tokens, embed_dims[i]))
                    for i in range(pyramid_depth)
                ]
            )
            if extra_tokens
            else None
        )

        # If using masks
        self.mask_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, embed_dims[i])) for i in range(pyramid_depth)]
        )

    def forward(
        self,
        pixel_values: Tensor,
        masks: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_extra_tokens: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = ()
        all_self_attentions = ()
        all_extra_tokens = ()

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, (embedding_layer, blocks, norm_layer) in enumerate(
            zip(self.patch_embeddings, self.pyramid, self.layer_norm)
        ):
            # Step 1: obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # Optional Step 2: apply masks
            if masks is not None:
                hidden_states = torch.where(
                    masks.unsqueeze(-1),
                    self.mask_token[idx].to(hidden_states.dtype).unsqueeze(0),
                    hidden_states,
                )
                # !!!! Need to figure out how to address shrinking image dimension
                # !!!! when applying masks
            # Optional Step 3: add extra tokens
            if self.register_tokens is not None:
                hidden_states = torch.cat(
                    (
                        self.register_tokens[idx].expand(batch_size, -1, -1),
                        hidden_states,
                    ),
                    dim=1,
                )
            # Step 4: send embeddings through blocks
            for i, blk in enumerate(blocks):  # type: ignore
                blk_output = blk(hidden_states, height, width, output_attentions)
                hidden_states = blk_output[0]  # context layer
                if output_attentions:
                    # attention probabilities
                    all_self_attentions = all_self_attentions + (blk_output[1],)
            # Step 5: apply layer norm
            hidden_states = norm_layer(hidden_states)
            # Optional Step 6: remove extra tokens
            if self.register_tokens is not None:
                tokens = hidden_states[:, : self.num_extra_tokens + 1]
                hidden_states = hidden_states[:, self.num_extra_tokens + 1 :]
            if output_extra_tokens:
                all_extra_tokens = all_extra_tokens + (tokens,)
            # Optional Step 7: reshape back to (batch_size, num_channels, height, width)
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
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_extra_tokens,
                ]
                if v
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,  # type: ignore
            attentions=all_self_attentions,  # type: ignore
            register_tokens=all_extra_tokens,  # type: ignore
        )
