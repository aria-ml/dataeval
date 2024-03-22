# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT Liscence
# found in the LICENSE file in the root directory of this source tree.
#
# References:
#   https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py


from typing import Callable

import torch
from torch import Tensor, nn

from prototype.activations import ACT2FN
from prototype.attention import SegformerAttention


# Stochastic depth implementation
def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).

    Args:
        x:          Input Tensor
        drop_prob:  Probability of dropout, float between 0 and 1.
        training:   Only perform dropout when training, bool.

    Output:
        output:     Output Tensor
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).

    Args:
        drop_prob:  Probability of dropout, float between 0 and 1.

    Input:
        x:          Input Tensor (forward only)

    Output:
        output:     Output Tensor
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class SegformerDWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width).contiguous()
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer, drop, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        if isinstance(act_layer, str):
            self.intermediate_act_fn = ACT2FN[act_layer]
        else:
            self.intermediate_act_fn = act_layer
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)  # dim[nbatch,(h*w)/R^2,mlp_hidden_size]
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)  # dim[nbatch,(h*w)/R^2,hidden_size]
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerLayer(nn.Module):
    """
    This is the basic layer to the Segformer model.

    Args:
        embed_dim:       Number of linear projection channels, D - int.
        num_attn_heads:  Number of attention heads, int.
        drop_path:       Probability of drop out, float.
        sr_ratio:        Value to reduce the dimensions for self-attention by,
                          int[1, embed_dim] (sequence reduction ratio).
        mlp_ratio:       Scaling factor for determining the size of the internal mlp
                          space based on embed_dim, float -> mlp_ratio x embed_dim.
        norm_layer:      Torch module for normalization. Default nn.LayerNorm
        attn_class:      Torch module for attention. Default SegformerAttention
        ffn_layer:       Torch module for feedforward network. Default SegformerMixFFN
        act_layer:       Torch module for ffn_layer activation layer. Default nn.GELU

    Input:
        x:          Input Tensor (forward only)

    Output:
        output:     Output Tensor
    """

    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        drop_path: float = 0.0,
        attn_dropout: float = 0.0,  # config.attention_probs_dropout_prob
        dropout: float = 0.0,  # config.hidden_dropout_prob
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SegformerAttention,
        ffn_layer: Callable[..., nn.Module] = SegformerMixFFN,
        act_layer: Callable[..., nn.Module] = nn.GELU,  # config.hidden_act
    ) -> None:
        super().__init__()

        self.layer_norm_1 = norm_layer(embed_dim)
        self.attention = attn_class(
            embed_dim=embed_dim,
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = norm_layer(embed_dim)
        mlp_hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=embed_dim,
            hidden_features=mlp_hidden_size,
            act_layer=act_layer,
            drop=dropout,
        )

    def forward(self, embed_dim, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(embed_dim),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]  # context layer
        outputs = self_attention_outputs[1:]  # attention probs

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + embed_dim  # dim[nbatch,(h*w)/R^2,nchann], nchann: ref Table 6 in segformer

        # Mix-FFN, leak location information
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)  # same size as hidden_states

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs  # (layer out, attention probs)

        return outputs
