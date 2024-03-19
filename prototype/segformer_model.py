# coding=utf-8
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# TODO >> You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SegFormer model with ARiA additions """


import collections
import math
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from daml._prototype.activations import ACT2FN
from daml._prototype.modeling_outputs import (
    BaseModelOutput,
    SemanticSegmentationModelOutput,
)
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torchvision import transforms as T


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head
        # and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(
    layer: nn.Linear, index: torch.LongTensor, dim: int = 0
) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# Stochastic depth implementation
# Taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is the same as the
    DropConnect impl I created for EfficientNet, etc networks, however, the original name is misleading as 'Drop
    Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the patch embeddings from an image."""

    def __init__(
        self,
        image_size,
        patch_size,
        stride,
        num_channels,
        hidden_size,
        pad,
        dilate,
        mode,
    ):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        self.height, self.width = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.num_patches = self.height * self.width
        # ^ for reference only
        # The Output Image Size of each stage = [h/R,w/R]
        #   where R is governed by the patch size, stride
        # Output size of in later stages is further reduced
        #   sequentially by patch_size, stride

        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(pad, pad),
            dilation=dilate,
            padding_mode=mode,
        )
        # ^ reading in all parameters from model config

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        x = self.proj(pixel_values)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x, height, width


class SegformerEfficientSelfAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sr_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(
            config.attention_probs_dropout_prob
        )  # randomly 0's elements, scales by 1/(1-p)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape).contiguous()
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        """
        CONCEPT OVERVIEW:
        ----------------
        Query:      flattened, transformed patch embeddings - our sequence - where we want to look
                    - dim[b, a, (h*w)/R^2, c)]
        Key:        compressed, transformed, flattened patch embeddings - our available data - where we want to compare
                    - dim[b, a, sr(h)*sr(w), c]
        Attention:  similarity between our sequence and the data that's available - matmul(Q,K)
                    - dim[b, a, (h*w)/R^2, sr(h)*sr(w)]
        Value:      compressed, transformed, flattened patch embeddings - our available data - that will be weighted by
                    the attention scores to provide the data that matters most to us
                    - dim[b, a, sr(h)*sr(w), c]
        Context:    the weighted data that matters most to us, this is passed along in the forward call
                    - dim[b, a, (h*w)/R^2, c]
        b, a, c:    n_batch, n_attention_heads, n_channels, respectively
        h, w, R     native image height, native image width, downsample factor, respectively
        """

        query_layer = self.transpose_for_scores(
            self.query(hidden_states)
        )  # dim[nbatch,nattnhead,(h*w)/R^2,nchann)]

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape  # seq_len = h*w
            hidden_states = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )  # dim[nbatch,nchann,h,w]
            hidden_states = self.sr(
                hidden_states
            )  # compress along spatial dimension [h,w] using Conv2d
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(
                0, 2, 1
            )  # re-flatten [h,w]
            hidden_states = self.layer_norm(
                hidden_states
            )  # dim[nbatch,sr(h)*sr(w),nchann]

        # Transform patch embeddings (hidden_states) through SEPARATE linear layers and transpose to get keys, values
        key_layer = self.transpose_for_scores(
            self.key(hidden_states)
        )  # dim[nbatch,nattnhead,sr(h)*sr(w),nchann]
        value_layer = self.transpose_for_scores(
            self.value(hidden_states)
        )  # dim[nbatch,nattnhead,sr(h)*sr(w),nchann]
        # ^ NOTICE: key and value layer are the same. ergo, self attention, not cross attention.

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # dim[nbatch,nattnhead,(h*w)/R^2,sr(h)*sr(w)]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper. << JC - attn dropout prob = 0 in model config
        attention_probs = self.dropout(
            attention_probs
        )  # dim[nbatch,nattnhead,(h*w)/R^2,sr(h)*sr(w)]

        context_layer = torch.matmul(
            attention_probs, value_layer
        )  # dim[nbatch,nattnhead,(h*w)/R^2,nchann]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(
            *new_context_layer_shape
        ).contiguous()  # dim[nbatch,(h*w)/R^2,nchann]

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sr_ratio):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sr_ratio=sr_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_outputs = self.self(
            hidden_states, height, width, output_attentions
        )  # hidden_states = patch_embeddings
        # ^ Self attention output: (context_layer, attention_probs)

        attention_output = self.output(
            self_outputs[0], hidden_states
        )  # dim[nbatch, (h*w)/R^2, nchann], nchann: ref segformer Table 6
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # (context layer, attention probs)
        return outputs


class SegformerDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = (
            hidden_states.transpose(1, 2)
            .view(batch_size, num_channels, height, width)
            .contiguous()
        )
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(
            hidden_states
        )  # dim[nbatch,(h*w)/R^2,mlp_hidden_size]
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)  # dim[nbatch,(h*w)/R^2,hidden_size]
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self, config, hidden_size, num_attention_heads, drop_path, sr_ratio, mlp_ratio
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(
            config, in_features=hidden_size, hidden_features=mlp_hidden_size
        )

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(
                hidden_states
            ),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]  # context layer
        outputs = self_attention_outputs[1:]  # attention probs

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = (
            attention_output + hidden_states
        )  # dim[nbatch,(h*w)/R^2,nchann], nchann: ref Table 6 in segformer

        # Mix-FFN, leak location information
        mlp_output = self.mlp(
            self.layer_norm_2(hidden_states), height, width
        )  # same size as hidden_states

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs  # (layer out, attention probs)

        return outputs


class SegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    image_size=config.image_size // config.downsampling_rates[i],
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=(
                        config.num_channels if i == 0 else config.hidden_sizes[i - 1]
                    ),
                    hidden_size=config.hidden_sizes[i],
                    pad=(
                        config.square_pad[i]
                        if hasattr(config, "square_pad")
                        else config.patch_sizes[i] // 2
                    ),
                    dilate=config.dilation[i] if hasattr(config, "dilation") else 1,
                    mode=(
                        config.pad_method[i]
                        if hasattr(config, "pad_method")
                        else "zeros"
                    ),
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=dpr[cur + j],
                        sr_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_sizes[i])
                for i in range(config.num_encoder_blocks)
            ]
        )

    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]  # context layer
                if output_attentions:
                    all_self_attentions = all_self_attentions + (
                        layer_outputs[1],
                    )  # attention probs
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
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
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


SEGFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGFORMER_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`SegformerFeatureExtractor`]. See [`SegformerFeatureExtractor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


"""The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top."""


class SegformerModel(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel - JC - haven't tried to use this yet, but we extracted the functions at the top'
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


# From Pytorch source code
# https://pytorch.org/vision/stable/_modules/torchvision/ops/misc.html#SqueezeExcitation
class SELayer(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels,
        squeeze_channels,
    ):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = torch.nn.ReLU()
        self.scale_activation = torch.nn.Sigmoid()

    # Get channel-wise modulation weights
    def _scale(self, x):
        scale = self.avgpool(x)  # dim[b,c,1,1]
        scale = self.fc1(scale)  # dim[b,c/reduction,1,1]
        scale = self.activation(scale)
        scale = self.fc2(scale)  # dim[b,c,1,1]
        return self.scale_activation(scale)

    # Apply scaling to input
    def forward(self, x):
        scale = self._scale(x)
        return (
            scale * x,
            scale,
        )  # JC - now returning both the scaled output and the scale itself


# 20230228 - JC - Added to incorporate upsampling squentially as part of the classifier step for alphamix.py
class Interpolate(nn.Module):
    def __init__(self, size, mode, align):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align = align

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align)
        return x


class SegformerDecodeHead(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        self.config = config
        self.opt = opt
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(
                config, input_dim=config.hidden_sizes[i]
            )  # output_dim = config.decoder_hidden_size
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        # JC - added, not part of original Segformer model
        self.squeeze_excitation = SELayer(
            input_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            squeeze_channels=config.decoder_hidden_size,
        )

        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.upsample_hw = Interpolate(
            [self.config.image_size, self.config.image_size],
            "bilinear",
            (
                self.config.align_corners
                if hasattr(self.config, "align_corners")
                else False
            ),
        )

        self.classifier = nn.Conv2d(
            config.decoder_hidden_size, config.num_labels, kernel_size=1
        )

        # self.compress_features = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        # self.classifier = nn.Sequential(self.compress_features, self.upsample_hw)

        # JC -added, not part of original Segformer model. dim=1, for each pixel a prob. distribution over channels
        # 20230419, returning to optional BatchNorm (present in 22-11-08__16-13-49)
        self.final_norm = (
            nn.BatchNorm2d(config.num_labels) if opt.dec_final_norm else nn.Identity()
        )

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]  # tuple with dim[nblocks]

        all_hidden_states = ()
        for blk, (encoder_hidden_state, mlp) in enumerate(
            zip(encoder_hidden_states, self.linear_c)
        ):

            if (
                self.config.reshape_last_stage is False
                and encoder_hidden_state.ndim == 3
            ):
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1).contiguous()
            # convert 1D hidden state images into 2D
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )

            # upsample - all encoder hidden states now have config.decoder_hidden_size num channels, resample [h,w] to
            # largest features map size
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=(
                    self.config.align_corners
                    if hasattr(self.config, "align_corners")
                    else False
                ),
            )

            all_hidden_states += (encoder_hidden_state,)

        # Added as an additional step prior to the linear fuse to try an emphasize informative feature channels
        # and suppress unimportant feature channels
        stackstates = torch.cat(all_hidden_states[::-1], dim=1)

        # Decoder layers are here
        hidden_states, scale = self.squeeze_excitation(stackstates)
        hidden_states = self.linear_fuse(hidden_states)
        hidden_states = self.batch_norm(
            hidden_states
        )  # 0 mean, unit variance over batch
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # ^ dim[cfg.num_encoder_blocks] with each entry being dim[opt.batch_sz, cfg.hidden_sizes[-1], height/4, width/4]
        #                                                     dim[16,256,64,64]

        # Compress along channel dimension
        # Go from config.decoder_hidden_size >> config.num_labels channels
        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        # New as of 20221108 ... this actual creates wider spread in valid/train metrics & loss
        # - repeating norm as we're doing the same thing as above in the linear_fuse step
        # - preliminary decoder training results over small batch says it's better
        # BatchNorm at final layer, output of the classifer - if specified. Otherwise Identity.
        logits = self.final_norm(logits)  # batch norm over channels (classes)

        ###!???????!###
        # Should this be stackstates or hidden_states??
        return logits, stackstates, scale


"""Alternate decoder, up-path from a UNET"""


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        block = []
        # Use batchnorm after convolution but before activation to reduce dependency on bias from convolution
        block.append(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1, bias=False)
        )
        block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(
            nn.Conv2d(
                out_size, out_size, kernel_size=3, padding=1, stride=1, bias=False
            )
        )
        block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    # in_size: number of feature channels from current block (going in descending order)
    # out_size: number of feature channels in the prior block (blk-1)
    def __init__(self, config, in_size, out_size, scaleby, bridge_size=None):
        super().__init__()
        # Increase feature map size, reduce number of channels
        self.up = nn.Sequential(
            nn.Upsample(
                mode="bilinear",
                scale_factor=scaleby,
                align_corners=(
                    config.align_corners if hasattr(config, "align_corners") else False
                ),
            ),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1),
        )
        # Convolution of upsampled input and bridge feature maps
        if bridge_size is None:
            bridge_size = 2 * out_size
        self.conv_block = UNetConvBlock(bridge_size, out_size)

    def forward(self, x, bridge):
        # x:        current feature map that we need to usample to match the size of "bridge"
        # bridge:   feature map from prior encoder block
        x = self.up(x)
        if bridge is not None:
            x = torch.cat([x, bridge], 1)
        x = self.conv_block(x)
        return x


class UNETDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Have to manually calculate the expected output hidden state sizes from each encoder block
        # Then work backwards to find the ratio of the output size to the first hidden state size
        # These ratios become the kernel and stride for upsampling - the amount of which is unique to each block

        # !!! CAUTION !!! NOW USING THE CONFIG PARAMETER FOR IMAGE SIZE TO GUIDE THIS MODULE LIST. MUST MATCH REALITY!
        # Assumes no dilation as that would increase you effective kernel size
        assert np.all(np.array(self.config.dilation) == 1)
        # Assumes SQUARE kernels and image
        encsizes = []
        self.nblocks = len(
            self.config.patch_sizes
        )  # alternatively could get from config.num_encoder_blocks
        # Find the output size from each encoder block
        for blk in range(self.nblocks):
            s = self.config.strides[blk]
            p = self.config.square_pad[blk]
            k = self.config.patch_sizes[blk]
            if blk == 0:
                insz = self.config.image_size
            # outsz = int(((insz-k+2*p)/s)+1)
            outsz = np.floor(((insz - k + 2 * p) / s) + 1)
            encsizes.append(outsz)
            # Set the input to the next block to be the prior output size
            insz = outsz
        # Store these calculated encoder output sizes
        self.encsizes = np.array(encsizes, dtype=int)

        # Find the downsampling ratio from each block, use these values for kernel size and stride to upsample
        # relative to the next block (in ascending order) - alternatively could pull this from config.downsampling_rates
        self.hwratios = []
        for blkx, encsz in enumerate(self.encsizes):
            if blkx == 0:
                ratio = int(self.config.image_size / encsz)
            else:
                ratio = int(self.encsizes[blkx - 1] / encsz)
            self.hwratios.append(
                ratio
            )  # should ideally match self.config.downsample_rates

        self.final_hidden_size = self.config.num_labels * 2

        # Gradually reduce the number of channels as you upsample the feature maps and concatenate
        # Going in REVERSE block order
        self.up_path = nn.ModuleList()
        for blkx, szratio in reversed(list(enumerate(self.hwratios))):
            inputch = self.config.hidden_sizes[blkx]
            if blkx != 0:
                outputch = self.config.hidden_sizes[blkx - 1]
                upblock = UNetUpBlock(config, inputch, outputch, szratio)
            else:
                outputch = self.final_hidden_size
                upblock = UNetUpBlock(
                    config, inputch, outputch, szratio, bridge_size=outputch
                )  # no features to concatenate
            self.up_path.append(upblock)

        # Single (channel-wise) convolutional step for classification
        self.classifier = nn.Conv2d(
            self.final_hidden_size, self.config.num_labels, kernel_size=1
        )

    def forward(self, allstates):
        showdims = False
        # NOTE: up_path is expecting to go in REVERSE order (enc block 4 >> 1)
        lastblk = len(self.up_path) - 1
        for idx, upblock in enumerate(self.up_path):
            blkx = self.nblocks - idx - 1
            # Start with the last hidden state and the one prior, cat and pass through upblock
            if blkx == lastblk:
                x = allstates[blkx]
            # When back to the first block, there's nothing to cat, will just compress channel dim
            if blkx == 0:
                bridge = None
            else:
                bridge = allstates[blkx - 1]
            # Pass the conv state with the hidden state of the prior block
            x = upblock(x, bridge)  # combined, upsampled state
            if showdims:
                print(f" {x.shape} - input size, up_path step {idx}")

        logits = self.classifier(x)
        if showdims:
            print(f" {logits.shape} - Final (classifier) output size")

        return logits, None, None


"""Full Segformer Model """


class SegformerForSemanticSegmentation(nn.Module):

    def __init__(self, config, opt):
        super().__init__()
        # Instantiate encoder and decoder from model config
        self.config = config
        self.opt = opt
        self.segformer = SegformerModel(config)
        # Read the preferred decoder option from the user input args
        if opt.decoder == "segformer":
            self.decode_head = SegformerDecodeHead(config, opt)
        elif opt.decoder == "unet":
            self.decode_head = UNETDecodeHead(config)
        else:
            raise ValueError(f"Cannot instantiate decoder {opt.decoder}, not found")
        self.train_enc = opt.train_enc  # Note if we're retraining the encoder

    def forward(
        self,
        pixel_values,
        mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=False,
        output_dec_states=False,
        output_scale=False,
        return_dict=None,
        crop=False,
    ):
        """
        Forward call of the full Segformer model (encoder+decoder)

        Parameters
        ----------
        pixel_values : torch tensor
            Input to the model, will be either the image or the patched image with dim[b,c,h,w]
        mask : torch tensor
            Binary mask (for training encoder), indicating which pixels were masked (1) in the image
        labels : torch tensor
            This is our ground truth, = labels if retraining the decoder, **OR** = true image for training the encoder
        output_attentions:
        output_hidden_states :
        return_dict : boolean

        Returns
        -------


        """

        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # Encoder call
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # Decoder call
        logits, alldecstates, SEscale = self.decode_head(encoder_hidden_states)

        # alldecstates, SEscale, upsampled_logits = self.decode_head(encoder_hidden_states)  # new

        # upsample logits to the images' original size in the case of segformer decoder - MOVED to within decode_head
        if self.opt.decoder == "segformer":
            upsampled_logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=(
                    self.config.align_corners
                    if hasattr(self.config, "align_corners")
                    else False
                ),
            )
            # ^ dim[nbatch, nclass, h, w]

        elif self.opt.decoder == "unet":
            upsampled_logits = logits
        else:
            raise ValueError(
                f"Cannot provide output for unkown decoder {self.opt.decoder}"
            )

        # Initialize
        pred = None
        loss = None

        # LOSS calculation & PREDICTION
        if self.train_enc:
            if self.opt.loss_fn == "L1":
                loss_fct = L1Loss()
            elif self.opt.loss_fn == "MSE":
                loss_fct = MSELoss()
            else:
                loss_fct = MSELoss()
            # Remove any padding that was added so that it doesn't influence the loss
            if crop:
                b, c, h, w = upsampled_logits.size()
                upsampled_logits = T.CenterCrop(h - crop)(upsampled_logits)
                labels = T.CenterCrop(h - crop)(labels)
            loss = loss_fct(upsampled_logits.float(), labels.float())
        else:
            if labels is not None:
                if self.config.num_labels == 1:
                    raise ValueError("The number of labels should be greater than one")
                else:
                    loss_fct = CrossEntropyLoss(
                        ignore_index=self.config.semantic_loss_ignore_index
                    )
                    # NOTE: upsampled_logits do not need to be strictly + or [0,1], torch expects to be unnormalized
                    loss = loss_fct(upsampled_logits.float(), labels.long())
            # Segmentation prediction. For each pixel, find the max (int) value along the channel dimension,
            # this becomes the class of that pixel.
            pred = upsampled_logits.argmax(dim=1)  # will be dim[b,h,w]

        return SemanticSegmentationModelOutput(
            loss=loss,
            logits=logits,  # dim[b,c,h/R,w/R]
            upsampled_logits=upsampled_logits,  # dim[b,c,h,w]
            segmap=pred,  # dim[b,h,w]
            hidden_states=(
                outputs.hidden_states if output_hidden_states else None
            ),  # encoder hidden states
            attentions=outputs.attentions,
            decstates=(
                alldecstates if output_dec_states else None
            ),  # upsampled hidden states, input to decoder
            scale=SEscale if output_scale else None,  # SE layer scaling per channel
        )
