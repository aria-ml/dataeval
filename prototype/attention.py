# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT Liscence
# found in the LICENSE file in the root directory of this source tree.
#
# References:
#   https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py


import math
from typing import List, Set, Tuple

import torch
from torch import nn


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


class SegformerSelfOutput(nn.Module):
    def __init__(self, embed_dim, drop):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerEfficientSelfAttention(nn.Module):
    def __init__(self, attn_dropout, embed_dim, num_attn_heads, sr_ratio):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads

        if self.embed_dim % self.num_attn_heads != 0:
            raise ValueError(
                f"The hidden size ({self.embed_dim}) is not a multiple of the number of attention "
                f"heads ({self.num_attn_heads})"
            )

        self.attention_head_size = int(self.embed_dim / self.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attention_head_size

        self.query = nn.Linear(self.embed_dim, self.all_head_size)
        self.key = nn.Linear(self.embed_dim, self.all_head_size)
        self.value = nn.Linear(self.embed_dim, self.all_head_size)

        self.dropout = nn.Dropout(
            attn_dropout
        )  # randomly 0's elements, scales by 1/(1-p)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.layer_norm = nn.LayerNorm(embed_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attn_heads,
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


class SegformerAttention(nn.Module):
    def __init__(
        self, embed_dim, num_attn_heads, attn_dropout, dropout, sr_ratio
    ) -> None:
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            attn_dropout=attn_dropout,
            embed_dim=embed_dim,
            num_attn_heads=num_attn_heads,
            sr_ratio=sr_ratio,
        )
        self.output = SegformerSelfOutput(embed_dim=embed_dim, drop=dropout)
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

    def forward(self, embed_dim, height, width, output_attentions=False):
        self_outputs = self.self(
            embed_dim, height, width, output_attentions
        )  # hidden_states = patch_embeddings
        # ^ Self attention output: (context_layer, attention_probs)

        attention_output = self.output(
            self_outputs[0], embed_dim
        )  # dim[nbatch, (h*w)/R^2, nchann], nchann: ref segformer Table 6
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # (context layer, attention probs)
        return outputs
