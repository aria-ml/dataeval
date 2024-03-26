# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py


import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EfficientSelfAttention(nn.Module):
    """
    CONCEPT OVERVIEW:
    ----------------
    Query:      flattened, transformed patch embeddings - our sequence -
                where we want to look - dim[b, a, (h*w)/R^2, c)]
    Key:        compressed, transformed, flattened patch embeddings - our available
                data - where we want to compare - dim[b, a, sr(h)*sr(w), c]
    Attention:  similarity between our sequence and the data that's available -
                matmul(Q,K) - dim[b, a, (h*w)/R^2, sr(h)*sr(w)]
    Value:      compressed, transformed, flattened patch embeddings - our available
                data - that will be weighted by the attention scores to provide the
                data that matters most to us - dim[b, a, sr(h)*sr(w), c]
    Context:    the weighted data that matters most to us, this is passed along in
                the forward call - dim[b, a, (h*w)/R^2, c]

    b:          number of batches
    a:          number of attention heads
    c:          number of channels
    h:          embdding height
    w:          embdding width
    R:          downsample factor

    Args:
        attn_dropout:       Probability of dropout in the attention layer, float [0,1].
        embed_dim:          Size of the embedding dimension, D - int.
        num_heads:          Number of attention heads, int.
        sr_ratio:           Value to reduce the dimensions for self-attention by,
                             int [1, embed_dim] (sequence reduction ratio).
        drop:               Probability of dropout after attention, float [0,1].

    Input:
        patch_embedding:    Input Tensor in shape (B,N,D)
        height:             Embdding height, int
        width:              Embdding width, int
        output_attentions:  Boolean if you want to output attention probabilities

    Output:
        outputs:            Tuple of either just the attended Tensor or the attended
                             Tensor and the attention probabilities
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % self.num_heads == 0, ValueError(
            f"The hidden dimension ({self.embed_dim}) is not a multiple of the number \
            of attention heads ({self.num_heads})"
        )

        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(
            attn_dropout
        )  # randomly 0's elements, scales by 1/(1-p)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.layer_norm = nn.LayerNorm(embed_dim)

        self.context = nn.Linear(self.embed_dim, self.embed_dim)
        self.context_dropout = nn.Dropout(drop)

    def forward(
        self,
        patch_embedding: Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor]:
        B, N, D = patch_embedding.shape
        query_layer = (
            self.query(patch_embedding)
            .reshape(B, N, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            adjusted_embedding = patch_embedding.permute(0, 2, 1).reshape(
                B, D, height, width
            )
            adjusted_embedding = self.sr(
                adjusted_embedding
            )  # compress along spatial dimension [h,w] using Conv2d
            adjusted_embedding = adjusted_embedding.reshape(B, D, -1).permute(
                0, 2, 1
            )  # re-flatten [h,w] and flip channels (D) to last dimension
            adjusted_embedding = self.layer_norm(adjusted_embedding)
            # Transform patch embeddings (embed_dim) through SEPARATE linear layers
            # and transpose to get keys, values
            key_layer = (
                self.key(adjusted_embedding)
                .reshape(B, -1, self.num_heads, D // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            value_layer = (
                self.value(adjusted_embedding)
                .reshape(B, -1, self.num_heads, D // self.num_heads)
                .permute(0, 2, 1, 3)
            )
        else:
            # Transform patch embeddings (embed_dim) through SEPARATE linear layers
            # and transpose to get keys, values
            key_layer = (
                self.key(patch_embedding)
                .reshape(B, N, self.num_heads, D // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            value_layer = (
                self.value(patch_embedding)
                .reshape(B, N, self.num_heads, D // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            # ^ NOTICE: key and value layer are the same.
            # ergo, self attention, not cross attention.

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Take the dot product between attention probability and "value"
        # to get the attended embeddings.
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).reshape(B, N, D)

        context_layer = self.context(context_layer)
        context_layer = self.context_dropout(context_layer)

        # Allowing for visualization of the attention probabilities
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
