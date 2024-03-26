# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Only includes those activations which cannot be currently found
# in PyTorch 2.0, otherwise use the PyTorch implementation

import math

import torch
from torch import Tensor, nn


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo
    (identical to OpenAI GPT). Also see the Gaussian Error Linear Units paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate.
    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate.
    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially
    useful for quantization purpose, as it allows mapping negatives values in the
    GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function
    in Google Bert repo when initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly
    different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))).
    See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min_val: float, max_val: float):
        if min_val > max_val:
            raise ValueError(f"min should be < max (got min:{min_val}, max:{max_val})")

        super().__init__()
        self.min = min_val
        self.max = max_val

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(nn.functional.gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than
    QuickGELU. See: https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    self.precomputed_constant * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an
    attention activation. See https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(
        self, x: Tensor, mu: float = 0.707107, sigma: float = 0.282095
    ) -> Tensor:
        x = (x - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(x))


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, x: Tensor) -> Tensor:
        relu_applied = nn.functional.relu(x)
        squared = torch.square(relu_applied)
        return squared


ACT2FN = {
    "gelu_10": ClippedGELUActivation(-10, 10),
    "gelu_fast": FastGELUActivation(),
    "gelu_new": NewGELUActivation(),
    "gelu_accurate": AccurateGELUActivation,
    "quick_gelu": QuickGELUActivation(),
    "laplace": LaplaceActivation,
    "linear": LinearActivation,
    "relu2": ReLUSquaredActivation,
}
