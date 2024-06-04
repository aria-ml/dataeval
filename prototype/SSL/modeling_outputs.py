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

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple

import torch


# Pulled out of file_utils.py from transformers library and modified
class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows
    indexing by integer or slice (like a tuple) or strings (like a dictionary) that will
    ignore the `None` attributes. Otherwise behaves like a regular python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the
    [`~file_utils.ModelOutput.to_tuple`] method to convert it to a tuple before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)  # type: ignore

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more \
                    than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none:  # and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a
            # (key, value) iterator set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or len(element) != 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state:
            (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`)
            Sequence of hidden-states at the output of the last layer of the model.

        hidden_states:
            (`tuple(torch.Tensor)`, *optional*, returned when passed
                `output_hidden_states=True` or when `config.output_hidden_states=True`)
            Tuple of torch.Tensor of shape `(batch_size, sequence_length, hidden_size)`.
            (one for the output of the embeddings + one for the output of each layer)

            Hidden-states of the model at the output of each layer plus the initial
            embedding outputs.

        attentions:
            (`tuple(torch.Tensor)`, *optional*, returned when passed
                `output_attentions=True` or when `config.output_attentions=True`)
            Tuple of `torch.Tensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """

    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    register_tokens: Optional[Tuple[torch.Tensor]] = None


@dataclass
class SemanticSegmentationModelOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss:
            (`torch.Tensor` of shape `(1,)`, *optional*,
                returned when `labels` is provided)
            Classification (or regression if config.num_labels==1) loss.

        logits:
            (`torch.Tensor` of shape
                `(batch_size, config.num_labels, logits_height, logits_width)`)
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the
            `pixel_values` passed as inputs. This is to avoid doing two interpolations
            and lose some quality when a user needs to resize the logits to the original
            image size as post-processing. You should always check your logits shape and
            resize as needed.

            </Tip>

        upsampled_logits:
            (`torch.Tensor` of shape
                `(batch_size, config.num_labels, logits_height, logits_width)`)
            Same logits as above, upsampled using the interpolation method within the
            model. This is done to ensure conistency in the output so that it returns
            what is being used in calculating the loss.

        segmap:
            (`torch.Tensor` of shape `(batch_size, logits_height, logits_width)`)
            Prediction of the model, semantic segmentation output with a single class
            label (int) per pixel

        hidden_states:
            (`tuple(torch.Tensor)`, *optional*, returned when passed
                `output_hidden_states=True` or when `config.output_hidden_states=True`)
            Tuple of `torch.Tensor` of shape `(batch_size, patch_size, hidden_size)`.
            (one for the output of the embeddings + one for the output of each layer)

            Hidden-states of the model at the output of each layer plus the initial
            embedding outputs.

        attentions:
            (`tuple(torch.Tensor)`, *optional*, returned when passed
                `output_attentions=True` or when `config.output_attentions=True`)
            Tuple of `torch.Tensor` (one for each layer) of shape
            `(batch_size, num_heads, patch_size, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    upsampled_logits: Optional[torch.Tensor] = None
    segmap: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    decstates: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
