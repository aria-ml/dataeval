# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SegFormer model with ARiA additions """


# from typing import List, Set, Tuple

# import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torchvision import transforms as T

from daml._prototype.utils.modeling_outputs import (
    BaseModelOutput,
    SemanticSegmentationModelOutput,
)
from daml._prototype.utils.patch_embed import PatchEmbed


class Segformer(nn.Module):
    """
    This implements the ARiA adjusted Segformer model.
    See https://github.com/NVlabs/SegFormer/tree/master for the original.

    Args:


    Inputs:


    Outputs:

    """

    def __init__(self, config, opt):
        super().__init__()
        # Instantiate encoder and decoder from model config
        self.config = config
        self.opt = opt
        self.segformer = SegformerModel(config)
        # Read the preferred decoder option from the user input args
        if opt.decoder == "segformer":
            self.decode_head = SegformerDecodeHead(config, opt)
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


class SegformerModel(nn.Module):
    """
    This implements the ARiA adjusted Segformer model.
    See https://github.com/NVlabs/SegFormer/tree/master for the original.

    Args:


    Inputs:


    Outputs:

    """

    def __init__(self, config):

        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

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
