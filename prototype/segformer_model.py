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
"""PyTorch SegFormer model with ARiA additions"""


# from typing import List, Set, Tuple

# import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torchvision import transforms as T

from prototype.modeling_outputs import (
    BaseModelOutput,
    SemanticSegmentationModelOutput,
)
from prototype.patch_embed import PatchEmbed


class Segformer(nn.Module):
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

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
                align_corners=(self.config.align_corners if hasattr(self.config, "align_corners") else False),
            )
            # ^ dim[nbatch, nclass, h, w]

        elif self.opt.decoder == "unet":
            upsampled_logits = logits
        else:
            raise ValueError(f"Cannot provide output for unkown decoder {self.opt.decoder}")

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
                    loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
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
            hidden_states=(outputs.hidden_states if output_hidden_states else None),  # encoder hidden states
            attentions=outputs.attentions,
            decstates=(alldecstates if output_dec_states else None),  # upsampled hidden states, input to decoder
            scale=SEscale if output_scale else None,  # SE layer scaling per channel
        )


class SegformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel - JC - haven't tried to use this yet, but we extracted the functions at the bottom'
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

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


class SegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PatchEmbed(
                    image_size=config.image_size // config.downsampling_rates[i],
                    patch_size=config.patch_sizes[i],
                    in_channels=(config.num_channels if i == 0 else config.hidden_sizes[i - 1]),
                    embed_dim=config.hidden_sizes[i],
                    stride=config.strides[i],
                    pad=(config.square_pad[i] if hasattr(config, "square_pad") else config.patch_sizes[i] // 2),
                    padding_mode=(config.pad_method[i] if hasattr(config, "pad_method") else "zeros"),
                    dilate=config.dilation[i] if hasattr(config, "dilation") else 1,
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
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
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
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]  # context layer
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)  # attention probs
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegformerDecodeHead(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        self.config = config
        self.opt = opt
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])  # output_dim = config.decoder_hidden_size
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
            (self.config.align_corners if hasattr(self.config, "align_corners") else False),
        )

        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        # self.compress_features = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        # self.classifier = nn.Sequential(self.compress_features, self.upsample_hw)

        # JC -added, not part of original Segformer model. dim=1, for each pixel a prob. distribution over channels
        # 20230419, returning to optional BatchNorm (present in 22-11-08__16-13-49)
        self.final_norm = nn.BatchNorm2d(config.num_labels) if opt.dec_final_norm else nn.Identity()

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]  # tuple with dim[nblocks]

        all_hidden_states = ()
        for blk, (encoder_hidden_state, mlp) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1).contiguous()
            # convert 1D hidden state images into 2D
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)

            # upsample - all encoder hidden states now have config.decoder_hidden_size num channels, resample [h,w] to
            # largest features map size
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=(self.config.align_corners if hasattr(self.config, "align_corners") else False),
            )

            all_hidden_states += (encoder_hidden_state,)

        # Added as an additional step prior to the linear fuse to try an emphasize informative feature channels
        # and suppress unimportant feature channels
        stackstates = torch.cat(all_hidden_states[::-1], dim=1)

        # Decoder layers are here
        hidden_states, scale = self.squeeze_excitation(stackstates)
        hidden_states = self.linear_fuse(hidden_states)
        hidden_states = self.batch_norm(hidden_states)  # 0 mean, unit variance over batch
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
