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
