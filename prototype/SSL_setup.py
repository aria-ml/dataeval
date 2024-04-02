# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py
from functools import partial

import torch
from torch import nn

from daml._prototype.utils.encoder import SegformerEncoder


class SSLNetwork(nn.Module):
    """
    This
    """

    def __init__(
        self,
        cfg,
        encoder_kwargs,
        grad_scaler,
        final_out,
        dino_loss_weight,
        koleo_loss_weight,
        ibot_loss_weight,
    ):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler)

        self.student_backbone = SegformerEncoder(**encoder_kwargs)
        self.teacher_backbone = SegformerEncoder(**encoder_kwargs)
        self.embed_dim = encoder_kwargs["embed_dim"]

        self.out_dim = final_out

        self.dino_loss_weight = dino_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.ibot_loss_weight = ibot_loss_weight

        if self.dino_loss_weight:
            self.student_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.teacher_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.out_dim)
            if self.koleo_loss_weight:
                self.koleo_loss = KoLeoLoss()

        if self.ibot_loss_weight:
            assert (
                max(cfg.ibot.mask_ratio_min_max) > 0
            ), "please provide a positive mask ratio tuple for ibot"
            assert (
                cfg.ibot.mask_sample_probability > 0
            ), "please provide a positive mask probability for ibot"
            self.ibot_patch_loss = iBOTPatchLoss(self.out_dim)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, augmentation):
        """
        Need to adjust this such that it is augmentation invariant.

        """

        if augmentation == "crop":
            n_global_crops = 2
            n_local_crops = self.cfg.crops.local_crops_number

            global_crops = images["collated_global_crops"].cuda(non_blocking=True)
            local_crops = images["collated_local_crops"].cuda(non_blocking=True)

            masks = images["collated_masks"].cuda(non_blocking=True)
            mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
            n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
            n_masked_patches = mask_indices_list.shape[0]
            upperbound = images["upperbound"]
            masks_weight = images["masks_weight"].cuda(non_blocking=True)

            n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
            n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.dino_loss_weight
        do_ibot = self.ibot_loss_weight

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat(
                (teacher_cls_tokens[1], teacher_cls_tokens[0])
            )
            ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens, _dim
                )
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = (
                        masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_after_head[
                                :, :n_masked_patches
                            ],
                            teacher_temp=teacher_temp,
                        )
                    )
                    masked_teacher_ibot_softmaxed_centered = (
                        masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    )
                    self.ibot_patch_loss.update_center(
                        masked_teacher_patch_tokens_after_head[:n_masked_patches]
                    )

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = (
                    self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:],
                    )
                )

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = (
                        self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            masked_teacher_patch_tokens_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor=n_masked_patches_tensor,
                        )
                    )

            else:
                raise NotImplementedError

            return (
                teacher_dino_softmaxed_centered_list,
                masked_teacher_ibot_softmaxed_centered,
            )

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            get_teacher_output()
        )

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # !!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!! Need this
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=cfg.compute_precision.grad_scaler,
        ):
            output = model(input)
            loss = loss_fn(output, target)

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = self.teacher.dino_head._streams = (
                self.student.backbone._streams
            ) = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(
                    get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])
                ):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups
