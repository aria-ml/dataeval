# Copyright (c) ARiA. All rights reserved.
#
# Licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# Modified code from:
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/train.py
#   https://github.com/facebookresearch/dino/blob/main/main_dino.py

# PyTorch 1.12 sets this to False by default
torch.backends.cuda.matmul.allow_tf32 = True


def do_train(cfg, model):
    model.train()

    # setup optimizer

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = 0
    for batch in data_loader:
        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        if cfg.optim.clip_grad:
            for v in model.student.values():
                v.clip_grad_norm_(cfg.optim.clip_grad)
        optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)


def backprop_loss(self, loss):
    if self.fp16_scaler is not None:
        self.fp16_scaler.scale(loss).backward()
    else:
        loss.backward()


def forward_backward(self, images, teacher_temp):
    n_global_crops = 2
    assert n_global_crops == 2
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

    do_dino = self.do_dino
    do_ibot = self.do_ibot

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
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
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
        elif do_ibot and self.ibot_separate_head:
            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                upperbound, _dim
            )
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
                buffer_tensor_teacher
            )[:n_masked_patches]
        else:
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_ibot_softmaxed_centered = None

        if self.cfg.train.centering == "centering":
            teacher_dino_softmaxed_centered_list = (
                self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(
                    n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:]
                )
            )
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            if do_ibot:
                masked_teacher_patch_tokens_after_head = (
                    masked_teacher_patch_tokens_after_head.unsqueeze(0)
                )
                masked_teacher_ibot_softmaxed_centered = (
                    self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches],
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
                    n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:]
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
    reshard_fsdp_model(self.teacher)

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
    student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
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
                ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list
            )
        )
        if not self.ibot_separate_head:
            inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
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
        student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[
            :n_masked_patches
        ]

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

    self.backprop_loss(loss_accumulator)

    self.fsdp_synchronize_streams()

    return loss_dict


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
