# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, xavier_uniform_
import os
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim, # 256
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    # mobile_sam.eval()
    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        # with open(checkpoint, "rb") as f:
        #     state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam


def build_sam_wxr_vit(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim, # 256
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=2,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=1024,
                    num_heads=4,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=128,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        state_dict_filtered = {k: v for k, v in state_dict.items() if ("mask_decoder") not in k}

        mobile_sam.load_state_dict(state_dict_filtered, strict=False)
        

def build_tiny_msam(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    if os.environ['INFERENCE_MODE'] == "test":
        mobile_sam = Sam(
                image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                    embed_dims=[64, 96, 128, 320],
                    depths=[2, 2, 4, 2],
                    num_heads=[2, 4, 4, 8],
                    window_sizes=[7, 7, 14, 7],
                    mlp_ratio=4.,
                    drop_rate=0.,
                    drop_path_rate=0.0,
                    use_checkpoint=False,
                    mbconv_expand_ratio=4.0,
                    local_conv_size=3,
                    layer_lr_decay=0.8
                ),
                prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim, # 256
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
                ),
                mask_decoder=MaskDecoder(
                        num_multimask_outputs=3,
                        transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=prompt_embed_dim,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
                ),
                pixel_mean=[123.675, 116.28, 103.53],
                pixel_std=[58.395, 57.12, 57.375],)
        
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            state_dict_filtered = {k: v for k, v in state_dict.items() if ("mask_decoder") not in k}
        mobile_sam.load_state_dict(state_dict, strict=False)
        mobile_sam.eval()
        return mobile_sam
    
    elif os.environ['INFERENCE_MODE'] == "train":
        mobile_sam = Sam(
                image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                    embed_dims=[64, 96, 128, 320],
                    depths=[2, 2, 4, 2],
                    num_heads=[2, 4, 4, 8],
                    window_sizes=[7, 7, 14, 7],
                    mlp_ratio=4.,
                    drop_rate=0.,
                    drop_path_rate=0.0,
                    use_checkpoint=False,
                    mbconv_expand_ratio=4.0,
                    local_conv_size=3,
                    layer_lr_decay=0.8
                ),
                prompt_encoder=None,
                mask_decoder=None,
                pixel_mean=[123.675, 116.28, 103.53],
                pixel_std=[58.395, 57.12, 57.375],)

        mobile_sam.image_encoder.apply(init_weights)
        return mobile_sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "wxr_vit": build_sam_wxr_vit,
    "tiny_msam": build_tiny_msam,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


