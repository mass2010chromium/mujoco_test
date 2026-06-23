#!/usr/bin/env python3
# File written by claude opus 4.8
"""
Isolate the jepa-wms visual encoder and run it on a single image.

This reproduces the *observation-encoding* path of `EncPredWM.encode`
(== `VideoWM.encode_obs` + preprocessing) but calls the underlying encoder
module directly, so you can see exactly how a raw RGB frame becomes a latent
patch grid.

Pipeline (from the repo source):
    raw pixels (B,T,C,H,W) in [0,255]
      -> / 255
      -> preprocessor.transform        (resize to img_size + ImageNet normalize)
      -> dup/batchify/tubelet reshape   (enc_type-specific)
      -> encoder(x)                     (the ViT: V-JEPA-2 or DINOv2/3)
      -> rearrange to latent grid (B,T,1,Hp,Wp,D)

Note: the released JEPA-WM checkpoints (metaworld/pusht/pointmaze/wall/droid)
use enc_type='dino'. The true V-JEPA encoder (enc_type='vjepa', V-JEPA-2
ViT-G/16) lives in the 'vjepa2_ac_droid'/'vjepa2_ac_oss' baselines. This script
adapts to whichever you load.

Requires: torch, torchvision, einops, numpy, pillow, huggingface_hub
Usage:
    python encode_image.py path/to/image.png
    python encode_image.py path/to/image.png --model jepa_wm_pusht --device cpu
"""

import argparse

import numpy as np
import torch
from einops import rearrange
from PIL import Image

# The following code is not used; just keep as personal reference as now
# Claude copied out the relevant encoding code from the repository.... but we can just call model.encode()
def to_encoder_input(visual, model, preprocessor):
    """Replicate the pre-encoder visual pipeline of EncPredWM.encode.

    Args:
        visual: float tensor (B, T, C, H, W) with pixel values in [0, 255].
    Returns:
        Tensor shaped for model.model.encoder, plus (b, t) for the output reshape.
    """
    wm = model.model  # the inner VideoWM
    b, t, c, h, w = visual.shape
    x = visual / 255.0
    x = preprocessor.transform(x)  # resize + ImageNet-normalize -> (B,T,C,img,img)

    if wm.batchify_video:
        x = rearrange(x, "b t ... -> (b t) ...")
    if wm.dup_image:
        if not wm.batchify_video:
            x = x.repeat_interleave(2, dim=1)            # (B, 2T, C, H, W)
        else:
            x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)     # (B*T, C, 2, H, W)
    elif wm.enc_type == "vjepa":
        # a single frame fed to a temporal model needs tubelet_size_enc frames
        x = x.repeat(1, wm.tubelet_size_enc, 1, 1, 1)    # (B, tubelet, C, H, W)

    if wm.enc_type == "vjepa" and not wm.batchify_video:
        x = rearrange(x, "b t c h w -> b c t h w")       # V-JEPA wants (B,C,T,H,W)
    return x, b, t


def to_latent_grid(tokens, model, b, t):
    """Reshape encoder patch tokens into the (B, T, 1, Hp, Wp, D) latent grid."""
    wm = model.model
    g = wm.grid_size
    if wm.enc_type == "dino" or wm.batchify_video:
        return rearrange(tokens, "(b t) (h w) d -> b t 1 h w d", b=b, t=t, h=g, w=g)
    return rearrange(tokens, "b (t h w) d -> b t 1 h w d", h=g, w=g)
# END dead code


def load_jepawm(model_name: str = "vjepa2_ac_droid", device: str = "cuda:0"):
    """
    Load a frame encoder from the jepawm set.
    Either loads v-jepa 2, or dino v2/v3.

    Parameters:
    --------------------------
    model_name:     torch.hub entry, e.g. vjepa2_ac_droid for V-JEPA encoder,
                        or jepa_wm_pusht for DINOv2 encoder
    device:         cuda device
    """
    device = torch.device(device)

    # 1. Load (model, preprocessor); the encoder is model.model.encoder
    model, preprocessor = torch.hub.load(
        "facebookresearch/jepa-wms", model_name, device=str(device)
    )
    model.eval()
    encoder = model.model.encoder  # <-- the isolated ViT visual encoder
    print(f"Loaded '{model_name}': enc_type={model.model.enc_type}, "
          f"encoder={type(encoder).__name__}, grid_size={model.model.grid_size}")

    def encode(image_chw_rgb):
        """
        Note: Expects a [C, H, W] RGB image, in pytorch tensor convention.
        """

        # 2. Image -> (B=1, T=1, C=3, H, W) float tensor in [0, 255]
        visual = rearrange(torch.tensor(image_chw_rgb).float(), "c h w -> 1 1 c h w").to(device)

        with torch.no_grad():
            return model.encode(visual)
    return model


if __name__ == "__main__":
    load_jepawm()
    print("Load success")
