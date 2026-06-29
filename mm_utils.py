# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared helpers for the mm demo scripts (demo_colmap_mm.py, demo_pairwise_mm.py):
seeding, device/dtype selection, model loading and image loading.
"""

import glob
import os
import random

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square

MODEL_REPOS = {
    "1b": "facebook/VGGT-1B",
    "1b-commercial": "facebook/VGGT-1B-Commercial",
}

# Local checkpoints live next to this file in checkpoints/<name>/.
# If present they are used directly (no Hugging Face download/cache lookup).
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 needs compute capability >= 8 (Ampere or newer)
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    return device, dtype


def load_model(name, device, checkpoint=None):
    if checkpoint is not None:
        # Explicit checkpoint folder wins; fail loudly if it's not a real one.
        if not os.path.isfile(os.path.join(checkpoint, "model.safetensors")):
            raise FileNotFoundError(
                f"--checkpoint {checkpoint!r} has no model.safetensors"
            )
        source = checkpoint
    else:
        local_dir = os.path.join(CHECKPOINTS_DIR, name)
        if os.path.isfile(os.path.join(local_dir, "model.safetensors")):
            source = local_dir
        else:
            source = MODEL_REPOS[name]
    print(f"Loading model from {source}")
    model = VGGT.from_pretrained(source)
    model.eval()
    model = model.to(device)
    print("Model loaded")
    return model


def list_images(image_dir):
    image_path_list = sorted(
        p for p in glob.glob(os.path.join(image_dir, "*"))
        if p.lower().endswith(IMAGE_EXTS)
    )
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    return image_path_list


def load_images(image_dir, size, device):
    image_path_list = list_images(image_dir)
    images, original_coords = load_and_preprocess_images_square(image_path_list, size)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")
    return images, original_coords, image_path_list
