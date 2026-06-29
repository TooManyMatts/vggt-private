# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Pairwise Dump (mm)
=======================

Runs VGGT on a folder of images and writes a DUSt3R-style *pairwise* prediction
dict that can be dropped into the dust3r repo's `demo_mm.py` global-alignment
stage (in place of `output = inference(pairs, model, ...)`).

VGGT predicts everything in a single global frame; DUSt3R's global aligner wants
per-pair pointmaps expressed in view1's camera frame. We bridge that here: for
every ordered pair (i, j) we re-express image i's points in camera i's frame
(`pred1['pts3d']`) and image j's points in camera i's frame
(`pred2['pts3d_in_other_view']`). See OUTPUT_FORMAT.md for the exact contract.

Input:
    examples_mm/a01_ipad_lts_s06_2img/
    ├── image_000001.jpg
    └── image_000002.jpg

Output:
    examples_mm/a01_ipad_lts_s06_2img/pairwise_output.pth   # torch.save'd dict

Example
-------
python demo_pairwise_mm.py --scene_dir examples_mm/a01_ipad_lts_s06 --model 1b --checkpoint checkpoints/1b --output outputs/a01
python demo_pairwise_mm.py --scene_dir examples_mm/a01_ipad_lts_s06 --model 1b-commercial --checkpoint checkpoints/1b-commercial --output outputs/a01

With --output <folder>, the result is saved to <folder>/<model>/pairwise_output.pth.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ipdb import set_trace

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from mm_utils import set_seed, get_device_and_dtype, load_model, load_images
from demo_colmap_mm import run_VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT pairwise (DUSt3R-format) dump")
    parser.add_argument(
        "--scene_dir", type=str, default="examples_mm/a01_ipad_lts_s06_2img",
        help="Directory containing the scene images",
    )
    parser.add_argument(
        "--model", type=str, default="1b", choices=["1b", "1b-commercial"],
        help="Which VGGT checkpoint to use: '1b' (research-only) or '1b-commercial' (gated)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint folder (containing model.safetensors). Overrides --model "
             "lookup; default is checkpoints/<model>/ or the Hugging Face repo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out_h", type=int, default=368, help="Output (DUSt3R) pointmap/image height")
    parser.add_argument("--out_w", type=int, default=512, help="Output (DUSt3R) pointmap/image width")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output base folder. Result is written to <output>/<model>/pairwise_output.pth. "
             "If omitted, falls back to --output_path or <scene_dir>/pairwise_output.pth.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Explicit full path for the .pth dict; overrides --output when set.",
    )
    return parser.parse_args()


def run_vggt_pair(model, image_pair, dtype, resolution=518):
    """Run VGGT on a single image pair [2, 3, H, W].

    VGGT fixes the first camera of each run as the world origin, so the returned
    world points are (up to that convention) already in view1's frame; we still
    re-express them via extrinsic[0] to be robust.
    Returns (world_points [2, H, W, 3], depth_conf [2, H, W], extrinsic [2, 3, 4]).
    """
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, image_pair, dtype, resolution)
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    return world_points, depth_conf, extrinsic


def transform_points_world_to_cam(world_points, extrinsic):
    """Express world-frame points in a camera's frame.

    world_points: [H, W, 3] in the world frame.
    extrinsic:    [3, 4] OpenCV cam-from-world ([R | t]), so X_cam = R @ X_world + t.
    Returns [H, W, 3] in that camera's frame.
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return world_points @ R.T + t


def to_conf(conf):
    """Coerce VGGT depth confidence into the GA's expected scale (>= 1).

    The global aligner applies log(conf) and uses it as a per-pixel weight, so
    values < 1 would become negative weights. Clamp at 1.0: low-confidence
    pixels collapse to ~zero weight rather than fighting the optimization.
    (Revisit if VGGT's conf range warrants a smarter remap.)
    """
    return np.maximum(conf.astype(np.float32), 1.0)


def crop_and_resize_to_dust3r(arr, coords, out_h, out_w, mode="bilinear"):
    """Strip VGGT's square padding and resize a map to the DUSt3R resolution.

    VGGT center-pads each image to a square (black bars on the short side) before
    running, so its 518x518 maps contain a valid sub-rectangle plus padding.
    `coords` is that valid box (x1, y1, x2, y2) in the map's own grid; we crop to
    it (dropping the top/bottom or left/right padding) and resize to (out_h, out_w).

    arr: torch tensor [H, W] or [H, W, C]. Returns [out_h, out_w] or [out_h, out_w, C].
    """
    H, W = arr.shape[:2]
    x1, y1, x2, y2 = (int(round(float(c))) for c in coords)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W), min(y2, H)
    cropped = arr[y1:y2, x1:x2]

    # to NCHW for F.interpolate, then back to the original channel layout
    chw = cropped[None, None] if cropped.ndim == 2 else cropped.permute(2, 0, 1)[None]
    resized = F.interpolate(chw.float(), size=(out_h, out_w), mode=mode, align_corners=False)[0]
    return resized[0] if cropped.ndim == 2 else resized.permute(1, 2, 0)


def build_pairwise_output(model, images, dtype, valid_coords, out_h, out_w, resolution=518):
    """Assemble the DUSt3R-format `output` dict by running VGGT once per pair.

    Emits a symmetrized complete graph: every ordered pair (i, j), i != j. VGGT
    is run *independently* on each 2-image pair (~N*(N-1) forward passes), and
    both pointmaps are expressed in view i's (= view1's) camera frame.
    """
    N = images.shape[0]

    # Images are only used to color the exported cloud. Resize to the pointmap
    # resolution so pixels line up, and normalize [0, 1] -> ~[-1, 1] (DUSt3R's
    # ImgNorm convention).
    imgs = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    imgs = imgs.detach().cpu().float() * 2.0 - 1.0  # [N, 3, resolution, resolution]
    # Crop padding + resize each source image once to the DUSt3R resolution.
    imgs = torch.stack([
        crop_and_resize_to_dust3r(imgs[k].permute(1, 2, 0), valid_coords[k], out_h, out_w).permute(2, 0, 1)
        for k in range(N)
    ])  # [N, 3, out_h, out_w]

    view1_img, view2_img = [], []
    view1_idx, view2_idx = [], []
    pts3d_1_list, pts3d_2_list = [], []
    conf1_list, conf2_list = [], []

    for i in tqdm(range(N)):
        for j in range(N):
            if i == j:
                continue
            # Run VGGT on just this pair; its world frame is camera i's frame.
            world_points, depth_conf, extrinsic = run_vggt_pair(model, images[[i, j]], dtype, resolution)

            # Express both pointmaps in view1's (camera i / first-of-pair) frame.
            pts3d_1 = transform_points_world_to_cam(world_points[0], extrinsic[0])  # img i in cam i
            pts3d_2 = transform_points_world_to_cam(world_points[1], extrinsic[0])  # img j in cam i
            pts3d_1 = torch.from_numpy(np.ascontiguousarray(pts3d_1)).float()
            pts3d_2 = torch.from_numpy(np.ascontiguousarray(pts3d_2)).float()
            conf1 = torch.from_numpy(to_conf(depth_conf[0])).float()
            conf2 = torch.from_numpy(to_conf(depth_conf[1])).float()

            # Crop padding + resize to the DUSt3R resolution (image i for view1, j for view2).
            pts3d_1 = crop_and_resize_to_dust3r(pts3d_1, valid_coords[i], out_h, out_w)
            pts3d_2 = crop_and_resize_to_dust3r(pts3d_2, valid_coords[j], out_h, out_w)
            conf1 = crop_and_resize_to_dust3r(conf1, valid_coords[i], out_h, out_w)
            conf2 = crop_and_resize_to_dust3r(conf2, valid_coords[j], out_h, out_w)

            view1_img.append(imgs[i])
            view2_img.append(imgs[j])
            view1_idx.append(i)
            view2_idx.append(j)
            pts3d_1_list.append(pts3d_1)
            pts3d_2_list.append(pts3d_2)
            conf1_list.append(conf1)
            conf2_list.append(conf2)

    B = len(view1_idx)
    H, W = out_h, out_w
    true_shape = torch.tensor([[H, W]] * B, dtype=torch.int32)

    output = {
        "view1": {
            "img": torch.stack(view1_img).float(),
            "true_shape": true_shape.clone(),
            "idx": view1_idx,
            "instance": [str(k) for k in view1_idx],
        },
        "view2": {
            "img": torch.stack(view2_img).float(),
            "true_shape": true_shape.clone(),
            "idx": view2_idx,
            "instance": [str(k) for k in view2_idx],
        },
        "pred1": {
            "pts3d": torch.stack(pts3d_1_list),
            "conf": torch.stack(conf1_list),
        },
        "pred2": {
            "pts3d_in_other_view": torch.stack(pts3d_2_list),
            "conf": torch.stack(conf2_list),
        },
        "loss": None,
    }
    return output


def save_output(output, path):
    """torch.save the pairwise dict (CPU tensors) so it round-trips into demo_mm.py."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(output, path)
    print(f"Saved pairwise output ({len(output['view1']['idx'])} pairs) to {path}")


def main(args):
    print("Arguments:", vars(args))

    set_seed(args.seed)
    device, dtype = get_device_and_dtype()
    model = load_model(args.model, device, args.checkpoint)

    # Load images at high res; VGGT internally runs at 518.
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    images, original_coords, image_path_list = load_images(args.scene_dir, img_load_resolution, device)

    # Valid (non-padded) image box per image, scaled from the load grid to the 518 grid.
    valid_coords = original_coords[:, :4].cpu().numpy() * (vggt_fixed_resolution / img_load_resolution)

    output = build_pairwise_output(
        model, images, dtype, valid_coords, args.out_h, args.out_w, vggt_fixed_resolution,
    )
    set_trace()

    if args.output_path:
        output_path = args.output_path
    elif args.output:
        output_path = os.path.join(args.output, args.model, "pairwise_output.pth")
    else:
        output_path = os.path.join(args.scene_dir, "pairwise_output.pth")
    save_output(output, output_path)
    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)
