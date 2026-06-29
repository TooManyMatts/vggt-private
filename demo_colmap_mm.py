# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT COLMAP Runner (mm)
=======================

Runs VGGT on a folder of images and exports a COLMAP-format sparse
reconstruction. Images are read directly from --scene_dir (no images/
subfolder); results are written to <scene_dir>/sparse/.

Input:
    examples_mm/a01_ipad_lts_s06/
    ├── image_000001.jpg
    ├── image_000002.jpg
    └── ...

Output (folder name encodes BA / point cap / model, e.g. sparse_noba_10M_1b):
    examples_mm/a01_ipad_lts_s06/
    └── sparse_noba_10M_1b/
        ├── cameras.bin    # Camera parameters (COLMAP format)
        ├── images.bin     # Pose for each image (COLMAP format)
        ├── points3D.bin   # 3D points (COLMAP format)
        ├── cameras.txt    # Same as above, COLMAP text format
        ├── images.txt
        ├── points3D.txt
        └── points.ply     # Point cloud visualization file

Example
-------
# Feed-forward reconstruction (default scene_dir, VGGT-1B)
python demo_colmap_mm.py

# Point at a different folder of images
python demo_colmap_mm.py --scene_dir examples_mm/a01_ipad_lts_s06

# Use the commercial-licensed checkpoint (gated on HF, requires login)
python demo_colmap_mm.py --scene_dir examples_mm/a01_ipad_lts_s06 --model 1b-commercial

# Higher quality with bundle adjustment
python demo_colmap_mm.py --scene_dir examples_mm/a01_ipad_lts_s06 --use_ba

# Bundle adjustment with the commercial checkpoint
python demo_colmap_mm.py --scene_dir examples_mm/a01_ipad_lts_s06 --use_ba --model 1b-commercial
"""

import numpy as np
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import pycolmap


from mm_utils import set_seed, get_device_and_dtype, load_model, load_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, default="examples_mm/a01_ipad_lts_s06", help="Directory containing the scene images")
    parser.add_argument(
        "--model", type=str, default="1b", choices=["1b", "1b-commercial"],
        help="Which VGGT checkpoint to use: '1b' (research-only) or '1b-commercial' (commercial license, gated)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument(
        "--max_points_for_colmap", type=int, default=10000000,
        help="Max number of 3D points to keep in the feed-forward (wo BA) reconstruction",
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def reconstruct_with_ba(
    args, images, extrinsic, intrinsic, depth_conf, points_3d, dtype,
    img_load_resolution, vggt_fixed_resolution,
):
    """Build a COLMAP reconstruction refined with bundle adjustment.

    Predicts 2D point tracks across all frames (VGGSfM tracker), keeps tracks
    above the visibility threshold, builds a pycolmap reconstruction from the
    tracked points + VGGT cameras, and refines it with pycolmap bundle
    adjustment. Works at img_load_resolution (1024); the intrinsics from VGGT
    (estimated at 518) are rescaled accordingly.

    Returns (reconstruction, points_3d, points_rgb, shared_camera,
    reconstruction_resolution), where points_3d/points_rgb are the triangulated
    track points used for the .ply export and reconstruction_resolution is the
    image size the reconstruction's 2D coordinates live in.
    """
    image_size = np.array(images.shape[-2:])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = args.shared_camera

    with torch.cuda.amp.autocast(dtype=dtype):
        # Predicting Tracks
        # Using VGGSfM tracker instead of VGGT tracker for efficiency
        # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
        # Will be fixed in VGGT v2

        # You can also change the pred_tracks to tracks from any other methods
        # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
        )

        torch.cuda.empty_cache()

    # rescale the intrinsic matrix from 518 to 1024
    intrinsic[:, :2, :] *= scale
    track_mask = pred_vis_scores > args.vis_thresh

    # TODO: radial distortion, iterative BA, masks
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    return reconstruction, points_3d, points_rgb, shared_camera, img_load_resolution


def reconstruct_feedforward(
    args, images, extrinsic, intrinsic, depth_conf, points_3d, vggt_fixed_resolution,
):
    """Build a COLMAP reconstruction directly from feed-forward VGGT outputs.

    No tracks and no bundle adjustment: the per-pixel 3D points (depth maps
    unprojected with the predicted cameras) are filtered by depth confidence
    (>= args.conf_thres_value), randomly capped at args.max_points_for_colmap,
    colored by sampling the input images, and written into a pycolmap
    reconstruction as-is. Works at vggt_fixed_resolution (518) with PINHOLE
    cameras (shared cameras unsupported in this mode).

    Returns (reconstruction, points_3d, points_rgb, shared_camera,
    reconstruction_resolution), where points_3d/points_rgb are the filtered
    point cloud used for the .ply export and reconstruction_resolution is the
    image size the reconstruction's 2D coordinates live in.
    """
    shared_camera = False  # in the feedforward manner, we do not support shared camera
    camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= args.conf_thres_value
    # randomly sample 3D points: at most args.max_points_for_colmap points are written to the reconstruction
    conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    return reconstruction, points_3d, points_rgb, shared_camera, vggt_fixed_resolution


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    set_seed(args.seed)
    device, dtype = get_device_and_dtype()
    model = load_model(args.model, device)

    # Load images in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords, image_path_list = load_images(args.scene_dir, img_load_resolution, device)
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    result = run_reconstruction(
        args, model, images, dtype, base_image_path_list, original_coords,
        img_load_resolution, vggt_fixed_resolution,
    )

    save_reconstruction(args, result["reconstruction"], result["points_3d"], result["points_rgb"])

    return True


def run_reconstruction(args, model, images, dtype, base_image_path_list, original_coords,
                       img_load_resolution, vggt_fixed_resolution):
    """Run VGGT and build a finalized COLMAP reconstruction.

    Returns a dict with the reconstruction plus the intermediate estimates a
    caller might need (points, cameras, depth, extrinsics/intrinsics).
    """
    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        reconstruction, points_3d, points_rgb, shared_camera, reconstruction_resolution = reconstruct_with_ba(
            args, images, extrinsic, intrinsic, depth_conf, points_3d, dtype,
            img_load_resolution, vggt_fixed_resolution,
        )
    else:
        reconstruction, points_3d, points_rgb, shared_camera, reconstruction_resolution = reconstruct_feedforward(
            args, images, extrinsic, intrinsic, depth_conf, points_3d, vggt_fixed_resolution,
        )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    return {
        "reconstruction": reconstruction,
        "points_3d": points_3d,
        "points_rgb": points_rgb,
        "shared_camera": shared_camera,
        "reconstruction_resolution": reconstruction_resolution,
        "depth_map": depth_map,
        "depth_conf": depth_conf,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
    }


def save_reconstruction(args, reconstruction, points_3d, points_rgb):
    """Write the COLMAP reconstruction (binary + text) and point cloud to disk."""
    if args.use_ba:
        output_subdir = f"sparse_ba_{args.model}"
    else:
        cap = args.max_points_for_colmap
        cap_label = f"{cap // 1_000_000}M" if cap % 1_000_000 == 0 else f"{cap // 1000}k" if cap % 1000 == 0 else str(cap)
        output_subdir = f"sparse_noba_{cap_label}_{args.model}"
    print(f"Saving reconstruction to {args.scene_dir}/{output_subdir}")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, output_subdir)
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Also export the text format (cameras.txt, images.txt, points3D.txt)
    reconstruction.write_text(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""

