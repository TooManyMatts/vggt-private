# BA_SPECS — `reconstruct_with_ba()` interface

Contract for the bundle-adjustment stage in `demo_colmap_mm.py`
(`reconstruct_with_ba`, demo_colmap_mm.py:141). Any foundation model that
produces the inputs below can be plugged into this BA stage — the model is
only needed *before* this call; BA itself uses no VGGT internals (it runs the
VGGSfM tracker + pycolmap).

## Call site

```python
reconstruction, points_3d, points_rgb, shared_camera, reconstruction_resolution = reconstruct_with_ba(
    args, images, extrinsic, intrinsic, depth_conf, points_3d, dtype,
    img_load_resolution, vggt_fixed_resolution,
)
```

## Inputs

`S` = number of frames (example run: 15).

| Name | Type / shape | Notes |
|---|---|---|
| `images` | `torch.Tensor [S, 3, H, W]` on GPU, float, values in [0, 1] | Square, `H = W = img_load_resolution` (1024). Used for track prediction, so this is the resolution the tracks (and the reconstruction's 2D points) live in. |
| `extrinsic` | `np.ndarray (S, 3, 4)` | Camera-from-world (world→cam), OpenCV convention `[R | t]`. |
| `intrinsic` | `np.ndarray (S, 3, 3)` | Pinhole K, **expressed at `vggt_fixed_resolution` (518)** — the function rescales it internally by `img_load_resolution / vggt_fixed_resolution`. If your model estimates K at a different resolution, adjust accordingly. |
| `depth_conf` | `np.ndarray (S, 518, 518)` | Per-pixel confidence for the depth/points. Used only to score/filter candidate query points for tracking (higher = better); any per-pixel quality map works. |
| `points_3d` | `np.ndarray (S, 518, 518, 3)` | Per-pixel 3D points in **world** coordinates (depth maps unprojected with `extrinsic`/`intrinsic`). Provides the initial 3D position for each track before BA refines it. Same grid as `depth_conf`. |
| `dtype` | `torch.dtype` | Autocast dtype for the tracker forward pass (`bfloat16` here). |
| `img_load_resolution` | `int` (1024) | Resolution of `images`; also returned as `reconstruction_resolution`. |
| `vggt_fixed_resolution` | `int` (518) | Resolution the intrinsics / depth grid were estimated at. |

### `args` fields actually consumed

| Arg | Default | Role |
|---|---|---|
| `max_query_pts` | 4096 | Max keypoints extracted per query frame. |
| `query_frame_num` | 8 | Number of query frames for the tracker. |
| `fine_tracking` | True | Fine (slower, more accurate) tracking stage. |
| `vis_thresh` | 0.2 | Keep track observations with visibility score > this. |
| `max_reproj_error` | 8.0 | Inlier threshold (px) when building the pycolmap reconstruction. |
| `shared_camera` | False | Single shared camera for all frames. |
| `camera_type` | `SIMPLE_PINHOLE` | pycolmap camera model. |

(`scene_dir`, `model`, `seed`, `use_ba`, `conf_thres_value`,
`max_points_for_colmap` are **not** used by this function.)

## Outputs

`P` = number of surviving 3D track points (example run: 37 432).

| Name | Type / shape | Notes |
|---|---|---|
| `reconstruction` | `pycolmap.Reconstruction` | BA-refined cameras, images and points. 1-indexed image/camera ids; image *names* are still placeholders — the caller renames them and rescales cameras to original image sizes afterwards (`rename_colmap_recons_and_rescale_camera`). 2D point coords are at `reconstruction_resolution`. |
| `points_3d` | `np.ndarray (P, 3)` | Triangulated track points, world coords. **Shadows the dense input of the same name.** Used for `points.ply` export. |
| `points_rgb` | `np.ndarray (P, 3)` uint8 | Point colors, 0–255. |
| `shared_camera` | `bool` | Echo of `args.shared_camera`. |
| `reconstruction_resolution` | `int` | = `img_load_resolution` (1024); the pixel space of the reconstruction's 2D points. |

Raises `ValueError("No reconstruction can be built with BA")` if too few
inliers per frame (< 64) survive filtering.
