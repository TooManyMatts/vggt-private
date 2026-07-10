# Output format for injection into `demo_mm.py`

`demo_mm.py` runs in two stages:

1. **Pairwise prediction** — `output = inference(pairs, model, ...)` produces a dict of
   per-pair pointmaps and confidences, saved with `save_output()` to
   `<output_dir>/pairwise_output.pth`.
2. **Global alignment (GA)** — `global_aligner(output, ...)` +
   `scene.compute_global_alignment(...)` stitches those pairwise pointmaps into one
   consistent scene.

To plug a **different model** into this pipeline, produce an `output` dict in the format
below, save it with `torch.save`, and load it in place of the DUSt3R prediction. This
file is the contract that dict must satisfy.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `N`    | number of input images |
| `B`    | number of pairs (batch dim). For a complete, symmetrized graph: `B = N*(N-1)` |
| `H,W`  | pointmap / image height and width (per pair) |

All tensors are plain `torch.Tensor` on CPU (move to CPU before saving so the file is
portable). `float32` unless noted.

---

## Full structure

```
output                                    dict (5 keys)
├─ 'view1'                                dict
│   ├─ 'img'          Tensor [B, 3, H, W] float32   normalized RGB, ~[-1, 1]
│   ├─ 'true_shape'   Tensor [B, 2]       int32     real (H, W) per image
│   ├─ 'idx'          list[int]  len B              source-image index
│   ├─ 'instance'     list[str]  len B              source-image id, e.g. '0'
│   └─ 'filename'     list[str]  len B              source-image filename, e.g. 'image_000001.jpg'
├─ 'view2'                                dict       same fields as view1
│   ├─ 'img'          Tensor [B, 3, H, W] float32
│   ├─ 'true_shape'   Tensor [B, 2]       int32
│   ├─ 'idx'          list[int]  len B
│   ├─ 'instance'     list[str]  len B
│   └─ 'filename'     list[str]  len B
├─ 'pred1'                                dict       prediction for view1, in view1's frame
│   ├─ 'pts3d'        Tensor [B, H, W, 3] float32   per-pixel 3D points
│   └─ 'conf'         Tensor [B, H, W]    float32   per-pixel confidence (>= 1)
├─ 'pred2'                                dict       prediction for view2, in view1's frame
│   ├─ 'pts3d_in_other_view'  Tensor [B, H, W, 3] float32
│   └─ 'conf'                 Tensor [B, H, W]    float32
└─ 'loss'             None                          not used downstream
```

---

## What the global aligner actually requires

Not every field is consumed. Tracing `global_aligner` →
`BasePCOptimizer._init_from_views`:

### Required (geometry — GA will not work without these)

| Field | Role |
|-------|------|
| `view1['idx']`, `view2['idx']` | define `edges` = which image pairs with which. **Must form a symmetric set**: for every `(i, j)` there must be a `(j, i)`. The `PointCloudOptimizer` assumes symmetrized pairs. |
| `pred1['pts3d']` | pointmap of view1, in view1's camera frame |
| `pred2['pts3d_in_other_view']` | pointmap of view2, **already transformed into view1's frame** (note the key name — it is *not* `pts3d`) |
| `pred1['conf']`, `pred2['conf']` | per-pixel weights; also used for MST-based init scoring and the final confidence masks. Values are expected `>= 1` (DUSt3R uses `1 + exp(...)`). |

Per-image resolution `(H, W)` is inferred from the **shape of `pred['pts3d']`**, not from
`true_shape`.

### Required for naming the outputs (not consumed by the GA itself)

| Field | Role |
|-------|------|
| `view1['filename']`, `view2['filename']` | actual source-image filename per row (parallel to `idx`: `filename[b]` names image `idx[b]`). The geometry optimization ignores it, but `run_global_alignment.py` uses it to key every per-image field of `results.pth` (`pts3d[name]`, `poses[name]`, `focals[name]`, `confidence_masks[name]`) by image filename. Must exactly match the `NAME` field in COLMAP's `images.{txt,bin}` so downstream code needs no ordering assumptions. |

### Optional (color only)

| Field | Role |
|-------|------|
| `view1['img']`, `view2['img']` | used **only** to color the exported pointcloud (`scene.imgs`). Geometry optimization ignores them. Omit them and GA still runs, but the `.glb`/`.ply` will have no colors and `get_3D_model_from_scene` expects them — keep `img` if you want a colored export. |

### Ignored entirely (safe to fill with placeholders)

- `view1['true_shape']`, `view2['true_shape']`
- `view1['instance']`, `view2['instance']`
- `output['loss']`  (`global_aligner` only reads `view1, view2, pred1, pred2`)

Provide them anyway to keep the format identical and tooling happy.

---

## Conventions you must respect

- **Pairing by row.** Row `b` of `view1`/`pred1` is paired with row `b` of
  `view2`/`pred2`. Index `b` of every tensor/list refers to the same pair.
- **`idx` indexes into `[0, N)`** and labels which original image each row's view is.
  `edges = [(view1['idx'][b], view2['idx'][b]) for b in range(B)]`.
- **`filename` is a function of `idx`.** Every row with the same `idx` must carry the
  same `filename`, and it must equal the image's `NAME` in COLMAP's `images.{txt,bin}`.
- **Symmetrization.** Mirror every pair. If you emit `(i, j)`, also emit `(j, i)` with
  the views swapped. This matches `make_pairs(..., symmetrize=True)`.
- **Frames.** `pred1['pts3d']` is in view1's frame; `pred2['pts3d_in_other_view']` is
  view2's points **expressed in view1's frame**. Both pointmaps of a pair therefore live
  in the same (view1) coordinate system.
- **Confidence scale.** `conf >= 1`. Larger = more confident. The GA applies
  `log(conf)` (default `conf='log'`) and thresholds with `min_conf_thr`.

---

## Minimal construction example

```python
import torch
from demo_mm import save_output, load_output

B, H, W = N * (N - 1), 368, 512   # symmetrized complete graph

output = {
    'view1': {
        'img':         imgs1,                 # [B, 3, H, W] float32  (optional, for color)
        'true_shape':  torch.tensor([[H, W]] * B, dtype=torch.int32),  # [B, 2] (ignored)
        'idx':         idx1,                  # list[int] len B
        'instance':    [str(i) for i in idx1],# list[str] len B (ignored)
        'filename':    [names[i] for i in idx1],  # list[str] len B, COLMAP NAME per image
    },
    'view2': {
        'img':         imgs2,
        'true_shape':  torch.tensor([[H, W]] * B, dtype=torch.int32),
        'idx':         idx2,
        'instance':    [str(i) for i in idx2],
        'filename':    [names[i] for i in idx2],
    },
    'pred1': {'pts3d':                pts3d_1,  # [B, H, W, 3] float32, view1 frame
              'conf':                 conf_1},  # [B, H, W]    float32, >= 1
    'pred2': {'pts3d_in_other_view':  pts3d_2,  # [B, H, W, 3] float32, view1 frame
              'conf':                 conf_2},
    'loss': None,
}

save_output(output, 'results/my_model/pairwise_output.pth')
# ... then in demo_mm.py: output = load_output('results/my_model/pairwise_output.pth')
```

`save_output` / `load_output` (in `demo_mm.py`) use `torch.save`/`torch.load`, so the
nested dicts, tensors, Python lists and `None` round-trip with exact shape and dtype.
