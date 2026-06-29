# VGGT → DUSt3R Pairwise Export

This repository is a **fork of [VGGT (Visual Geometry Grounded Transformer)](https://github.com/facebookresearch/vggt)**.
It adds tooling to run VGGT and export its predictions as a DUSt3R-style *pairwise*
prediction dict that can be fed into a DUSt3R global-alignment pipeline. The upstream
VGGT code is otherwise unchanged.

## Installation

Clone the repo and install dependencies (torch, torchvision, numpy, Pillow,
huggingface_hub, tqdm):

```bash
git clone <this-repo-url>
cd vggt-private
pip install -r requirements.txt
```

### Checkpoints

The export script loads weights in this order:

1. `--checkpoint <folder>` if passed (must contain `model.safetensors`),
2. otherwise `checkpoints/<model>/` if present locally,
3. otherwise the model is downloaded from Hugging Face
   (`facebook/VGGT-1B` or `facebook/VGGT-1B-Commercial`).

To keep a local copy under `checkpoints/`, download the two files
(`config.json` and `model.safetensors`) for the model you want into
`checkpoints/1b/` and/or `checkpoints/1b-commercial/`. Note that
`VGGT-1B-Commercial` is a gated repo and requires Hugging Face access.

## Running `export_pairwise_mm.py`

```bash
python export_pairwise_mm.py \
    --scene_dir examples_mm/a01_ipad_lts_s06 \
    --model 1b \
    --checkpoint checkpoints/1b \
    --output outputs/a01

# commercial checkpoint
python export_pairwise_mm.py \
    --scene_dir examples_mm/a01_ipad_lts_s06 \
    --model 1b-commercial \
    --checkpoint checkpoints/1b-commercial \
    --output outputs/a01
```

The result is written to `<output>/<model>/pairwise_output.pth` — e.g.
`outputs/a01/1b/pairwise_output.pth`.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene_dir` | `examples_mm/a01_ipad_lts_s06_2img` | Folder of input images (images only). |
| `--model` | `1b` | Which checkpoint: `1b` (research-only) or `1b-commercial` (gated). Also names the output subfolder. |
| `--checkpoint` | `None` | Explicit checkpoint folder containing `model.safetensors`. Overrides the `--model` lookup. |
| `--output` | `None` | Output base folder; result goes to `<output>/<model>/pairwise_output.pth`. |
| `--output_path` | `None` | Explicit full path for the `.pth`; overrides `--output`. |
| `--out_h` / `--out_w` | `368` / `512` | Output pointmap/image resolution. |
| `--seed` | `42` | Random seed. |

If neither `--output` nor `--output_path` is given, the result defaults to
`<scene_dir>/pairwise_output.pth`.

## What `export_pairwise_mm.py` does

VGGT predicts every image in a single global frame, but a DUSt3R global aligner
expects **per-pair pointmaps expressed in view1's camera frame**. This script bridges
that gap.

For a scene of `N` images it builds a symmetrized complete graph: for every ordered
pair `(i, j)`, `i != j` (so `N*(N-1)` pairs), it runs VGGT independently on just that
2-image pair and re-expresses both pointmaps in view `i`'s camera frame:

- `pred1['pts3d']` — image `i`'s points in camera `i`'s frame,
- `pred2['pts3d_in_other_view']` — image `j`'s points in camera `i`'s frame.

VGGT's square padding is stripped and the maps are resized to the DUSt3R resolution
(`--out_h` × `--out_w`). Depth confidences are clamped to `>= 1` (the aligner applies
`log(conf)` as a per-pixel weight).

The output is a single `torch.save`'d dict (`pairwise_output.pth`) with `view1`,
`view2`, `pred1`, `pred2`, and `loss` keys, ready to drop into a DUSt3R
global-alignment stage in place of `output = inference(pairs, model, ...)`. The exact
tensor shapes and contract are documented in [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).
