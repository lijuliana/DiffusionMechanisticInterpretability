# Generalization Eval: Methods Used

## Overview

The `run_generalization_eval_local.py` script evaluates your trained PixArt-mini model on spatial relationship prompts. It generates images, runs object detection, and measures how often the model produces the correct spatial layout.

## Evaluation Pipeline

1. **Load model** — Load checkpoint (e.g. epoch 4000) into the diffusion transformer.
2. **Load cached embeddings** — Use pre-computed T5 embeddings from `t5_embedding_cache.pt` (no T5 in memory).
3. **Generate images** — For each prompt, run the pipeline with cached `prompt_embeds` and `negative_prompt_embeds` (implementation: `utils/eval_cached_embeddings.py`; must pass `prompt=None` and `negative_prompt=None` with embeds, and use a CFG similar to training, e.g. **4.5** — not **1.0**).
4. **Object detection** — OpenCV contour detection + shape classification (triangle/square/circle) + color (red/blue).
5. **Spatial evaluation** — Compare detected object positions to the expected spatial relationship.

## Metrics (from `utils/cv2_eval_utils.py`)

| Metric | Description |
|--------|-------------|
| **overall** | All of: correct shape, color, unique binding, and strict spatial relationship |
| **shape** | Both objects have the expected shape |
| **color** | Both objects have the expected color (red/blue) |
| **exist_binding** | At least one detected object per role (shape+color) |
| **unique_binding** | Exactly one object matches each role (no duplicates) |
| **spatial_relationship** | Strict: observed relation exactly equals expected (above/below/left/right/diagonals) |
| **spatial_relationship_loose** | Loose: direction correct (e.g. Dy < -5 for "above") |

## Spatial Relation Logic

- **Strict** (`identity_spatial_relation`): Uses 5px threshold. If \|Dx\| ≤ 5 → vertical (above/below). If \|Dy\| ≤ 5 → horizontal (left/right). Else → diagonal (upper_left, etc.).
- **Loose** (`evaluate_spatial_relation_loose_row`): Checks only direction, e.g. `above` → Dy < -threshold.

## Usage

```bash
# Quick run: 1 checkpoint, 22 prompts, 5 images each
python experimental_scripts/run_generalization_eval_local.py --single_prompt_mode

# Full run: all checkpoints, 264 prompts
python experimental_scripts/run_generalization_eval_local.py --checkpoints

# Custom
python experimental_scripts/run_generalization_eval_local.py \
  --checkpoints epoch_1000_step_40000.pth epoch_4000_step_160000.pth \
  --num_images 3 --guidance_scale 4.5
```

Output: `results/objrel_T5_DiT_mini_pilot/generalization_eval/eval_*.csv`
