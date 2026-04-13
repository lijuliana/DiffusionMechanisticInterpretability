#!/usr/bin/env python3
"""
Local adaptation of generalization_profile_eval_cli.py for objrel_T5_DiT_mini_pilot.
Uses cached T5 embeddings, supports MPS (Apple Silicon), and workspace-relative paths.

Evaluates all checkpoints by default and generates training-curve plots
for exist_binding, unique_binding, spatial_relationship, overall, etc.

Usage:
    # Evaluate ALL checkpoints (default), 22 prompts, 5 images each
    python experimental_scripts/run_generalization_eval_local.py --single_prompt_mode

    # Evaluate specific checkpoints
    python experimental_scripts/run_generalization_eval_local.py \\
        --checkpoints epoch_1000_step_40000.pth epoch_4000_step_160000.pth \\
        --num_images 5 --single_prompt_mode

    # Full 264 prompts (slower)
    python experimental_scripts/run_generalization_eval_local.py --num_images 3
"""
import os
import sys
import ssl
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())

import argparse
import glob
import time
from os.path import join, dirname, basename
import torch
import pandas as pd
import numpy as np
import pickle as pkl

# Project paths
WORKSPACE = dirname(dirname(os.path.abspath(__file__)))
sys.path.insert(0, join(WORKSPACE, "PixArt-alpha"))
sys.path.insert(0, WORKSPACE)

from diffusion.utils.misc import read_config
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from transformers import T5Tokenizer

from utils.pixart_utils import construct_diffuser_transformer_from_config, load_pixart_ema_into_transformer
from utils.pixart_sampling_utils import PixArtAlphaPipeline_custom
from utils.relation_shape_dataset_lib import ShapesDataset
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate objrel_T5_DiT_mini_pilot locally")
    p.add_argument("--model_run_name", type=str, default="objrel_T5_DiT_mini_pilot")
    p.add_argument("--checkpoints", type=str, nargs="*", default=None,
                   help="Checkpoint filenames.  Default: 3 spread checkpoints (early/mid/late).")
    p.add_argument("--all_checkpoints", action="store_true",
                   help="Evaluate ALL checkpoints instead of just 3")
    p.add_argument("--num_images", type=int, default=2,
                   help="Images per prompt (default 2 for speed)")
    p.add_argument("--num_inference_steps", type=int, default=8,
                   help="Denoising steps (default 8; use 14 for higher quality)")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--generator_seed", type=int, default=42)
    p.add_argument("--single_prompt_mode", action="store_true",
                   help="Use 22 prompts (blue circle / red square) instead of full 264")
    p.add_argument("--output_dir", type=str, help="Override output dir")
    p.add_argument("--skip_plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Prompt generation (from generalization_profile_eval_cli.py)
# ---------------------------------------------------------------------------

def generate_prompt_collection(spatial_phrases,
                               template="{color1} {shape1} is {rel_text} {color2} {shape2}",
                               color1="blue", shape1="circle", color2="red", shape2="square"):
    prompts, scene_infos = [], []
    for rel, rel_texts in spatial_phrases.items():
        if rel in ["in_front", "behind"]:
            continue
        for rt in rel_texts:
            prompts.append(template.format(color1=color1, shape1=shape1,
                                           rel_text=rt, color2=color2, shape2=shape2))
            scene_infos.append({"color1": color1, "shape1": shape1,
                                "color2": color2, "shape2": shape2,
                                "spatial_relationship": rel})
    return prompts, scene_infos


def generate_all_prompt_collection(spatial_phrases,
                                   template="{color1} {shape1} is {rel_text} {color2} {shape2}"):
    from itertools import product
    prompts, scene_infos = [], []
    for c1, c2 in product(["red", "blue"], ["red", "blue"]):
        if c1 == c2:
            continue
        for s1, s2 in product(["square", "triangle", "circle"],
                              ["square", "triangle", "circle"]):
            if s1 == s2:
                continue
            for rel, rel_texts in spatial_phrases.items():
                if rel in ["in_front", "behind"]:
                    continue
                for rt in rel_texts:
                    prompts.append(template.format(color1=c1, shape1=s1,
                                                   rel_text=rt, color2=c2, shape2=s2))
                    scene_infos.append({"color1": c1, "shape1": s1,
                                        "color2": c2, "shape2": s2,
                                        "spatial_relationship": rel})
    return prompts, scene_infos


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def load_embedding_cache(cache_path):
    """Load T5 cache; expects '' key for unconditional and 'base::prompt' keys."""
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    emb = data.get("embedding_allrel_allobj", data)
    if "" not in emb:
        raise ValueError("Cache must have '' key.  Re-run scripts/precompute_t5_embeddings.py")
    return emb


# ---------------------------------------------------------------------------
# Core evaluation  (delegates to utils.eval_cached_embeddings)
# ---------------------------------------------------------------------------

def evaluate_pipeline_with_cache(pipeline, prompts, scene_infos, embedding_cache,
                                 device, args):
    """Generate images with cached embeddings + OpenCV eval; returns (eval_df, object_df)."""
    return evaluate_pipeline_on_prompts_with_cached_embeddings(
        pipeline,
        prompts,
        scene_infos,
        embedding_cache,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator_seed=args.generator_seed,
        device=device,
        weight_dtype=pipeline.transformer.dtype,
        show_prompt_progress=True,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "exist_binding", "unique_binding",
    "shape", "color",
    "spatial_relationship", "spatial_relationship_loose",
    "overall", "overall_loose",
]

METRIC_LABELS = {
    "exist_binding": "Exist Correct Binding",
    "unique_binding": "Unique Correct Binding",
    "shape": "Shape",
    "color": "Color",
    "spatial_relationship": "Spatial (strict)",
    "spatial_relationship_loose": "Spatial (loose)",
    "overall": "Overall (strict)",
    "overall_loose": "Overall (loose)",
}


def extract_step(ckpt_name: str) -> int:
    """Parse training step from checkpoint filename."""
    if "_step_" in ckpt_name:
        return int(ckpt_name.split("_step_")[-1].split(".pth")[0])
    return 0


def build_training_curve_df(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-checkpoint means for each metric."""
    rows = []
    for df in all_dfs:
        if df.empty:
            continue
        ckpt = df["checkpoint"].iloc[0]
        step = extract_step(ckpt)
        avail = [m for m in METRIC_COLS if m in df.columns]
        means = df[avail].mean()
        row = {"checkpoint": ckpt, "step": step, **means.to_dict()}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def plot_training_curves(curve_df: pd.DataFrame, eval_dir: str):
    """Generate and save training-curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avail = [m for m in METRIC_COLS if m in curve_df.columns]
    if not avail or curve_df.empty:
        print("  No metrics to plot.")
        return

    steps = curve_df["step"]

    # --- Graph 1: All metrics over training ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in avail:
        ax.plot(steps, curve_df[m], "o-", label=METRIC_LABELS.get(m, m))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("All Metrics over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(join(eval_dir, "plot_all_metrics_over_training.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved plot_all_metrics_over_training.png")

    # --- Graph 2: Binding metrics comparison ---
    binding_cols = [m for m in ["exist_binding", "unique_binding", "overall", "overall_loose"]
                    if m in curve_df.columns]
    if binding_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in binding_cols:
            ax.plot(steps, curve_df[m], "o-", label=METRIC_LABELS.get(m, m), linewidth=2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Binding & Overall Metrics over Training")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(eval_dir, "plot_binding_metrics_over_training.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved plot_binding_metrics_over_training.png")

    # --- Graph 3: Spatial strict vs loose ---
    spatial_cols = [m for m in ["spatial_relationship", "spatial_relationship_loose"]
                    if m in curve_df.columns]
    if spatial_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in spatial_cols:
            ax.plot(steps, curve_df[m], "o-", label=METRIC_LABELS.get(m, m), linewidth=2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Spatial Relationship Metrics over Training")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(eval_dir, "plot_spatial_metrics_over_training.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved plot_spatial_metrics_over_training.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()
    t0 = time.time()
    print(f"Device: {device}")

    savedir = join(WORKSPACE, "results", args.model_run_name)
    eval_dir = args.output_dir or join(savedir, "generalization_eval")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Output: {eval_dir}")

    # ---- Config & cache ----
    config = read_config(join(savedir, "config.py"))
    cache_path = join(savedir, "t5_embedding_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing: {cache_path}")
    embedding_cache = load_embedding_cache(cache_path)
    print(f"Loaded {len(embedding_cache)} cached embeddings")

    # ---- Prompts ----
    ds = ShapesDataset(num_images=100)
    template = "{color1} {shape1} is {rel_text} {color2} {shape2}"
    if args.single_prompt_mode:
        prompts, scene_infos = generate_prompt_collection(ds.spatial_phrases, template)
    else:
        prompts, scene_infos = generate_all_prompt_collection(ds.spatial_phrases, template)
    print(f"Prompts (before cache filter): {len(prompts)}")

    # Filter to prompts actually in the cache
    in_cache = [p for p in prompts if any(k.endswith(f"::{p}") for k in embedding_cache if k != "")]
    idx = [prompts.index(p) for p in in_cache]
    prompts = in_cache
    scene_infos = [scene_infos[i] for i in idx]
    print(f"Prompts in cache: {len(prompts)}")

    # ---- Pipeline ----
    # float16 is slower on CPU; only use it on GPU
    if device == "cpu":
        weight_dtype = torch.float32
    elif config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    transformer = construct_diffuser_transformer_from_config(config)
    ckptdir = join(savedir, "checkpoints")

    # Resolve checkpoints
    if args.checkpoints is not None:
        checkpoint_paths = [join(ckptdir, c) if not os.path.isabs(c) else c
                           for c in args.checkpoints]
        checkpoint_paths = [c for c in checkpoint_paths if os.path.exists(c)]
    else:
        all_ckpts = sorted(
            glob.glob(join(ckptdir, "*.pth")),
            key=lambda x: extract_step(basename(x)),
        )
        if args.all_checkpoints or len(all_ckpts) <= 3:
            checkpoint_paths = all_ckpts
        else:
            # Pick 3 spread checkpoints: early, mid, late
            checkpoint_paths = [all_ckpts[0],
                                all_ckpts[len(all_ckpts) // 2],
                                all_ckpts[-1]]
    print(f"Checkpoints to evaluate: {len(checkpoint_paths)}")
    for cp in checkpoint_paths:
        print(f"  {basename(cp)}")
    total_gens = len(checkpoint_paths) * len(prompts) * args.num_images
    est_sec_per_img = 6 if device != "cpu" else 15
    est_min = (total_gens * est_sec_per_img) / 60
    print(f"Total images to generate: {total_gens}  (est ~{est_min:.0f} min on {device})")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    pipeline = PixArtAlphaPipeline_custom(
        transformer=transformer,
        vae=vae,
        scheduler=DPMSolverMultistepScheduler(),
        tokenizer=tokenizer,
        text_encoder=None,
    )
    pipeline.transformer = pipeline.transformer.to(device=device, dtype=weight_dtype)
    pipeline.vae = pipeline.vae.to(device=device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # ---- Evaluate each checkpoint ----
    all_eval_dfs = []
    all_obj_dfs = []

    for ckpt_path in checkpoint_paths:
        ckpt_name = basename(ckpt_path)
        step = extract_step(ckpt_name)
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}  (step {step})")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_pixart_ema_into_transformer(pipeline.transformer, ckpt["state_dict_ema"])
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()

        eval_df, object_df = evaluate_pipeline_with_cache(
            pipeline, prompts, scene_infos, embedding_cache, device, args
        )
        eval_df["checkpoint"] = ckpt_name
        eval_df["step"] = step
        object_df["checkpoint"] = ckpt_name
        object_df["step"] = step

        all_eval_dfs.append(eval_df)
        all_obj_dfs.append(object_df)

        # Save per-checkpoint results
        eval_df.to_csv(join(eval_dir, f"eval_{ckpt_name}.csv"), index=False)
        object_df.to_pickle(join(eval_dir, f"objects_{ckpt_name}.pkl"))

        # Print summary
        if eval_df.empty:
            print("  No results (all prompts failed)")
        else:
            avail = [m for m in METRIC_COLS if m in eval_df.columns]
            means = eval_df[avail].mean()
            print("  Metric means:")
            for m in avail:
                print(f"    {METRIC_LABELS.get(m, m):30s}: {means[m]:.3f}")
            # Also show location stats
            for loc in ["Dx", "Dy", "x1", "y1", "x2", "y2"]:
                if loc in eval_df.columns:
                    valid = eval_df[loc].dropna()
                    if len(valid):
                        print(f"    {loc:30s}: mean={valid.mean():.1f}  std={valid.std():.1f}")

    # ---- Combine and save ----
    if not all_eval_dfs:
        print("\nNo successful evaluations.")
        return

    summary_eval = pd.concat(all_eval_dfs, ignore_index=True)
    summary_eval.to_csv(join(eval_dir, "eval_all_checkpoints.csv"), index=False)

    summary_obj = pd.concat(all_obj_dfs, ignore_index=True)
    summary_obj.to_pickle(join(eval_dir, "objects_all_checkpoints.pkl"))

    # Per-checkpoint aggregate
    curve_df = build_training_curve_df(all_eval_dfs)
    curve_df.to_csv(join(eval_dir, "training_curve_metrics.csv"), index=False)
    print(f"\n--- Training curve summary ---")
    print(curve_df.to_string(index=False))

    # ---- Plots ----
    if not args.skip_plots:
        print("\nGenerating plots...")
        plot_training_curves(curve_df, eval_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.  Results: {eval_dir}")


if __name__ == "__main__":
    main()
