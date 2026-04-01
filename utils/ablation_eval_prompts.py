"""
Prompt lists for head ablation experiments, aligned with training / generalization eval.

Use ``training_template`` to match ``run_generalization_eval_local.py --single_prompt_mode``
(blue circle / red square, all spatial phrases). Use ``diverse`` for a harder combinatorial subset.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import pandas as pd


def build_training_template_prompts(
    embedding_cache: dict,
    spatial_phrases: dict,
    template: str = "{color1} {shape1} is {rel_text} {color2} {shape2}",
    color1: str = "blue",
    shape1: str = "circle",
    color2: str = "red",
    shape2: str = "square",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Same prompt distribution as generalization eval single-pair mode.
    Drops prompts missing from ``embedding_cache`` (keys like ``base::<prompt>``).
    """
    prompts: List[str] = []
    scene_infos: List[Dict[str, Any]] = []
    for rel, rel_texts in spatial_phrases.items():
        if rel in ("in_front", "behind"):
            continue
        for rt in rel_texts:
            p = template.format(
                color1=color1, shape1=shape1, rel_text=rt, color2=color2, shape2=shape2
            )
            prompts.append(p)
            scene_infos.append(
                {
                    "color1": color1,
                    "shape1": shape1,
                    "color2": color2,
                    "shape2": shape2,
                    "spatial_relationship": rel,
                }
            )

    kept_p: List[str] = []
    kept_s: List[Dict[str, Any]] = []
    for p, s in zip(prompts, scene_infos):
        if any(k != "" and k.endswith(f"::{p}") for k in embedding_cache):
            kept_p.append(p)
            kept_s.append(s)
    return kept_p, kept_s


def build_diverse_subset_prompts(
    prompt_scene_info_all_df: pd.DataFrame,
    embedding_cache: dict,
    n_prompts: int = 12,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Legacy behavior: sample up to ``n_prompts`` from combinatorial dataframe (harder).
    """
    ablation_subset = (
        prompt_scene_info_all_df.groupby("spatial_relationship")
        .head(2)
        .reset_index(drop=True)
        .head(n_prompts)
    )
    prompts = ablation_subset["prompt"].tolist()
    scene_infos = [
        {
            "color1": r.color1,
            "shape1": r.shape1,
            "color2": r.color2,
            "shape2": r.shape2,
            "spatial_relationship": r.spatial_relationship,
        }
        for r in ablation_subset.itertuples()
    ]
    return prompts, scene_infos


def eval_means_line(eval_df: pd.DataFrame, prefix: str = "") -> str:
    """One-line summary of key metrics (for notebooks)."""
    if eval_df is None or eval_df.empty:
        return f"{prefix}empty eval_df"
    cols = [
        "overall",
        "unique_binding",
        "exist_binding",
        "spatial_relationship",
        "spatial_relationship_loose",
    ]
    parts = [f"{c}={eval_df[c].mean():.3f}" for c in cols if c in eval_df.columns]
    return prefix + (" | ".join(parts) if parts else "no metric cols")
