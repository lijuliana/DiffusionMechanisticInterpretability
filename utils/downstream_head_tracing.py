"""
Utilities for tracing downstream reader heads of a source relation head.

This module is notebook-oriented: it packages dataframe joins, candidate ranking,
pair-ablation sweeps, and functional summaries so the notebook can stay concise.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from os.path import basename, join
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings
from utils.zero_head_ablation_utils import apply_zero_head_ablation_multi, restore_processors


DEFAULT_BEHAVIOR_COLS = [
    "spatial_relationship",
    "spatial_relationship_loose",
    "overall",
    "overall_loose",
    "unique_binding",
    "exist_binding",
]


@dataclass(frozen=True)
class HeadCondition:
    condition_key: str
    condition_label: str
    layer_head_pairs: tuple[tuple[int, int], ...]


def _as_pair_tuple_list(layer_head_pairs: Sequence[Sequence[int]] | Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    return [tuple(map(int, pair)) for pair in layer_head_pairs]


def head_pairs_to_str(layer_head_pairs: Sequence[Sequence[int]] | Sequence[tuple[int, int]]) -> str:
    return "+".join([f"L{int(layer)}H{int(head)}" for layer, head in layer_head_pairs])


def _resolve_cross_attn_transformer(transformer):
    if hasattr(transformer, "transformer_blocks"):
        return transformer
    if hasattr(transformer, "module"):
        return _resolve_cross_attn_transformer(transformer.module)
    if hasattr(transformer, "transformer"):
        return _resolve_cross_attn_transformer(transformer.transformer)
    raise AttributeError("Expected transformer with `.transformer_blocks` for cross-attention analysis.")


def _iter_candidate_heads(
    *,
    n_layers: int,
    n_heads: int,
    source_head: tuple[int, int],
    candidate_layers: str = "later_only",
    candidate_head_pairs: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    source_layer, source_head_idx = map(int, source_head)
    if candidate_head_pairs is not None:
        return _as_pair_tuple_list(candidate_head_pairs)

    pairs: list[tuple[int, int]] = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if candidate_layers == "later_only" and layer_idx <= source_layer:
                continue
            if candidate_layers == "same_or_later":
                if layer_idx < source_layer:
                    continue
                if layer_idx == source_layer and head_idx == source_head_idx:
                    continue
            elif candidate_layers == "all":
                if layer_idx == source_layer and head_idx == source_head_idx:
                    continue
            elif candidate_layers != "later_only":
                raise ValueError(f"Unknown candidate_layers mode: {candidate_layers}")
            pairs.append((layer_idx, head_idx))
    return pairs


def _head_weight_views(cross_attn_module, head_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = int(cross_attn_module.to_q.weight.shape[0] // cross_attn_module.heads)
    inner_dim = int(cross_attn_module.to_q.weight.shape[0])
    start = int(head_idx) * head_dim
    stop = start + head_dim

    q_w = cross_attn_module.to_q.weight[start:stop, :].detach()
    k_w = cross_attn_module.to_k.weight[start:stop, :].detach()
    v_w = cross_attn_module.to_v.weight[start:stop, :].detach()
    o_w = cross_attn_module.to_out[0].weight[:, start:stop].detach()
    return q_w, k_w, v_w, o_w


def _top_singular_triplet(matrix: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return float(s[0].item()), u[:, 0], vh[0, :]


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    ab = pd.concat([a, b], axis=1).dropna()
    if len(ab) < 2:
        return np.nan
    if ab.iloc[:, 0].nunique() <= 1 or ab.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(ab.iloc[:, 0].corr(ab.iloc[:, 1]))


def _rank_turn_on(series: pd.Series) -> float:
    if series.empty or series.isna().all():
        return np.nan
    s = series.astype(float)
    lo = float(s.min())
    hi = float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    threshold = lo + 0.7 * (hi - lo)
    idx = s.index[s >= threshold]
    if len(idx) == 0:
        return np.nan
    return float(idx[0])


def build_head_metric_trajectory_df(
    head_align_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None,
    source_head: tuple[int, int],
    *,
    candidate_layers: str = "later_only",
    candidate_head_pairs: Iterable[tuple[int, int]] | None = None,
    alignment_col: str = "cosine",
    behavior_cols: Sequence[str] | None = None,
    source_prefix: str = "source",
) -> pd.DataFrame:
    """
    Build one row per (epoch, layer_idx, head_idx) with source-head alignment and
    checkpoint-level behavior metrics attached.
    """
    if behavior_cols is None:
        behavior_cols = DEFAULT_BEHAVIOR_COLS
    required = {"epoch", "layer_idx", "head_idx", alignment_col}
    missing = required - set(head_align_df.columns)
    if missing:
        raise KeyError(f"head_align_df missing required columns: {sorted(missing)}")

    source_layer, source_head_idx = map(int, source_head)
    work_df = head_align_df.copy()
    work_df["layer_idx"] = work_df["layer_idx"].astype(int)
    work_df["head_idx"] = work_df["head_idx"].astype(int)
    work_df["epoch"] = work_df["epoch"].astype(int)
    work_df["alignment_value"] = work_df[alignment_col].astype(float)
    work_df["alignment_abs"] = work_df["alignment_value"].abs()

    source_df = (
        work_df[(work_df["layer_idx"] == source_layer) & (work_df["head_idx"] == source_head_idx)]
        [["epoch", "alignment_value", "alignment_abs"]]
        .rename(
            columns={
                "alignment_value": f"{source_prefix}_{alignment_col}",
                "alignment_abs": f"{source_prefix}_abs_{alignment_col}",
            }
        )
    )

    if candidate_head_pairs is not None:
        keep = set(_as_pair_tuple_list(candidate_head_pairs))
        work_df = work_df[work_df.apply(lambda row: (int(row["layer_idx"]), int(row["head_idx"])) in keep, axis=1)]
    elif candidate_layers == "later_only":
        work_df = work_df[work_df["layer_idx"] > source_layer]
    elif candidate_layers == "same_or_later":
        work_df = work_df[work_df["layer_idx"] >= source_layer]
        work_df = work_df[~((work_df["layer_idx"] == source_layer) & (work_df["head_idx"] == source_head_idx))]
    elif candidate_layers != "all":
        raise ValueError(f"Unknown candidate_layers mode: {candidate_layers}")

    traj_df = work_df.merge(source_df, on="epoch", how="left")

    if behavior_df is not None and not behavior_df.empty:
        available_cols = [c for c in behavior_cols if c in behavior_df.columns]
        behavior_keep = ["epoch", *available_cols]
        if "step" in behavior_df.columns:
            behavior_keep.append("step")
        if "checkpoint" in behavior_df.columns:
            behavior_keep.append("checkpoint")
        behavior_small = behavior_df[behavior_keep].copy()
        behavior_small = behavior_small.drop_duplicates(subset=["epoch"]).sort_values("epoch")
        traj_df = traj_df.merge(behavior_small, on="epoch", how="left")

    return traj_df.sort_values(["layer_idx", "head_idx", "epoch"]).reset_index(drop=True)


def rank_downstream_candidates_by_correlation(
    trajectory_df: pd.DataFrame,
    *,
    source_alignment_col: str = "source_abs_cosine",
    candidate_alignment_col: str = "alignment_abs",
    behavior_cols: Sequence[str] | None = None,
    show_progress: bool = False,
    progress_desc: str = "Rank downstream heads",
) -> pd.DataFrame:
    """
    Rank downstream candidates by how their trajectory covaries with the source head
    and with behavioral metrics.
    """
    if behavior_cols is None:
        behavior_cols = ["spatial_relationship", "unique_binding", "exist_binding", "overall"]

    rows = []
    group_cols = ["layer_idx", "head_idx"]
    grouped_items = list(trajectory_df.groupby(group_cols))
    iterator = tqdm(grouped_items, desc=progress_desc, unit="head", mininterval=1) if show_progress else grouped_items
    for (layer_idx, head_idx), dfh in iterator:
        dfh = dfh.sort_values("epoch")
        cand_series = dfh[candidate_alignment_col]
        source_series = dfh[source_alignment_col]

        row = {
            "layer_idx": int(layer_idx),
            "head_idx": int(head_idx),
            "head_label": f"L{int(layer_idx)}H{int(head_idx)}",
            "n_epochs": int(dfh["epoch"].nunique()),
            "candidate_mean_alignment": float(cand_series.mean(skipna=True)),
            "candidate_max_alignment": float(cand_series.max(skipna=True)),
            "corr_source_alignment": _safe_corr(cand_series, source_series),
            "source_turn_on_epoch": _rank_turn_on(pd.Series(source_series.values, index=dfh["epoch"].values)),
            "candidate_turn_on_epoch": _rank_turn_on(pd.Series(cand_series.values, index=dfh["epoch"].values)),
        }
        if np.isfinite(row["source_turn_on_epoch"]) and np.isfinite(row["candidate_turn_on_epoch"]):
            row["turn_on_delay"] = row["candidate_turn_on_epoch"] - row["source_turn_on_epoch"]
        else:
            row["turn_on_delay"] = np.nan

        corr_terms: list[float] = []
        for col in behavior_cols:
            if col in dfh.columns:
                corr_val = _safe_corr(cand_series, dfh[col])
                row[f"corr_{col}"] = corr_val
                if np.isfinite(corr_val):
                    corr_terms.append(abs(corr_val))
        row["behavior_corr_score"] = float(np.mean(corr_terms)) if corr_terms else np.nan

        score_terms = []
        if np.isfinite(row["corr_source_alignment"]):
            score_terms.append(abs(row["corr_source_alignment"]))
        if np.isfinite(row["behavior_corr_score"]):
            score_terms.append(row["behavior_corr_score"])
        if np.isfinite(row["turn_on_delay"]):
            # Prefer modest positive delays; penalize negative delays.
            score_terms.append(max(0.0, 1.0 - abs(row["turn_on_delay"]) / max(1.0, row["n_epochs"] - 1)))
        row["composite_score"] = float(np.mean(score_terms)) if score_terms else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["composite_score", "corr_source_alignment", "candidate_max_alignment"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def screen_downstream_candidates_by_ov_qk(
    transformer,
    source_head: tuple[int, int],
    *,
    candidate_layers: str = "later_only",
    candidate_head_pairs: Iterable[tuple[int, int]] | None = None,
    show_progress: bool = False,
    progress_desc: str = "OV-QK screening",
) -> pd.DataFrame:
    """
    Weight-based candidate screen using the source head's OV map and downstream heads'
    Q/K projections. This is a cheap structural proxy for "can this head read what the
    source head writes?" and is intended as a screening heuristic, not a causal test.
    """
    transformer = _resolve_cross_attn_transformer(transformer)
    if len(transformer.transformer_blocks) == 0:
        raise AttributeError("Expected transformer.transformer_blocks for OV-QK screening.")

    source_layer, source_head_idx = map(int, source_head)
    if source_layer >= len(transformer.transformer_blocks):
        raise IndexError(f"Source layer {source_layer} is out of range for transformer.transformer_blocks.")

    source_attn = transformer.transformer_blocks[source_layer].attn2
    if source_head_idx >= int(source_attn.heads):
        raise IndexError(f"Source head {source_head_idx} is out of range for layer {source_layer}.")

    _, _, source_v_w, source_o_w = _head_weight_views(source_attn, source_head_idx)
    source_ov = source_o_w.float() @ source_v_w.float()
    source_ov_fro = float(torch.linalg.matrix_norm(source_ov, ord="fro").item())
    source_ov_spectral, source_write_vec, source_cond_vec = _top_singular_triplet(source_ov)

    candidate_pairs = _iter_candidate_heads(
        n_layers=len(transformer.transformer_blocks),
        n_heads=int(source_attn.heads),
        source_head=source_head,
        candidate_layers=candidate_layers,
        candidate_head_pairs=candidate_head_pairs,
    )
    iterator = tqdm(candidate_pairs, desc=progress_desc, unit="head", mininterval=1) if show_progress else candidate_pairs

    rows: list[dict] = []
    for layer_idx, head_idx in iterator:
        cross_attn = transformer.transformer_blocks[int(layer_idx)].attn2
        q_w, k_w, _, _ = _head_weight_views(cross_attn, int(head_idx))
        q_w = q_w.float()
        k_w = k_w.float()

        q_read_vec = q_w @ source_write_vec
        k_read_vec = k_w @ source_cond_vec
        ov_qk = q_w @ source_ov @ k_w.T

        q_norm = float(torch.linalg.vector_norm(q_w).item())
        k_norm = float(torch.linalg.vector_norm(k_w).item())
        row = {
            "layer_idx": int(layer_idx),
            "head_idx": int(head_idx),
            "head_label": f"L{int(layer_idx)}H{int(head_idx)}",
            "source_head": f"L{source_layer}H{source_head_idx}",
            "q_read_score": float(torch.linalg.vector_norm(q_read_vec).item()),
            "k_source_score": float(torch.linalg.vector_norm(k_read_vec).item()),
            "q_weight_norm": q_norm,
            "k_weight_norm": k_norm,
            "q_read_score_norm": float(torch.linalg.vector_norm(q_read_vec).item() / max(q_norm, 1e-8)),
            "k_source_score_norm": float(torch.linalg.vector_norm(k_read_vec).item() / max(k_norm, 1e-8)),
            "ov_qk_frob_norm": float(torch.linalg.matrix_norm(ov_qk, ord="fro").item()),
            "ov_qk_spectral_norm": float(torch.linalg.matrix_norm(ov_qk, ord=2).item()),
            "source_ov_frob_norm": source_ov_fro,
            "source_ov_spectral_norm": float(source_ov_spectral),
        }
        score_terms = [
            row["q_read_score_norm"],
            row["k_source_score_norm"],
            row["ov_qk_frob_norm"] / max(source_ov_fro, 1e-8),
        ]
        row["ov_qk_composite_score"] = float(np.mean(score_terms))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["ov_qk_composite_score", "ov_qk_frob_norm", "q_read_score_norm", "k_source_score_norm"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _mean_eval_metrics(eval_df: pd.DataFrame, metric_cols: Sequence[str]) -> dict[str, float]:
    metrics = {}
    for col in metric_cols:
        metrics[col] = float(eval_df[col].mean()) if col in eval_df.columns and not eval_df.empty else np.nan
    if "Dx" in eval_df.columns and not eval_df.empty:
        metrics["Dx_mean"] = float(eval_df["Dx"].mean())
        metrics["Dx_abs_mean"] = float(eval_df["Dx"].abs().mean())
    else:
        metrics["Dx_mean"] = np.nan
        metrics["Dx_abs_mean"] = np.nan
    if "Dy" in eval_df.columns and not eval_df.empty:
        metrics["Dy_mean"] = float(eval_df["Dy"].mean())
        metrics["Dy_abs_mean"] = float(eval_df["Dy"].abs().mean())
    else:
        metrics["Dy_mean"] = np.nan
        metrics["Dy_abs_mean"] = np.nan
    return metrics


def run_pair_ablation_grid(
    *,
    pipeline,
    ckptdir: str,
    ckpt_list: Sequence[str],
    prompts: Sequence[str],
    scene_infos: Sequence[dict],
    embedding_cache: dict,
    source_head: tuple[int, int],
    candidate_heads: Sequence[tuple[int, int]],
    state_dict_convert,
    device,
    weight_dtype,
    num_images: int = 5,
    num_inference_steps: int = 14,
    guidance_scale: float = 4.5,
    generator_seed: int = 42,
    metric_cols: Sequence[str] | None = None,
    show_prompt_progress: bool = False,
    progress_mode: str = "tqdm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate baseline, source-only, candidate-only, and source+candidate conditions
    across checkpoints and return both a summary dataframe and the per-image eval rows.
    """
    if metric_cols is None:
        metric_cols = DEFAULT_BEHAVIOR_COLS

    conditions = [HeadCondition("baseline", "Baseline", tuple())]
    for cand in candidate_heads:
        cand = tuple(map(int, cand))
        conditions.extend(
            [
                HeadCondition(f"src__{cand[0]}_{cand[1]}", f"Source only vs L{cand[0]}H{cand[1]}", (tuple(map(int, source_head)),)),
                HeadCondition(f"cand__{cand[0]}_{cand[1]}", f"Candidate only L{cand[0]}H{cand[1]}", (cand,)),
                HeadCondition(
                    f"pair__{cand[0]}_{cand[1]}",
                    f"Source + L{cand[0]}H{cand[1]}",
                    (tuple(map(int, source_head)), cand),
                ),
            ]
        )

    n_prompts = len(prompts)
    total_steps = len(ckpt_list) * len(conditions) * n_prompts
    if progress_mode not in {"tqdm", "print", "none"}:
        raise ValueError(f"Unknown progress_mode: {progress_mode}")
    pbar = tqdm(total=total_steps, desc="Pair ablations", unit="prompt", mininterval=2) if progress_mode == "tqdm" else None

    summary_rows: list[dict] = []
    eval_rows: list[pd.DataFrame] = []

    def _run_eval(layer_head_pairs: Sequence[tuple[int, int]]):
        orig = None
        if layer_head_pairs:
            orig = apply_zero_head_ablation_multi(pipeline.transformer, list(layer_head_pairs))
        try:
            eval_df, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipeline,
                prompts,
                scene_infos,
                embedding_cache,
                num_images=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator_seed=generator_seed,
                device=device,
                weight_dtype=weight_dtype,
                show_prompt_progress=show_prompt_progress,
                progress_callback=(lambda: pbar.update(1)) if pbar is not None else None,
            )
        finally:
            if orig is not None:
                restore_processors(pipeline.transformer, orig)
        return eval_df

    for ckpt_idx, ckpt_name in enumerate(ckpt_list, start=1):
        ckpt = torch.load(join(ckptdir, ckpt_name), map_location="cpu", weights_only=False)
        pipeline.transformer.load_state_dict(state_dict_convert(ckpt["state_dict_ema"]))
        pipeline.transformer = pipeline.transformer.to(device=device, dtype=weight_dtype)
        del ckpt
        gc.collect()

        epoch = int(basename(ckpt_name).split("_")[1])
        baseline_metrics = None
        for cond_idx, cond in enumerate(conditions, start=1):
            if pbar is not None:
                pbar.set_postfix(epoch=epoch, condition=cond.condition_key)
            elif progress_mode == "print":
                print(
                    f"[pair ablations] checkpoint {ckpt_idx}/{len(ckpt_list)} "
                    f"(epoch {epoch}) | condition {cond_idx}/{len(conditions)}: {cond.condition_key}",
                    flush=True,
                )
            eval_df = _run_eval(cond.layer_head_pairs)
            metrics = _mean_eval_metrics(eval_df, metric_cols)
            if cond.condition_key == "baseline":
                baseline_metrics = metrics
            row = {
                "epoch": epoch,
                "checkpoint": basename(ckpt_name),
                "condition_key": cond.condition_key,
                "condition_label": cond.condition_label,
                "source_head": f"L{int(source_head[0])}H{int(source_head[1])}",
                "candidate_head": None,
                "ablated_heads": head_pairs_to_str(cond.layer_head_pairs),
                **metrics,
            }
            if cond.condition_key.startswith(("src__", "cand__", "pair__")):
                parts = cond.condition_key.split("__", 1)[1].split("_")
                row["candidate_head"] = f"L{int(parts[0])}H{int(parts[1])}"
            if baseline_metrics is not None:
                for col in metric_cols:
                    row[f"{col}_delta_from_baseline"] = baseline_metrics.get(col, np.nan) - metrics.get(col, np.nan)
            summary_rows.append(row)
            eval_df = eval_df.copy()
            eval_df["epoch"] = epoch
            eval_df["checkpoint"] = basename(ckpt_name)
            eval_df["condition_key"] = cond.condition_key
            eval_df["condition_label"] = cond.condition_label
            eval_rows.append(eval_df)

    if pbar is not None:
        pbar.close()
    return pd.DataFrame(summary_rows), (pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame())


def summarize_functional_effects(
    pair_ablation_df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] | None = None,
    show_progress: bool = False,
    progress_desc: str = "Summarize functional effects",
) -> pd.DataFrame:
    """
    Collapse pair-ablation results into candidate-level functional role summaries.
    """
    if metric_cols is None:
        metric_cols = DEFAULT_BEHAVIOR_COLS
    if pair_ablation_df.empty:
        return pd.DataFrame()

    rows = []
    df = pair_ablation_df.copy()
    grouped_items = list(df[df["candidate_head"].notna()].groupby("candidate_head"))
    iterator = tqdm(grouped_items, desc=progress_desc, unit="head", mininterval=1) if show_progress else grouped_items
    for candidate_head, dfg in iterator:
        row = {"candidate_head": candidate_head}
        for cond_key, prefix in [("src__", "source_only"), ("cand__", "candidate_only"), ("pair__", "pair")]:
            sub = dfg[dfg["condition_key"].str.startswith(cond_key)]
            if sub.empty:
                continue
            for metric in metric_cols:
                delta_col = f"{metric}_delta_from_baseline"
                if delta_col in sub.columns:
                    row[f"{prefix}_{metric}_delta_mean"] = float(sub[delta_col].mean())
            for metric in ["Dx_abs_mean", "Dy_abs_mean"]:
                if metric in sub.columns and metric in df.columns:
                    base_vals = (
                        df[(df["epoch"].isin(sub["epoch"])) & (df["condition_key"] == "baseline")]
                        .set_index("epoch")[metric]
                    )
                    merged = sub.set_index("epoch")[metric].to_frame("metric").join(base_vals.to_frame("base"), how="left")
                    if not merged.empty:
                        row[f"{prefix}_{metric}_delta_mean"] = float((merged["base"] - merged["metric"]).mean())

        binding_terms = [
            row.get("candidate_only_unique_binding_delta_mean", np.nan),
            row.get("candidate_only_exist_binding_delta_mean", np.nan),
        ]
        location_terms = [
            row.get("candidate_only_spatial_relationship_delta_mean", np.nan),
            row.get("candidate_only_Dx_abs_mean_delta_mean", np.nan),
            row.get("candidate_only_Dy_abs_mean_delta_mean", np.nan),
        ]
        row["binding_reader_score"] = float(np.nanmean(binding_terms)) if np.isfinite(np.nanmean(binding_terms)) else np.nan
        row["location_reader_score"] = float(np.nanmean(location_terms)) if np.isfinite(np.nanmean(location_terms)) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def _role(row):
        b = row.get("binding_reader_score", np.nan)
        l = row.get("location_reader_score", np.nan)
        if not np.isfinite(b) and not np.isfinite(l):
            return "unknown"
        if np.isfinite(b) and np.isfinite(l):
            if b > l + 1e-6:
                return "binding_reader"
            if l > b + 1e-6:
                return "location_reader"
            return "mixed_reader"
        return "binding_reader" if np.isfinite(b) else "location_reader"

    out["role_hypothesis"] = out.apply(_role, axis=1)
    return out.sort_values(["location_reader_score", "binding_reader_score"], ascending=[False, False]).reset_index(drop=True)
