"""
Evaluate a PixArt/DiT pipeline using pre-computed T5 (or other) text embeddings.

This is the **single source of truth** for cached-embedding eval used by:
- ``experimental_scripts/run_generalization_eval_local.py``
- ``experimental_scripts/generalization_profile_eval_cli.py`` (re-exports)
- Notebooks (import from here instead of the CLI to avoid path/sys.path issues)

**Important:** When passing ``prompt_embeds`` / ``negative_prompt_embeds``, diffusers also
requires ``prompt=None`` and ``negative_prompt=None`` — otherwise generation can error or
behave inconsistently.

Default ``guidance_scale=4.5`` matches typical PixArt training; ``1.0`` often collapses
conditional quality and can drive **~0%** spatial accuracy on object-relation prompts.
"""

from __future__ import annotations

import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable, Optional


def evaluate_pipeline_on_prompts_with_cached_embeddings(
    pipeline,
    prompts,
    scene_infos,
    embedding_cache,
    num_images: int = 49,
    num_inference_steps: int = 14,
    guidance_scale: float = 4.5,
    generator_seed: int = 42,
    color_margin: int = 25,
    spatial_threshold: int = 5,
    device=None,
    weight_dtype=None,
    show_prompt_progress: bool = True,
    progress_callback: Optional[Callable] = None,
):
    """
    Generate images with cached caption embeddings and run OpenCV + parametric relation eval.

    Returns
    -------
    eval_df : pandas.DataFrame
        One row per generated image with columns from ``evaluate_parametric_relation``
        (``overall``, ``shape``, ``color``, ``exist_binding``, ``unique_binding``,
        ``spatial_relationship``, ``spatial_relationship_loose``, ``Dx``, ``Dy``, …).
    object_df : pandas.DataFrame
        Stacked detection rows (all contours) with ``prompt_id``, ``sample_id``, ``prompt``.
    """
    from utils.cv2_eval_utils import find_classify_objects, evaluate_parametric_relation

    if not show_prompt_progress:
        pipeline.set_progress_bar_config(disable=True)

    if len(prompts) != len(scene_infos):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match scene_infos ({len(scene_infos)})"
        )

    if device is None:
        device = next(pipeline.transformer.parameters()).device
    if weight_dtype is None:
        weight_dtype = pipeline.transformer.dtype

    uncond_data = embedding_cache[""]
    uncond_prompt_embeds = uncond_data["caption_embeds"].to(device)
    uncond_prompt_attention_mask = uncond_data["emb_mask"].to(device)

    all_eval_results = []
    all_object_results = []

    gen_device = "cpu" if str(device) == "mps" else device

    for prompt_id, (prompt, scene_info) in tqdm(
        enumerate(zip(prompts, scene_infos)),
        desc="Evaluating prompts",
        total=len(prompts),
        disable=not show_prompt_progress,
    ):
        cached_data = None
        for cache_key in embedding_cache.keys():
            if cache_key != "" and cache_key.endswith(f"::{prompt}"):
                cached_data = embedding_cache[cache_key]
                break

        if cached_data is None:
            print(f"Warning: No cached embeddings found for prompt: '{prompt}'")
            if progress_callback is not None:
                progress_callback()
            continue

        caption_embeds = cached_data["caption_embeds"].to(device)
        emb_mask = cached_data["emb_mask"].to(device)

        try:
            out = pipeline(
                prompt=None,
                negative_prompt=None,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device=gen_device).manual_seed(generator_seed),
                guidance_scale=guidance_scale,
                prompt_embeds=caption_embeds,
                prompt_attention_mask=emb_mask,
                negative_prompt_embeds=uncond_prompt_embeds,
                negative_prompt_attention_mask=uncond_prompt_attention_mask,
                use_resolution_binning=False,
                prompt_dtype=weight_dtype,
                verbose=False,
            )

            for sample_id, image in enumerate(out.images):
                try:
                    classified_objects_df = find_classify_objects(image)
                    eval_result = evaluate_parametric_relation(
                        classified_objects_df,
                        scene_info,
                        color_margin=color_margin,
                        spatial_threshold=spatial_threshold,
                    )
                    eval_record = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "sample_id": sample_id,
                        **eval_result,
                    }
                    all_eval_results.append(eval_record)

                    classified_objects_df = classified_objects_df.copy()
                    classified_objects_df["prompt_id"] = prompt_id
                    classified_objects_df["sample_id"] = sample_id
                    classified_objects_df["prompt"] = prompt
                    all_object_results.append(classified_objects_df)
                except Exception as e:
                    print(f"Error evaluating sample {sample_id} for prompt '{prompt}': {e}")
                    continue
            if progress_callback is not None:
                progress_callback()

        except Exception as e:
            print(f"Error generating images for prompt '{prompt}': {e}")
            if progress_callback is not None:
                progress_callback()
            continue

    eval_df = pd.DataFrame(all_eval_results)
    object_df = (
        pd.concat(all_object_results, ignore_index=True)
        if all_object_results
        else pd.DataFrame()
    )
    return eval_df, object_df
