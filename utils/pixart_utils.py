"""
PixArt-Alpha (training checkpoint) ↔ Diffusers ``Transformer2DModel`` utilities.

Checkpoint tensors use the PixArt-alpha training module names
(``blocks.*``, ``x_embedder``, ``y_embedder``, ...).  Diffusers packs the
same weights under ``transformer_blocks.*``, ``pos_embed.proj``, etc.
"""

from __future__ import annotations

import re
from os.path import dirname
from typing import Any, Mapping

import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from transformers import T5Tokenizer


# Keys: config.model string -> architecture for DiT-mini / XL style models used in this repo.
PixArt_model_configs: dict[str, dict[str, Any]] = {
    "PixArt_mini_2": dict(
        hidden_size=384,
        num_heads=6,
        depth=6,
        caption_channels=4096,
    ),
    # Fallbacks for notebooks that reference XL naming
    "PixArt_XL_2": dict(
        hidden_size=1152,
        num_heads=16,
        depth=28,
        caption_channels=4096,
    ),
}


def _arch_from_config(config: Any) -> dict[str, Any]:
    name = getattr(config, "model", None) or "PixArt_mini_2"
    if name not in PixArt_model_configs:
        raise KeyError(f"Unknown config.model {name!r}; extend PixArt_model_configs in utils/pixart_utils.py")
    return PixArt_model_configs[name]


def construct_diffuser_transformer_from_config(config: Any) -> Transformer2DModel:
    arch = _arch_from_config(config)
    hidden = int(arch["hidden_size"])
    depth = int(arch["depth"])
    heads = int(arch["num_heads"])
    head_dim = hidden // heads
    caption_channels = int(arch.get("caption_channels", getattr(config, "caption_channels", 4096)))
    patch_size = int(getattr(config, "patch_size", 2))
    image_size = int(config.image_size)
    sample_size = image_size // 8
    learn_sigma = True
    pred_sigma = getattr(config, "pred_sigma", True)
    if not pred_sigma:
        learn_sigma = False
    in_ch = 4
    out_ch = in_ch * 2 if learn_sigma and pred_sigma else in_ch

    return Transformer2DModel(
        sample_size=sample_size,
        num_layers=depth,
        attention_head_dim=head_dim,
        in_channels=in_ch,
        out_channels=out_ch,
        patch_size=patch_size,
        attention_bias=True,
        num_attention_heads=heads,
        cross_attention_dim=hidden,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=int(getattr(config, "train_sampling_steps", 1000)),
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=caption_channels,
    )


def state_dict_convert(src: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map PixArt-alpha ``state_dict`` / ``state_dict_ema`` tensors to diffusers
    ``Transformer2DModel`` keys.  Does **not** include ``pos_embed.pos_embed``
    (non-persistent buffer); use ``load_pixart_ema_into_transformer`` to copy it.
    """
    out: dict[str, torch.Tensor] = {}

    def cp(src_key: str, dst_key: str) -> None:
        out[dst_key] = src[src_key]

    # Patch embed + pos (conv weights only)
    if "x_embedder.proj.weight" in src:
        cp("x_embedder.proj.weight", "pos_embed.proj.weight")
    if "x_embedder.proj.bias" in src:
        cp("x_embedder.proj.bias", "pos_embed.proj.bias")

    # Timestep / adaLN-single
    for suf in ("weight", "bias"):
        k0 = f"t_embedder.mlp.0.{suf}"
        k2 = f"t_embedder.mlp.2.{suf}"
        if k0 in src:
            cp(k0, f"adaln_single.emb.timestep_embedder.linear_1.{suf}")
        if k2 in src:
            cp(k2, f"adaln_single.emb.timestep_embedder.linear_2.{suf}")
    if "t_block.1.weight" in src:
        cp("t_block.1.weight", "adaln_single.linear.weight")
    if "t_block.1.bias" in src:
        cp("t_block.1.bias", "adaln_single.linear.bias")

    # Caption projection (CaptionEmbedder MLP)
    for suf in ("weight", "bias"):
        if f"y_embedder.y_proj.fc1.{suf}" in src:
            cp(f"y_embedder.y_proj.fc1.{suf}", f"caption_projection.linear_1.{suf}")
        if f"y_embedder.y_proj.fc2.{suf}" in src:
            cp(f"y_embedder.y_proj.fc2.{suf}", f"caption_projection.linear_2.{suf}")

    # Final layer
    if "final_layer.scale_shift_table" in src:
        cp("final_layer.scale_shift_table", "scale_shift_table")
    for suf in ("weight", "bias"):
        k = f"final_layer.linear.{suf}"
        if k in src:
            cp(k, f"proj_out.{suf}")

    # Transformer blocks
    pat = re.compile(r"^blocks\.(\d+)\.")
    block_keys = [k for k in src if pat.match(k)]
    indices = sorted({int(pat.match(k).group(1)) for k in block_keys})  # type: ignore[union-attr]

    for i in indices:
        pre = f"blocks.{i}."
        dst_b = f"transformer_blocks.{i}."
        if f"{pre}scale_shift_table" in src:
            cp(f"{pre}scale_shift_table", f"{dst_b}scale_shift_table")

        # Self-attention: fused QKV
        wqkv = f"{pre}attn.qkv.weight"
        bqkv = f"{pre}attn.qkv.bias"
        if wqkv in src:
            w = src[wqkv]
            dim = w.shape[1]
            wq, wk, wv = w.chunk(3, dim=0)
            out[f"{dst_b}attn1.to_q.weight"] = wq
            out[f"{dst_b}attn1.to_k.weight"] = wk
            out[f"{dst_b}attn1.to_v.weight"] = wv
        if bqkv in src:
            b = src[bqkv]
            bq, bk, bv = b.chunk(3, dim=0)
            out[f"{dst_b}attn1.to_q.bias"] = bq
            out[f"{dst_b}attn1.to_k.bias"] = bk
            out[f"{dst_b}attn1.to_v.bias"] = bv
        for suf in ("weight", "bias"):
            k = f"{pre}attn.proj.{suf}"
            if k in src:
                cp(k, f"{dst_b}attn1.to_out.0.{suf}")

        # Cross-attention
        for suf in ("weight", "bias"):
            kq = f"{pre}cross_attn.q_linear.{suf}"
            if kq in src:
                cp(kq, f"{dst_b}attn2.to_q.{suf}")
        wkv = f"{pre}cross_attn.kv_linear.weight"
        bkv = f"{pre}cross_attn.kv_linear.bias"
        if wkv in src:
            w = src[wkv]
            dim = w.shape[1]
            wk, wv = w.split(dim, dim=0)
            out[f"{dst_b}attn2.to_k.weight"] = wk
            out[f"{dst_b}attn2.to_v.weight"] = wv
        if bkv in src:
            b = src[bkv]
            bk, bv = b.split(b.shape[0] // 2, dim=0)
            out[f"{dst_b}attn2.to_k.bias"] = bk
            out[f"{dst_b}attn2.to_v.bias"] = bv
        for suf in ("weight", "bias"):
            k = f"{pre}cross_attn.proj.{suf}"
            if k in src:
                cp(k, f"{dst_b}attn2.to_out.0.{suf}")

        # FFN: timm MLP fc1/fc2 ↔ diffusers FeedForward gelu-approximate
        for suf in ("weight", "bias"):
            k1 = f"{pre}mlp.fc1.{suf}"
            k2 = f"{pre}mlp.fc2.{suf}"
            if k1 in src:
                cp(k1, f"{dst_b}ff.net.0.proj.{suf}")
            if k2 in src:
                cp(k2, f"{dst_b}ff.net.2.{suf}")

    return out


def load_pixart_ema_into_transformer(
    transformer: Transformer2DModel,
    state_dict_ema: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Load converted weights and copy the learned positional embedding buffer
    (omitted from diffusers ``state_dict()`` because it is non-persistent).
    """
    converted = state_dict_convert(state_dict_ema)
    missing, unexpected = transformer.load_state_dict(converted, strict=strict)
    if "pos_embed" in state_dict_ema:
        transformer.pos_embed.pos_embed.data.copy_(
            state_dict_ema["pos_embed"].to(device=transformer.pos_embed.pos_embed.device)
        )
    return missing, unexpected


def construct_diffuser_pipeline_from_config(
    config: Any,
    *,
    ckpt_path: str | None = None,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float16,
    use_preview_generator: bool = False,
) -> PixArtAlphaPipeline:
    """
    Build a ``PixArtAlphaPipeline`` (HuggingFace) from a training ``config.py``.

    Downloads the published PixArt-XL skeleton from the Hub for tokenizer/VAE
    scaffolding (small JSON + shared VAE path).  Optionally loads a converted
    transformer from ``ckpt_path``.
    """
    _ = use_preview_generator  # reserved; diffusers' ``from_pretrained`` supports generator where needed.

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    transformer = construct_diffuser_transformer_from_config(config)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_pixart_ema_into_transformer(transformer, ckpt["state_dict_ema"], strict=True)

    model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    pipe: PixArtAlphaPipeline = PixArtAlphaPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    pipe.register_to_config(force_download=False)
    pipe.to(device)
    return pipe


class PixArtAlphaPipeline_custom(PixArtAlphaPipeline):
    """Thin alias kept for notebook compatibility."""


class PixArtAlphaPipeline_custom_CLIP(PixArtAlphaPipeline):
    """Placeholder; CLIP-control experiments are not used in split workflows."""


def pipeline_inference_custom(pipeline: PixArtAlphaPipeline, *args: Any, **kwargs: Any):
    """Delegate to ``pipeline``; exists so old notebook imports keep working."""
    return pipeline(*args, **kwargs)


__all__ = [
    "PixArt_model_configs",
    "construct_diffuser_transformer_from_config",
    "construct_diffuser_pipeline_from_config",
    "state_dict_convert",
    "load_pixart_ema_into_transformer",
    "PixArtAlphaPipeline_custom",
    "PixArtAlphaPipeline_custom_CLIP",
    "pipeline_inference_custom",
]
