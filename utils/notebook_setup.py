"""
Shared setup utilities for NeurIPS experiment notebooks (05-08).

Consolidates model loading, embedding cache, word vector extraction,
variance partition, head alignment computation, and head selection
so each notebook avoids ~200 lines of duplicated boilerplate.
"""

from __future__ import annotations

import os, sys, gc, time
from os.path import join
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Path helpers
# ---------------------------------------------------------------------------

def setup_paths(project_root: Optional[str] = None):
    """Add project root and PixArt-alpha to sys.path.  Returns project_root."""
    if project_root is None:
        project_root = os.path.abspath(join(os.path.dirname(__file__), ".."))
    for p in [project_root, join(project_root, "PixArt-alpha")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return project_root


def get_device():
    """Return (device_str, compute_dtype) with CUDA > MPS > CPU fallback."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


# ---------------------------------------------------------------------------
#  Model / pipeline loading
# ---------------------------------------------------------------------------

def load_model_and_pipeline(run_name: str, ckpt_epoch: int, ckpt_step: int,
                            project_root: Optional[str] = None):
    """
    Load PixArt-mini transformer + assemble lightweight pipeline.

    Returns
    -------
    transformer, pipe, tokenizer, device, compute_dtype
    """
    project_root = setup_paths(project_root)
    device, compute_dtype = get_device()

    from diffusion.utils.misc import read_config
    from utils.pixart_utils import (
        construct_diffuser_transformer_from_config,
        load_pixart_ema_into_transformer,
    )
    from utils.pixart_sampling_utils import PixArtAlphaPipeline_custom
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
    from transformers import T5Tokenizer

    savedir = join(project_root, "results", run_name)
    config = read_config(join(savedir, "config.py"))

    transformer = construct_diffuser_transformer_from_config(config)
    ckpt_path = join(savedir, "checkpoints",
                     f"epoch_{ckpt_epoch}_step_{ckpt_step}.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_pixart_ema_into_transformer(transformer, ckpt["state_dict_ema"])
    del ckpt; gc.collect()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema",
                                        torch_dtype=compute_dtype)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    scheduler = DPMSolverMultistepScheduler()

    pipe = PixArtAlphaPipeline_custom(
        transformer=transformer, vae=vae, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=None,
    )
    pipe.transformer = pipe.transformer.to(device=device, dtype=compute_dtype)
    pipe.vae = pipe.vae.to(device=device, dtype=compute_dtype)
    gc.collect()

    print(f"Pipeline ready on {device} ({compute_dtype})")
    return transformer, pipe, tokenizer, device, compute_dtype


# ---------------------------------------------------------------------------
#  Embedding cache
# ---------------------------------------------------------------------------

def load_embedding_cache(run_name: str, project_root: Optional[str] = None):
    """Load pre-computed T5 embedding cache.  Returns dict."""
    if project_root is None:
        project_root = setup_paths()
    cache_path = join(project_root, "results", run_name, "t5_embedding_cache.pt")
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = raw["embedding_allrel_allobj"]
    del raw; gc.collect()
    print(f"Loaded {len(cache)} cached embeddings")
    return cache


# ---------------------------------------------------------------------------
#  Prompt / scene-info generation
# ---------------------------------------------------------------------------

def generate_prompts_and_scene_info(spatial_phrases=None):
    """
    Build the full combinatorial prompt set (same logic as NB01).

    Returns (prompts, scene_infos, scene_info_df)
    """
    if spatial_phrases is None:
        from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES
        spatial_phrases = DEFAULT_SPATIAL_PHRASES

    colors = ["red", "blue"]
    shapes = ["square", "triangle", "circle"]
    prompts, scene_infos = [], []

    for c1, c2 in product(colors, colors):
        if c1 == c2:
            continue
        for s1, s2 in product(shapes, shapes):
            if s1 == s2:
                continue
            for rel, texts in spatial_phrases.items():
                if rel in ("in_front", "behind"):
                    continue
                for text in texts:
                    prompts.append(f"{c1} {s1} is {text} {c2} {s2}")
                    scene_infos.append(dict(
                        color1=c1, shape1=s1, color2=c2, shape2=s2,
                        spatial_relationship=rel,
                    ))

    df = pd.DataFrame(scene_infos)
    df["color1shape1"] = df["color1"] + "_" + df["shape1"]
    df["color2shape2"] = df["color2"] + "_" + df["shape2"]
    return prompts, scene_infos, df


# ---------------------------------------------------------------------------
#  Token helpers
# ---------------------------------------------------------------------------

def find_token_index(tokens: list[str], word: str) -> Optional[int]:
    """Find token index for *word* in a T5-tokenized list."""
    w = word.strip().lower()
    for i, tok in enumerate(tokens):
        t = tok.strip().lower().lstrip("\u2581")   # T5 sentencepiece prefix
        if t == w:
            return i
    for i, tok in enumerate(tokens):
        if w in tok.strip().lower():
            return i
    return None


# ---------------------------------------------------------------------------
#  Word-vector extraction & projection
# ---------------------------------------------------------------------------

def extract_projected_word_vectors(
    embedding_cache: dict,
    transformer,
    tokenizer,
    prompts: list[str],
    scene_infos: list[dict],
    *,
    target_object: str = "shape2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each prompt, extract the *target_object* token's T5 embedding
    and project it through ``transformer.caption_projection``.

    Returns (raw_4096, projected_384) as numpy arrays, shape (N, dim).
    """
    device_orig = next(transformer.parameters()).device
    dtype_orig = next(transformer.parameters()).dtype
    cap_proj = transformer.caption_projection.cpu().float()

    raw_vecs = []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.tokenize(prompt)
        target_word = scene_infos[i][target_object]
        idx = find_token_index(tokens, target_word)
        if idx is None:
            idx = max(len(tokens) - 2, 0)

        # Look up cached embedding
        emb_data = None
        for k in embedding_cache:
            if k == "":
                continue
            if k.endswith(f"::{prompt}") or k == prompt:
                emb_data = embedding_cache[k]
                break
        if emb_data is None:
            for k in embedding_cache:
                if prompt in str(k):
                    emb_data = embedding_cache[k]
                    break

        if emb_data is not None:
            caption_emb = emb_data["caption_embeds"]        # (seq_len, 4096)
            if idx < caption_emb.shape[0]:
                raw_vecs.append(caption_emb[idx].float())
            else:
                raw_vecs.append(caption_emb[-2].float())
        else:
            raw_vecs.append(torch.zeros(4096))

    raw_mat = torch.stack(raw_vecs)                          # (N, 4096)
    with torch.no_grad():
        proj_mat = cap_proj(raw_mat)                         # (N, 384)

    cap_proj.to(device_orig, dtype=dtype_orig)
    return raw_mat.numpy(), proj_mat.detach().numpy()


# ---------------------------------------------------------------------------
#  Head alignment & selection
# ---------------------------------------------------------------------------

def compute_head_alignment(
    transformer,
    wordvec_proj: np.ndarray,
    scene_info_df: pd.DataFrame,
    vp_features: list[str],
    *,
    n_perm: int = 100,
    base_size: int = 8,
    verbose: bool = True,
):
    """
    Run variance partition, then compute spatial-ramp alignment for every
    cross-attention head.

    Returns
    -------
    align_df : DataFrame  (layer, head, cosine, projection, energy, abs_cosine)
    effect_vecs : dict
    levels_map : dict
    r2_total : float
    pos_embed_2d : Tensor  (N_patches, hidden_size)
    ramp_templates : dict   direction_name -> Tensor(N_patches)
    """
    from utils.variance_partition_with_effects import variance_partition_with_effects
    from utils.pixart_pos_embed import get_2d_sincos_pos_embed

    vp_df, intercept, effect_vecs, levels_map, r2_total = \
        variance_partition_with_effects(
            wordvec_proj, scene_info_df[vp_features],
            n_perm=n_perm, verbose=verbose,
        )

    hidden_size = transformer.config.hidden_size
    pos_embed_2d = torch.tensor(
        get_2d_sincos_pos_embed(hidden_size, base_size),
        dtype=torch.float32,
    )

    directions = {
        "above": (-1, 0), "below": (1, 0),
        "left_of": (0, -1), "right_of": (0, 1),
        "upper_left": (-1, -1), "upper_right": (-1, 1),
        "lower_left": (1, -1), "lower_right": (1, 1),
    }
    ramp_templates = {}
    for name, (dy, dx) in directions.items():
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, base_size),
            np.linspace(-1, 1, base_size),
            indexing="ij",
        )
        ramp = (dy * yy + dx * xx).astype(np.float32)
        ramp -= ramp.mean()
        ramp_templates[name] = torch.tensor(ramp.flatten())

    n_layers = len(transformer.transformer_blocks)
    n_heads = transformer.config.num_attention_heads
    head_dim = hidden_size // n_heads

    rows = []
    for li in range(n_layers):
        blk = transformer.transformer_blocks[li]
        W_k = blk.attn2.to_k.weight.detach().float().cpu()
        W_q = blk.attn2.to_q.weight.detach().float().cpu()
        for hi in range(n_heads):
            s = slice(hi * head_dim, (hi + 1) * head_dim)
            W_k_h = W_k[s, :]
            W_q_h = W_q[s, :]
            best_cos, best_proj, best_energy = 0.0, 0.0, 0.0

            for rel_key, ev_arr in effect_vecs.items():
                if not rel_key.startswith("spatial_relationship"):
                    continue
                # ev_arr shape: (n_levels, dim) — use each row
                if ev_arr.ndim == 1:
                    ev_arr = ev_arr[np.newaxis, :]
                for ev_row in ev_arr:
                    ev_t = torch.tensor(ev_row, dtype=torch.float32)
                    k_proj = W_k_h @ ev_t
                    for dir_name, ramp in ramp_templates.items():
                        qk_score = pos_embed_2d @ W_q_h.T @ k_proj
                        qk_score = qk_score - qk_score.mean()
                        cos = F.cosine_similarity(
                            qk_score.unsqueeze(0), ramp.unsqueeze(0),
                        ).item()
                        proj = (qk_score @ ramp).item() / (ramp.norm().item() + 1e-12)
                        energy = qk_score.norm().item()
                        if abs(cos) > abs(best_cos):
                            best_cos = cos
                            best_proj = proj
                            best_energy = energy

            rows.append(dict(layer=li, head=hi, cosine=best_cos,
                             projection=best_proj, energy=best_energy))

    align_df = pd.DataFrame(rows)
    align_df["abs_cosine"] = align_df["cosine"].abs()
    return align_df, effect_vecs, levels_map, r2_total, pos_embed_2d, ramp_templates


def select_spatial_and_control_heads(align_df: pd.DataFrame,
                                     n_spatial: int = 3, n_control: int = 3):
    """Pick top-N spatial heads (highest |cosine|) and bottom-N control heads."""
    spatial = (align_df.nlargest(n_spatial, "abs_cosine")
               [["layer", "head"]].values.tolist())
    control = (align_df.nsmallest(n_control, "abs_cosine")
               [["layer", "head"]].values.tolist())
    return [tuple(x) for x in spatial], [tuple(x) for x in control]


# ---------------------------------------------------------------------------
#  Weight extraction helpers
# ---------------------------------------------------------------------------

def get_head_weights(transformer, layer_idx: int, head_idx: int):
    """
    Return per-head weight matrices (all float32, CPU).

    W_q_h : (head_dim, hidden_size)
    W_k_h : (head_dim, hidden_size)
    W_v_h : (head_dim, hidden_size)
    W_o_h : (hidden_size, head_dim)
    """
    blk = transformer.transformer_blocks[layer_idx]
    hd = transformer.config.hidden_size // transformer.config.num_attention_heads
    s = slice(head_idx * hd, (head_idx + 1) * hd)
    return (
        blk.attn2.to_q.weight[s, :].detach().float().cpu(),
        blk.attn2.to_k.weight[s, :].detach().float().cpu(),
        blk.attn2.to_v.weight[s, :].detach().float().cpu(),
        blk.attn2.to_out[0].weight[:, s].detach().float().cpu(),
    )


def compute_ov_matrix(transformer, layer_idx: int, head_idx: int):
    """W_OV = W_o @ W_v  maps text-space → residual-stream.  Shape (d, d)."""
    _, _, W_v, W_o = get_head_weights(transformer, layer_idx, head_idx)
    return W_o @ W_v          # (hidden_size, head_dim) @ (head_dim, hidden_size)


def compute_qk_matrix(transformer, layer_idx: int, head_idx: int):
    """W_QK = W_q^T @ W_k  bilinear form on (image, text).  Shape (d, d)."""
    W_q, W_k, _, _ = get_head_weights(transformer, layer_idx, head_idx)
    return W_q.T @ W_k        # (hidden_size, head_dim) @ (head_dim, hidden_size)


# ---------------------------------------------------------------------------
#  Custom attention processors for interventions
# ---------------------------------------------------------------------------

def _forward_attn_scaled(attn, hidden_states, encoder_hidden_states,
                         attention_mask, temb, target_heads, scale):
    """Cross-attention forward with per-head output scaling."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        bs, ch, h, w = hidden_states.shape
        hidden_states = hidden_states.view(bs, ch, h * w).transpose(1, 2)

    bs, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                      else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs)
        attention_mask = attention_mask.view(bs, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)
    enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and attn.norm_cross:
        enc = attn.norm_encoder_hidden_states(enc)
    key = attn.to_k(enc)
    value = attn.to_v(enc)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    sc = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * sc
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = F.softmax(scores, dim=-1)
    hidden_states = torch.matmul(probs, value)

    for h in target_heads:
        if 0 <= h < attn.heads:
            hidden_states[:, h] *= scale

    hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(bs, ch, h, w)
    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
    return hidden_states


class ScaledHeadProcessor:
    """Multiply specific head outputs by *scale* (1.0 = identity)."""
    def __init__(self, original, layer_idx, target_layer, target_heads, scale):
        self.original = original
        self.layer_idx = layer_idx
        self.target_layer = target_layer
        self.target_heads = list(target_heads)
        self.scale = float(scale)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx == self.target_layer and self.target_heads:
            return _forward_attn_scaled(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, self.target_heads, self.scale,
            )
        return self.original(attn, hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             attention_mask=attention_mask, temb=temb, *a, **kw)


def _forward_attn_capture(attn, hidden_states, encoder_hidden_states,
                          attention_mask, temb, storage: dict):
    """Cross-attention forward that stores attention weights."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        bs, ch, h, w = hidden_states.shape
        hidden_states = hidden_states.view(bs, ch, h * w).transpose(1, 2)
    bs, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                      else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs)
        attention_mask = attention_mask.view(bs, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and attn.norm_cross:
        enc = attn.norm_encoder_hidden_states(enc)
    key = attn.to_k(enc)
    value = attn.to_v(enc)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    sc = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * sc
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = F.softmax(scores, dim=-1)

    # Store attention weights (detach + CPU to save GPU memory)
    storage["attn_weights"] = probs.detach().cpu()

    hidden_states = torch.matmul(probs, value)
    hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(bs, ch, h, w)
    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
    return hidden_states


class AttentionCaptureProcessor:
    """Processor that runs normal cross-attention and stores attention weights."""
    def __init__(self, original, layer_idx, target_layer):
        self.original = original
        self.layer_idx = layer_idx
        self.target_layer = target_layer
        self.storage = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx == self.target_layer:
            return _forward_attn_capture(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, self.storage,
            )
        return self.original(attn, hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             attention_mask=attention_mask, temb=temb, *a, **kw)


def _forward_attn_patched(attn, hidden_states, encoder_hidden_states,
                          attention_mask, temb, target_heads, cached_head_out):
    """Cross-attention forward that patches specific heads with cached output."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        bs, ch, h, w = hidden_states.shape
        hidden_states = hidden_states.view(bs, ch, h * w).transpose(1, 2)
    bs, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                      else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs)
        attention_mask = attention_mask.view(bs, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and attn.norm_cross:
        enc = attn.norm_encoder_hidden_states(enc)
    key = attn.to_k(enc)
    value = attn.to_v(enc)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    sc = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * sc
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = F.softmax(scores, dim=-1)
    hidden_states = torch.matmul(probs, value)

    # Patch target heads with cached activations
    cached = cached_head_out.to(hidden_states.device, dtype=hidden_states.dtype)
    for h in target_heads:
        if 0 <= h < attn.heads:
            hidden_states[:, h] = cached[:, h]

    hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(bs, ch, h, w)
    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
    return hidden_states


class ActivationPatchProcessor:
    """Replace specific head outputs with cached activations from another prompt."""
    def __init__(self, original, layer_idx, target_layer, target_heads, cached_head_out):
        self.original = original
        self.layer_idx = layer_idx
        self.target_layer = target_layer
        self.target_heads = list(target_heads)
        self.cached_head_out = cached_head_out

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx == self.target_layer and self.target_heads:
            return _forward_attn_patched(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, self.target_heads, self.cached_head_out,
            )
        return self.original(attn, hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             attention_mask=attention_mask, temb=temb, *a, **kw)


# ---------------------------------------------------------------------------
#  Processor application helpers
# ---------------------------------------------------------------------------

def apply_scaled_heads(transformer, target_layer: int, target_heads: list[int],
                       scale: float):
    """Apply ScaledHeadProcessor. Returns original_processors dict for restore."""
    from utils.zero_head_ablation_utils import restore_processors
    orig = {}
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2"):
            orig[li] = blk.attn2.processor
            if li == target_layer:
                blk.attn2.processor = ScaledHeadProcessor(
                    blk.attn2.processor, li, target_layer, target_heads, scale,
                )
    return orig


def apply_attention_capture(transformer, target_layer: int):
    """Apply AttentionCaptureProcessor to one layer. Returns (orig_procs, capture_proc)."""
    orig = {}
    capture = None
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2"):
            orig[li] = blk.attn2.processor
            if li == target_layer:
                capture = AttentionCaptureProcessor(blk.attn2.processor, li, target_layer)
                blk.attn2.processor = capture
    return orig, capture


def apply_activation_patch(transformer, target_layer: int, target_heads: list[int],
                           cached_head_out: torch.Tensor):
    """Apply ActivationPatchProcessor. Returns orig_procs dict."""
    orig = {}
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2"):
            orig[li] = blk.attn2.processor
            if li == target_layer:
                blk.attn2.processor = ActivationPatchProcessor(
                    blk.attn2.processor, li, target_layer, target_heads, cached_head_out,
                )
    return orig


# ---------------------------------------------------------------------------
#  Figure / publication helpers
# ---------------------------------------------------------------------------

def set_publication_style():
    """Apply publication-quality matplotlib defaults."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def saveallforms(figdir, fname, figh=None, fmts=("png", "pdf")):
    """Save current figure (or *figh*) in multiple formats."""
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    fig = figh if figh is not None else plt.gcf()
    for fmt in fmts:
        fig.savefig(join(figdir, f"{fname}.{fmt}"), bbox_inches="tight", dpi=300)
