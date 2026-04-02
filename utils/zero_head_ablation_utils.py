"""
Zero-head ablation utilities for DiT/PixArt cross-attention.

Used for the Early Ablation Sweep experiment: at each checkpoint, ablate the candidate
relation head and measure spatial relation accuracy.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _forward_attn_with_zero_heads(attn, hidden_states, encoder_hidden_states, attention_mask,
                                  temb, target_head_indices):
    """
    Full cross-attention forward with specified heads zeroed.
    Mirrors diffusers CrossAttention logic but zeros target heads after attn @ value.
    """
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    scale = head_dim ** -0.5
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = F.softmax(attention_scores, dim=-1)
    hidden_states = torch.matmul(attention_probs, value)

    # Zero-ablate target heads
    for h in target_head_indices:
        if 0 <= h < attn.heads:
            hidden_states[:, h, :, :] = 0.0

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)

    return hidden_states


def _forward_attn_with_noisy_heads(attn, hidden_states, encoder_hidden_states, attention_mask,
                                   temb, target_head_indices, noise_scale: float):
    """
    Full cross-attention forward with specified heads perturbed by additive Gaussian noise.
    """
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    scale = head_dim ** -0.5
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = F.softmax(attention_scores, dim=-1)
    hidden_states = torch.matmul(attention_probs, value)

    # Add noise to target heads (perturbation sensitivity)
    for h in target_head_indices:
        if 0 <= h < attn.heads:
            noise = torch.randn_like(hidden_states[:, h, :, :], device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, h, :, :] = hidden_states[:, h, :, :] + noise_scale * noise

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)

    return hidden_states


class ZeroHeadAblationProcessorFull:
    """
    Processor that performs full cross-attention forward and zero-ablates specified heads.
    """

    def __init__(self, original_processor, layer_idx: int, target_layer_idx: int,
                 target_head_indices: List[int]):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.target_layer_idx = target_layer_idx
        self.target_head_indices = list(target_head_indices) if target_head_indices else []

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if (self.layer_idx == self.target_layer_idx and self.target_head_indices):
            return _forward_attn_with_zero_heads(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                self.target_head_indices,
            )
        return self.original_processor(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


class ZeroHeadAblationProcessorStepRange:
    """
    Zero-ablates specified heads only when current inference step is in [step_start, step_end].
    Pipeline must set transformer._current_inference_step = i before each call.
    """

    def __init__(self, original_processor, layer_idx: int, target_layer_idx: int,
                 target_head_indices: List[int], transformer, step_start: int, step_end: int):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.target_layer_idx = target_layer_idx
        self.target_head_indices = list(target_head_indices) if target_head_indices else []
        self.transformer = transformer
        self.step_start = int(step_start)
        self.step_end = int(step_end)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        current_step = getattr(self.transformer, "_current_inference_step", None)
        if current_step is None or not (self.step_start <= current_step <= self.step_end):
            return self.original_processor(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )
        if (self.layer_idx == self.target_layer_idx and self.target_head_indices):
            return _forward_attn_with_zero_heads(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                self.target_head_indices,
            )
        return self.original_processor(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


class NoiseHeadPerturbationProcessor:
    """
    Processor that adds small Gaussian noise to specified heads (perturbation sensitivity).
    """

    def __init__(self, original_processor, layer_idx: int, target_layer_idx: int,
                 target_head_indices: List[int], noise_scale: float = 0.1):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.target_layer_idx = target_layer_idx
        self.target_head_indices = list(target_head_indices) if target_head_indices else []
        self.noise_scale = float(noise_scale)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if (self.layer_idx == self.target_layer_idx and self.target_head_indices):
            return _forward_attn_with_noisy_heads(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                self.target_head_indices, self.noise_scale,
            )
        return self.original_processor(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


def apply_noise_head_perturbation(transformer, target_layer_idx: int, target_head_indices: List[int],
                                  noise_scale: float = 0.1):
    """
    Replace attn2 processors to add Gaussian noise to specified heads.
    Returns dict of original processors for restoration.
    """
    original_processors = {}
    for name, block in transformer.transformer_blocks.named_children():
        layer_idx = int(name)
        if hasattr(block, "attn2"):
            original_processors[layer_idx] = block.attn2.processor
            block.attn2.processor = NoiseHeadPerturbationProcessor(
                original_processor=block.attn2.processor,
                layer_idx=layer_idx,
                target_layer_idx=target_layer_idx,
                target_head_indices=target_head_indices,
                noise_scale=noise_scale,
            )
    return original_processors


def all_cross_attn_head_pairs(transformer) -> List[Tuple[int, int]]:
    """
    Return every (layer_idx, head_idx) for cross-attention (attn2) across all DiT blocks.
    Use with ``apply_zero_head_ablation_multi`` to zero **all** cross-attn heads (strongest intervention).
    """
    n_layers = len(transformer.transformer_blocks)
    n_heads = transformer.transformer_blocks[0].attn2.heads
    return [(layer, h) for layer in range(n_layers) for h in range(n_heads)]


def apply_zero_head_ablation_multi(transformer, layer_head_pairs: List[Tuple[int, int]]):
    """
    Ablate multiple (layer_idx, head_idx) pairs at once.
    layer_head_pairs: e.g. [(0, 2), (1, 3)] to ablate head 2 in layer 0 and head 3 in layer 1.
    Returns dict of original processors for restoration.
    """
    from collections import defaultdict
    layer_to_heads = defaultdict(list)
    for (l, h) in layer_head_pairs:
        layer_to_heads[l].append(h)
    original_processors = {}
    for name, block in transformer.transformer_blocks.named_children():
        layer_idx = int(name)
        if hasattr(block, "attn2"):
            original_processors[layer_idx] = block.attn2.processor
            heads_to_ablate = layer_to_heads.get(layer_idx, [])
            if heads_to_ablate:
                block.attn2.processor = ZeroHeadAblationProcessorFull(
                    original_processor=block.attn2.processor,
                    layer_idx=layer_idx,
                    target_layer_idx=layer_idx,
                    target_head_indices=heads_to_ablate,
                )
    return original_processors


def apply_zero_head_ablation(transformer, target_layer_idx: int, target_head_indices: List[int],
                            step_range: Optional[tuple] = None):
    """
    Replace attn2 processors for the given layer/heads.
    step_range: optional (step_start, step_end) to ablate only at those inference steps (inclusive).
    Returns dict of original processors for restoration.
    """
    original_processors = {}
    for name, block in transformer.transformer_blocks.named_children():
        layer_idx = int(name)
        if hasattr(block, "attn2"):
            original_processors[layer_idx] = block.attn2.processor
            if step_range is not None:
                step_start, step_end = step_range
                block.attn2.processor = ZeroHeadAblationProcessorStepRange(
                    original_processor=block.attn2.processor,
                    layer_idx=layer_idx,
                    target_layer_idx=target_layer_idx,
                    target_head_indices=target_head_indices,
                    transformer=transformer,
                    step_start=step_start,
                    step_end=step_end,
                )
            else:
                block.attn2.processor = ZeroHeadAblationProcessorFull(
                    original_processor=block.attn2.processor,
                    layer_idx=layer_idx,
                    target_layer_idx=target_layer_idx,
                    target_head_indices=target_head_indices,
                )
    return original_processors


def restore_processors(transformer, original_processors: dict):
    """Restore original attn2 processors after ablation."""
    for layer_idx, proc in original_processors.items():
        if hasattr(transformer.transformer_blocks[layer_idx], "attn2"):
            transformer.transformer_blocks[layer_idx].attn2.processor = proc
