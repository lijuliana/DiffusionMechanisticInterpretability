"""Vendored 2D sinusoidal position embeddings from PixArt (numpy-only).

Import this instead of ``diffusion.model.nets.PixArt`` when you only need
``get_2d_sincos_pos_embed``, so notebooks do not depend on xformers / full PixArt nets.
"""

from __future__ import annotations

import numpy as np


def _to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
    extra_tokens: int = 0,
    lewei_scale: float = 1.0,
    base_size: int = 16,
) -> np.ndarray:
    if isinstance(grid_size, int):
        grid_size = _to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
