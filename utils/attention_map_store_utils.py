"""
Attention visualization hooks (optional).

Stub implementations allow imports in setup notebooks; cells that need maps should
provide a real ``AttnProcessor2_0_Store`` implementation or use diffusers’ processors.
"""

from __future__ import annotations

from typing import Any


def replace_attn_processor(_model: Any, *args, **kwargs) -> None:
    raise NotImplementedError(
        "replace_attn_processor is not bundled in this minimal utils checkout; "
        "use diffusers ``set_attn_processor`` or copy the store processor from your full project."
    )


class AttnProcessor2_0_Store:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("AttnProcessor2_0_Store not implemented in minimal utils.")


class PixArtAttentionVisualizer_Store:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("PixArtAttentionVisualizer_Store not implemented in minimal utils.")
