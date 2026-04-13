"""
Lookup-table text encoders used for rndemb / pilot models (not T5).

Matches the interface expected by ``generalization_profile_eval_cli.precompute_embeddings``:
``encode(input_ids, attention_mask) -> (caption_embeds, emb_mask)``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_embedding_table(embedding_dict: Any) -> torch.Tensor:
    if isinstance(embedding_dict, torch.Tensor):
        return embedding_dict.float()
    if isinstance(embedding_dict, Mapping):
        keys = sorted(int(k) for k in embedding_dict.keys())
        rows = []
        for k in keys:
            v = embedding_dict[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=torch.float32)
            rows.append(v.reshape(-1))
        return torch.stack(rows, dim=0).float()
    raise TypeError(f"embedding_dict must be Tensor or mapping, got {type(embedding_dict)}")


def _tok2dict_tensor(input_ids2dict_ids: Any) -> torch.Tensor:
    if isinstance(input_ids2dict_ids, torch.Tensor):
        return input_ids2dict_ids.long().contiguous()
    if isinstance(input_ids2dict_ids, Mapping):
        if not input_ids2dict_ids:
            raise ValueError("input_ids2dict_ids is empty")
        max_tok = max(int(k) for k in input_ids2dict_ids.keys())
        t = torch.zeros(max_tok + 1, dtype=torch.long)
        for k, v in input_ids2dict_ids.items():
            t[int(k)] = int(v)
        return t
    try:
        import numpy as np

        if isinstance(input_ids2dict_ids, np.ndarray):
            return torch.from_numpy(input_ids2dict_ids.astype("int64", copy=False)).long()
    except ImportError:
        pass
    raise TypeError(f"Unsupported input_ids2dict_ids type: {type(input_ids2dict_ids)}")


class RandomEmbeddingEncoder(nn.Module):
    """
    Map tokenizer ids → rows of a fixed embedding table (pilot rndemb checkpoints).

    Parameters
    ----------
    embedding_dict : Tensor [N, D] or dict[int, Tensor]
    input_ids2dict_ids : LongTensor [V] or dict[int, int]  (tokenizer id → table row)
    dict_ids2input_ids : unused; kept for checkpoint format parity
    """

    def __init__(
        self,
        embedding_dict: Any,
        input_ids2dict_ids: Any,
        dict_ids2input_ids: Any | None = None,
    ):
        super().__init__()
        _ = dict_ids2input_ids
        weight = _stack_embedding_table(embedding_dict)
        self.register_parameter("embed_weight", nn.Parameter(weight, requires_grad=False))
        self.register_buffer("tok2dict", _tok2dict_tensor(input_ids2dict_ids), persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.encode(input_ids, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.embed_weight
        tmap = self.tok2dict.to(device=input_ids.device)
        safe = input_ids.long().clamp(0, tmap.shape[0] - 1)
        dix = tmap[safe].clamp(0, w.shape[0] - 1)
        emb = F.embedding(dix, w)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.long()
        return emb, attention_mask


class RandomEmbeddingEncoder_wPosEmb(RandomEmbeddingEncoder):
    """Adds learned absolute position embeddings (scaled) on top of ``RandomEmbeddingEncoder``."""

    def __init__(
        self,
        embedding_dict: Any,
        input_ids2dict_ids: Any,
        dict_ids2input_ids: Any | None = None,
        *,
        max_seq_len: int = 20,
        embed_dim: int | None = None,
        wpe_scale: float = 1.0 / 6.0,
    ):
        super().__init__(embedding_dict, input_ids2dict_ids, dict_ids2input_ids)
        d_model = int(self.embed_weight.shape[1])
        if embed_dim is not None and int(embed_dim) != d_model:
            raise ValueError(f"embed_dim={embed_dim} does not match table width {d_model}")
        self.max_seq_len = int(max_seq_len)
        self.wpe_scale = float(wpe_scale)
        pos = torch.randn(self.max_seq_len, d_model, dtype=torch.float32) * 0.02
        self.register_parameter("pos_embed", nn.Parameter(pos, requires_grad=False))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb, mask = super().encode(input_ids, attention_mask)
        seq = emb.shape[1]
        if seq > self.pos_embed.shape[0]:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len={self.pos_embed.shape[0]}")
        pos = self.pos_embed[:seq].to(device=emb.device, dtype=emb.dtype)
        emb = emb + self.wpe_scale * pos.unsqueeze(0)
        return emb, mask


__all__ = ["RandomEmbeddingEncoder", "RandomEmbeddingEncoder_wPosEmb"]
