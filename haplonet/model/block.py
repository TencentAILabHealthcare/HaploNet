# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""USM Transformer block with column + row attention."""

import torch
import torch.nn as nn

from .norm import RMSNorm, DropPath
from .attention import MSARowAttn, MSAColAttn
from .mlp import SwiGLUMlp
from .utils import masked_mean


class UBlock(nn.Module):
    def __init__(self, dim, num_heads, sliding_window_size=None, dropout=0., droppath=0.,
                 dual_path=False, with_col_attn=True, is_causal=False, eps=1e-6):
        super().__init__()
        self.with_col = with_col_attn; self.dual_path = dual_path
        if with_col_attn:
            self.n1 = RMSNorm(dim, eps=eps)
            self.col_attn = MSAColAttn(dim, num_heads, sliding_window_size,
                                        bias=False, attention_mode="sdpa", is_causal=is_causal)
        self.n2 = RMSNorm(dim, eps=eps)
        self.row_attn = MSARowAttn(dim, num_heads, sliding_window_size,
                                      bias=False, attention_mode="sdpa", is_causal=is_causal)
        self.mlp = SwiGLUMlp(dim, dropout=dropout, bias=False)
        self.n3 = RMSNorm(dim, eps=eps)
        self.drop = DropPath(droppath)
    def forward(self, m, mask=None, return_attn_weight=False, **kw):
        if self.with_col:
            h, _ = self.col_attn(self.n1(m, mask=mask), mask=mask); m = m + self.drop(h)
        sm = masked_mean(mask[...,None] if mask is not None else None, m.float(), dim=1, keepdim=True).type_as(m)
        col_mask = (mask.sum(dim=-1, keepdim=True) > 0) if mask is not None else None
        h, _ = self.row_attn(self.n2(sm, mask=col_mask)); m = m + h
        skip = sm if self.dual_path else m; skip_mask = col_mask if self.dual_path else mask
        m = m + self.mlp(self.n3(skip), mask=skip_mask)
        return m, None
