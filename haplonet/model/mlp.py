# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""SwiGLU MLP block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import _Linear


class SwiGLUMlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, ffn_multiplier=None, dropout=0., bias=False, multiple_of=256, **kw):
        super().__init__()
        hd = hidden_dim or 4*dim
        if ffn_multiplier is not None: hd = int(ffn_multiplier*hd)
        hd = int(2*hd/3); self.hd = multiple_of*((hd+multiple_of-1)//multiple_of)
        self.c_fc = _Linear(dim, 2*self.hd, bias=bias)
        self.c_proj = _Linear(self.hd, dim, bias=bias, init="final")
        self.dropout = dropout
    def forward(self, x, mask=None, inplace=False):
        if mask is None: return self._fwd(x)
        try:
            out = x if inplace else x.clone(); out[mask] = self._fwd(x[mask]); return out
        except (IndexError, RuntimeError):
            return self._fwd(x)
    def _fwd(self, x):
        x = self.c_fc(x); x = F.silu(x[..., :self.hd])*x[..., self.hd:]; x = self.c_proj(x)
        return F.dropout(x, p=self.dropout, training=self.training)
