# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Custom layer initializations."""

import math

import torch
import torch.nn as nn


def _init_lecun(w, b=None): nn.init.normal_(w, mean=0, std=math.sqrt(1./w.shape[1]))
def _init_relu(w, b=None): nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
def _init_he(w, b=None): nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu')
def _init_normal(w, b=None): nn.init.normal_(w, mean=0, std=1/math.sqrt(w.shape[0]))
def _init_glorot(w, b=None): nn.init.xavier_uniform_(w)
def _init_zeros_w(w, b=None): nn.init.zeros_(w)
def _init_final(w, b=None): nn.init.zeros_(w)
def _init_gating(w, b=None): nn.init.ones_(w); (nn.init.zeros_(b) if b is not None else None)

_INIT_FUNCS = {
    "lecun": _init_lecun, "relu": _init_relu, "he": _init_he, "normal": _init_normal,
    "glorot": _init_glorot, "zeros": _init_zeros_w, "final": _init_final, "gating": _init_gating,
}


class _Linear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, init=None, **kw):
        self._init = init
        super().__init__(in_f, out_f, bias=bias, **kw)
    def reset_parameters(self):
        if self._init and self.bias is not None:
            nn.init.zeros_(self.bias)
        if self._init and isinstance(self._init, str):
            _INIT_FUNCS[self._init](self.weight, self.bias)
        elif callable(self._init):
            self._init(self.weight, self.bias)
        else:
            super().reset_parameters()


class _Embedding(nn.Embedding):
    def __init__(self, n_emb, emb_dim, padding_idx=None, init=None, **kw):
        self._init = init
        super().__init__(n_emb, emb_dim, padding_idx=padding_idx, **kw)
    @property
    def dtype(self): return self.weight.dtype
    def forward(self, input, mask=None):
        if mask is None or input.ndim == 1:
            return super().forward(input)
        out = torch.zeros(*input.shape, self.embedding_dim, dtype=self.weight.dtype, device=input.device)
        out[mask] = super().forward(input[mask])
        return out
    def reset_parameters(self):
        if self._init and isinstance(self._init, str):
            _INIT_FUNCS[self._init](self.weight)
        elif callable(self._init):
            self._init(self.weight)
        else:
            super().reset_parameters()
