# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Normalization and dropout layers."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps']
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps; self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter('weight', None)
    def forward(self, input, mask=None, inplace=False):
        if mask is None: return self._impl(input)
        try:
            out = input if inplace else input.clone(); out[mask] = self._impl(input[mask]); return out
        except (IndexError, RuntimeError):
            return self._impl(input)
    def _impl(self, x):
        dims = tuple(range(-1, -len(self.normalized_shape)-1, -1))
        var = x.float().pow(2).mean(dims, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight if self.weight is not None else x


class DropPath(nn.Module):
    def __init__(self, p=0., scale=True):
        super().__init__(); self.p = p; self.scale = scale
    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep = 1 - self.p; shape = [1]*x.ndim
        r = x.new_empty(shape).bernoulli_(keep)
        if keep > 0 and self.scale: r.div_(keep)
        return x * r
