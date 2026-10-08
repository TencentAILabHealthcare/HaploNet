# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Attention mechanisms: RoPE, SDPA, MHA, row/column attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import _Linear


def precompute_freqs_cis(seq_len, rotary_dim=32, dtype=torch.float32, device="cpu"):
    freqs = 1.0 / (10000**(torch.arange(0, rotary_dim, 2, dtype=dtype, device=device)/rotary_dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis, seq_dim=-3):
    x_c = torch.view_as_complex(x.float() * torch.tensor([1,1], dtype=x.dtype, device=x.device))
    slen = x.shape[seq_dim]
    idx = freqs_cis.shape[0] - slen
    f = freqs_cis[idx:idx+slen] if idx > 0 else freqs_cis[:slen]
    x_out = torch.view_as_real(x_c * f).flatten(-2)
    return x_out.type_as(x)


class SdpaAttention(nn.Module):
    def __init__(self, window_size=None, is_causal=False, scale=None):
        super().__init__(); self.window_size = window_size; self.is_causal = is_causal; self.scale = scale
    def forward(self, q, k, v, attn_mask=None, dropout_p=0., return_attn_probs=False,
                 return_attn_weight=False, scale=None, is_causal=False):
        s = self.scale or scale; causal = self.is_causal or is_causal
        if attn_mask is not None:
            attn_mask = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(-1,-1,q.shape[2],k.shape[2])
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal)
        return out, None


class CoreAttention(nn.Module):
    def __init__(self, window_size=None, is_causal=False, scale=None):
        super().__init__(); self.window_size = window_size; self.is_causal = is_causal; self.scale = scale
    def forward(self, q, k, v, attn_mask=None, dropout_p=0., return_attn_probs=False,
                 return_attn_weight=False, scale=None, is_causal=False):
        s = self.scale or scale; causal = self.is_causal or is_causal
        head_dim = q.shape[-1]; scaling = head_dim**-0.5 if s is None else s**-0.5
        attn = (q @ k.transpose(-2,-1)) * scaling
        if causal:
            L = q.shape[-2]; causal_mask = torch.triu(torch.ones(L,L,dtype=bool,device=q.device),diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        if attn_mask is not None: attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        if dropout_p > 0: attn = F.dropout(attn, p=dropout_p)
        aw = attn if return_attn_probs else None
        return (attn @ v), aw


class MHA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout=0., attention_mode="sdpa",
                 sliding_window_size=None, gating=False, is_causal=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim; self.num_heads = num_heads; self.head_dim = dim // num_heads
        self.c_attn = _Linear(dim, 3*dim, bias=bias); self.c_proj = _Linear(dim, dim, bias=bias, init="final")
        self.gating = gating
        if gating: self.g_proj = _Linear(dim, dim, bias=bias, init="gating")
        mode = attention_mode.lower()
        self.attn_fn = SdpaAttention(window_size=sliding_window_size) if mode in ("sdpa","native") else CoreAttention(window_size=sliding_window_size)
        self.freqs_cis = None; self.is_causal = is_causal

    def update_freqs(self, seq_len, dtype, device):
        if self.freqs_cis is None or seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(seq_len, self.head_dim//2, dtype=dtype, device=device)

    def forward(self, x, freqs_cis=None, attn_mask=None, is_causal=False, return_attn_weight=False):
        B, S, _ = x.shape
        q, k, v = self.c_attn(x).split(self.dim, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        if freqs_cis is not None:
            rd = self.head_dim // 2
            q_rotary = q[..., :rd].permute(0,2,1,3).reshape(B*self.num_heads, S, rd//2, 2)
            k_rotary = k[..., :rd].permute(0,2,1,3).reshape(B*self.num_heads, S, rd//2, 2)
            q_r = apply_rotary_emb(q_rotary, freqs_cis, seq_dim=-3); k_r = apply_rotary_emb(k_rotary, freqs_cis, seq_dim=-3)
            q_r = q_r.reshape(B, self.num_heads, S, rd).permute(0,2,1,3)
            k_r = k_r.reshape(B, self.num_heads, S, rd).permute(0,2,1,3)
            q = torch.cat([q_r, q[..., rd:]], dim=-1)
            k = torch.cat([k_r, k[..., rd:]], dim=-1)
        y, aw = self.attn_fn(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                          attn_mask=attn_mask, dropout_p=0., return_attn_weight=return_attn_weight, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B, S, self.dim)
        if self.gating: y = y * torch.sigmoid(self.g_proj(x))
        return y, aw


class MSARowAttn(nn.Module):
    def __init__(self, dim, num_heads, sliding_window_size=None, bias=False, dropout=0.,
                 attention_mode="sdpa", is_causal=False):
        super().__init__()
        self.mha = MHA(dim, num_heads, bias=bias, dropout=dropout, attention_mode=attention_mode,
                      sliding_window_size=sliding_window_size, is_causal=is_causal)
        self.num_heads = num_heads; self.head_dim = self.mha.head_dim
        self.is_causal = is_causal; self.freqs_cis = None
    def forward(self, m, mask=None, return_attn_weight=False, **kw):
        if mask is None:
            mx = m.reshape(-1, *m.shape[-2:]); row_mask = None; amask = None
        else:
            row_mask = mask.sum(dim=-1) > 0; mx = m[row_mask]; amask = mask[row_mask]
        self.mha.update_freqs(mx.shape[-2], mx.dtype, mx.device)
        y, aw = self.mha(mx, freqs_cis=self.mha.freqs_cis, attn_mask=amask,
                       is_causal=self.is_causal, return_attn_weight=return_attn_weight)
        bs, row = m.shape[:-2]
        if mask is not None:
            out = m.clone(); out[row_mask] = y
            if return_attn_weight:
                raw_aw = aw.new_zeros(bs, row, *aw.shape[-3:]); raw_aw[row_mask] = aw; aw = raw_aw
        else:
            out = y.reshape(*m.shape)
            if return_attn_weight: aw = aw.reshape(bs, row, *m.shape[:-2:])
        return out, aw


class MSAColAttn(MSARowAttn):
    def forward(self, m, mask=None, return_attn_weight=False, **kw):
        m = m.transpose(-2,-3)
        if mask is not None: mask = mask.transpose(-1,-2)
        m, aw = super().forward(m, mask=mask, return_attn_weight=return_attn_weight, **kw)
        m = m.transpose(-2,-3); return m, aw
