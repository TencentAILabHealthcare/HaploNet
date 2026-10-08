# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Utility functions for tensor operations and sequence processing."""

import numpy as np
import torch


def masked_mean(mask, value, dim=None, keepdim=False, eps=1e-8):
    if mask is None: return value.mean(dim=dim, keepdim=keepdim)
    mask = mask.expand_as(value)
    x = torch.sum(mask*value, dim=dim, keepdim=keepdim) / (eps + torch.sum(mask, dim=dim, keepdim=keepdim))
    return x.to(value.dtype) if x.dtype != value.dtype else x


def collate_dense_tensors(samples, pad_v=0):
    if len(samples) == 0: return torch.Tensor()
    device = samples[0].device
    max_shape = [max(lst) for lst in zip(*[s.shape for s in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i, t in enumerate(samples):
        result[i][tuple(slice(0,k) for k in t.shape)] = t
    return result


def flatten_final_dims(x, ndims):
    return x.reshape(*x.shape[:-ndims], -1)


def normalize_iupac_seq(seq):
    IUPAC = {'R':'A','Y':'C','S':'G','W':'A','K':'G','M':'A',
             'B':'C','D':'A','H':'A','V':'A'}
    return "".join(IUPAC.get(b,b) for b in seq)


def get_mean_quality_from_msa(msa, mq_msa, bq_msa):
    slen = len(msa[0]); d = len(mq_msa); mean_bq = [0.]*slen; mean_mq = [0.]*slen
    for i in range(slen):
        bqs = []; mqs = []
        for dd in range(d):
            if msa[dd][i] in ".N": continue
            bqs.append(bq_msa[dd][i]); mqs.append(mq_msa[dd][i])
        if bqs: mean_bq[i] = np.mean(bqs); mean_mq[i] = np.mean(mqs)
    return mean_mq, mean_bq


def count_base_from_msa(msa, ins_msa=None, strand_msa=None):
    """Count bases per position. Returns normalized counts /100."""
    slen = len(msa[0]); depth = len(msa)
    if strand_msa is not None:
        base_types = ("A","C","G","T","N","D","I","a","c","g","t","n","d","i",
                    "A+","C+","G+","T+","N+","a+","c+","g+","t+","n+")
    else:
        from ..data.constants import BASE28_TYPES
        base_types = BASE28_TYPES
    counts = {t: np.zeros(slen, np.int32) for t in base_types}
    for d in range(depth):
        seq = msa[d][:slen]
        for j, c in enumerate(seq):
            if c == ".": continue
            if c not in "AGCT-": c = "N"
            if c == "-": c = "D"
            if strand_msa is not None and d < len(strand_msa):
                s = strand_msa[d][j] if j < len(strand_msa[d]) else 2
                if s == 1: c = c.lower()
            if c in counts: counts[c][j] += 1
    return np.stack([counts[t] for t in base_types], axis=-1) / 100.0


def make_sequencing_insert_msa(ins_msa, seq_len):
    """Convert insertion records -> per-position insertion strings."""
    mas = [['.']*seq_len for _ in range(len(ins_msa))]
    for i, ins_info in enumerate(ins_msa):
        seq = mas[i]
        for info in ins_info:
            if len(info) < 2: continue
            ins_idx, ins_seq = info[0], info[1]
            end_idx = min(ins_idx + len(ins_seq), seq_len)
            seq[ins_idx:end_idx] = list(ins_seq[:end_idx-ins_idx])
    return mas
