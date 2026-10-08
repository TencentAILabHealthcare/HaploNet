# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""USM backbone and MethylationUSM classification model."""

import torch
import torch.nn as nn

from .layers import _Linear, _Embedding
from .norm import RMSNorm
from .block import UBlock
from .head import ClsHead, XNATokenizer
from .utils import masked_mean


class USMBase(nn.Module):
    CONFIG = {"msa-30m": dict(num_layers=12, num_heads=12, embedding_dim=384, bias=False)}
    def __init__(self, vocab_size, embedding_dim=384, num_layers=12, num_heads=12,
                 bias=False, dropout=0., droppath=0., tokenizer=None, include_head=False):
        super().__init__()
        self.vocab_size = vocab_size; self.num_layers = num_layers; self.num_heads = num_heads
        self.embedding_dim = embedding_dim; self.head_dim = embedding_dim // num_heads
        self.dropout = dropout
        if tokenizer: self.padding_idx = tokenizer.pad
        else: self.padding_idx = None
        self.embed = _Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            UBlock(embedding_dim, num_heads, dropout=dropout, droppath=droppath,
                   with_col_attn=True, is_causal=False) for _ in range(num_layers)])
        self.norm_final = RMSNorm(embedding_dim); self.include_head = include_head
        self.lm_head = _Linear(embedding_dim, vocab_size, bias=bias) if include_head else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (_Linear, nn.Linear)):
            nn.init.normal_(m.weight, 0, 0.02/math.sqrt(2*self.num_layers))
        elif isinstance(m, (_Embedding, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02/math.sqrt(2*self.num_layers))

    def _embed(self, token_ids, mask=None, **kw):
        return self.embed(token_ids, mask=mask)

    def _transformer(self, x, mask=None):
        for block in self.layers:
            x, _ = block(x, mask=mask)
        return x, [], {}

    def load_model_weights(self, state_dict, strict=False):
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing: print(f"  [WARN] Missing keys: {missing}")
        if unexpected: print(f"  [WARN] Unexpected keys: {unexpected}")


class MethylationUSM(USMBase):
    """USM backbone for methylation classification. Supports 2-class and 3-class."""
    def __init__(self, vocab_size=16, embedding_dim=384, num_layers=12, num_heads=12,
                 coverage_dim=28, pretrained_coverage_dim=28,
                 with_haplotype_embedding=False, with_ins=True, with_quality=True, with_strand=True,
                 gt_class=2, tokenizer=None, **kw):
        super().__init__(vocab_size, embedding_dim, num_layers, num_heads,
                         tokenizer=tokenizer, include_head=False, **kw)
        self.coverage_dim = coverage_dim; self.pretrained_cov_dim = pretrained_coverage_dim
        self.with_hap = with_haplotype_embedding; self.with_ins = with_ins
        self.with_qual = with_quality; self.with_strand = with_strand; self.gt_class = gt_class
        self.strand_embed = _Embedding(4, embedding_dim, init="zeros")
        nn.init.zeros_(self.strand_embed.weight)
        self.qual_embed = _Linear(2, embedding_dim, init="zeros")
        if coverage_dim != pretrained_coverage_dim:
            self.cov_adapt = nn.Sequential(
                nn.Linear(coverage_dim, pretrained_coverage_dim), nn.ReLU(),
                nn.Linear(pretrained_coverage_dim, pretrained_coverage_dim))
        else:
            self.cov_adapt = None
        self.cov_embed = _Linear(pretrained_coverage_dim, embedding_dim, init="zeros")
        if with_haplotype_embedding:
            self.hap_embed = _Embedding(4, embedding_dim)
        self.gt_cls = ClsHead(embedding_dim, gt_class, dropout=self.dropout)

    @property
    def dtype(self):
        try: return next(self.parameters()).dtype
        except StopIteration: return torch.bfloat16

    def _embed(self, msa_ids, ins_ids=None, strand_ids=None, mq=None, bq=None,
               haps=None, cov_counts=None, mask=None):
        emb = self.embed(msa_ids, mask=mask)
        if ins_ids is not None:
            ins_mask = None
            if self.padding_idx is not None:
                ins_mask = ins_ids.eq(self.padding_idx)
                if not ins_mask.any(): ins_mask = None
            emb = (emb + self.embed(ins_ids, mask=ins_mask)) * 0.5
        if strand_ids is not None:
            emb = emb + self.strand_embed(strand_ids, mask=mask)
        if mq is not None and bq is not None:
            qual = torch.stack([mq, bq], dim=-1).to(self.dtype); emb = emb + self.qual_embed(qual)
        if cov_counts is not None:
            ci = cov_counts.to(self.dtype)
            if self.cov_adapt is not None: ci = self.cov_adapt(ci)
            ce = self.cov_embed(ci)
            if mask is not None:
                ce = ce * mask[:, 0].unsqueeze(-1)
            emb[:, 0] = emb[:, 0] + ce
        if haps is not None and self.with_hap:
            emb = emb + self.hap_embed(haps)
        return emb

    def forward(self, msa_ids, ins_ids=None, mapping_qualities=None, base_qualities=None,
                strand_ids=None, haplotypes=None, coverage_counts=None, target_mask=None):
        pad_mask = None
        if self.padding_idx is not None:
            pad_mask = msa_ids.eq(self.padding_idx)
            if not pad_mask.any(): pad_mask = None
        msa_mask = None if pad_mask is not None else ~pad_mask
        emb = self._embed(msa_ids, ins_ids, strand_ids, mapping_qualities, base_qualities,
                         haplotypes, coverage_counts, mask=msa_mask)
        hidden, _, _ = self._transformer(emb, mask=msa_mask)
        hidden = masked_mean(msa_mask.unsqueeze(-1) if msa_mask is not None else None,
                           hidden.float(), dim=1).type_as(hidden)
        smask = (msa_mask[:,0] if msa_mask is not None else None) if target_mask is None else target_mask
        if smask is not None and smask.dim() == 2:
            smask_4d = smask.unsqueeze(1).unsqueeze(-1)
        elif smask is not None:
            smask_4d = smask
        else:
            smask_4d = None
        pooled = masked_mean(smask_4d, hidden.float(), dim=2).type_as(hidden)
        pooled = pooled.reshape(pooled.shape[0], -1)
        pooled = self.norm_final(pooled); logits = self.gt_cls(pooled)
        return {"gt_seq": logits}
