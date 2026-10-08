# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Classification head and XNA tokenizer."""

import torch
import torch.nn as nn


class ClsHead(nn.Module):
    def __init__(self, dim, num_classes=2, dropout=0., activation="tanh"):
        super().__init__()
        self.dense = nn.Linear(dim, dim); self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.dropout(x); x = self.act(self.dense(x)); x = self.dropout(x); return self.out(x)


class XNATokenizer:
    """Alphabet tokenizer for extended nucleotide alphabet (28 tokens)."""
    def __init__(self):
        std = ['A','C','G','T','U','N','.','-']
        prep = ['<cls>','<pad>','<eos>','<unk>']; app = ['<mask>',]
        self.all_toks = list(prep) + std
        for i in range((8-(len(self.all_toks)%8))%8): self.all_toks.append(f'<null_{i+1}>')
        self.all_toks.extend(app)
        self.tok2id = {t: i for i, t in enumerate(self.all_toks)}
        self.pad = self.tok2id['<pad>']; self.unk = self.tok2id['<unk>']
        self.cls_id = self.tok2id['<cls>']; self.vocab_size = len(self.all_toks)

    def encode(self, text, bos=True, eos=True, to_tensor=True):
        tokens = [self.tok2id.get(c, self.unk) for c in text.upper()]
        if bos: tokens = [self.cls_id] + tokens
        if eos: tokens += [self.tok2id['<eos>']]
        return torch.tensor(tokens, dtype=torch.long) if to_tensor else tokens

    def __call__(self, text, bos=True, eos=True):
        return self.encode(text, bos=bos, eos=eos, to_tensor=True)
