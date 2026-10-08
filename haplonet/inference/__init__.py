# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Inference dataset preparation and PyTorch inference runner."""

import os
import pickle
import struct

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..model import XNATokenizer, MethylationUSM
from ..model.utils import (
    masked_mean, collate_dense_tensors, flatten_final_dims,
    normalize_iupac_seq, get_mean_quality_from_msa, count_base_from_msa,
    make_sequencing_insert_msa,
)


def prepare_single_record(record, tokenizer, max_seqs=32, max_depth=200):
    """Convert one MSA record -> model input tensors. Supports both 2-class and 3-class."""
    msa = record["msa"]; ins = record["ins"]; meta = record["meta"]
    strands = record.get("strands"); mq = record.get("mapping_qualities"); bq = record.get("base_qualities")
    haps = record.get("haplotype"); ref_seq = record["ref_seq"]; depth = record["depth"]

    def clip(arr, n):
        if len(arr) <= n: return arr
        indices = [0] + sorted(list(range(1, len(arr))))[:(n-1)]
        return [arr[i] for i in sorted(indices)[:n]]
    msa = clip(msa, max_depth); ins = clip(ins, max_depth)
    strands = clip(strands, max_depth) if strands else None
    mq = clip(mq, max_depth) if mq else None; bq = clip(bq, max_depth) if bq else None
    haps = clip(haps, max_depth) if haps else None
    mean_mq, mean_bq = get_mean_quality_from_msa(
        msa, mq or [[0]*len(msa[0]) for _ in msa],
        bq or [[0]*len(msa[0]) for _ in msa])
    msa = clip(msa, max_seqs); ins = clip(ins, max_seqs)
    strands = clip(strands, max_seqs) if strands else None
    mq = clip(mq, max_seqs) if mq else None; bq = clip(bq, max_seqs) if bq else None
    haps = clip(haps, max_seqs) if haps else None
    base_counts = count_base_from_msa(msa, ins, strand_msa=None)
    d = len(msa)
    if ins is not None and len(ins) < d: ins += [[]]*(d - len(ins))
    if strands is not None and len(strands) < d: strands += [strands[0]]*(d - len(strands))
    if mq is not None and len(mq) < d: mq += [mq[0]]*(d - len(mq))
    if bq is not None and len(bq) < d: bq += [bq[0]]*(d - len(bq))
    if haps is not None and len(haps) < d: haps += [haps[0]]*(d - len(haps))
    ref_norm = normalize_iupac_seq(ref_seq)
    msa_tokens = torch.stack([tokenizer(ref_norm)] + [tokenizer(normalize_iupac_seq(s)) for s in msa])
    ins_msa_seqs = make_sequencing_insert_msa(ins, len(ref_norm))
    ins_msa_tokenized = [ref_norm] + [normalize_iupac_seq(s) for s in ins_msa_seqs]
    ins_tokens = torch.stack([tokenizer(s, bos=True, eos=True) for s in ins_msa_tokenized])
    if ins_tokens.shape[0] != msa_tokens.shape[0]:
        if ins_tokens.shape[0] < msa_tokens.shape[0]:
            pad_tensor = torch.full((len(ref_norm)+2,), tokenizer.pad, dtype=torch.long)
            ins_tokens = torch.cat([ins_tokens, pad_tensor.unsqueeze(0).expand(msa_tokens.shape[0]-ins_tokens.shape[0], -1)])
        else:
            ins_tokens = ins_tokens[:msa_tokens.shape[0]]

    L = msa_tokens.shape[1]; D = msa_tokens.shape[0]; n_reads = D - 1; seq_len_raw = len(ref_norm)

    def _to_DL(t, default_val=0, dtype=torch.long, ref_fill=None):
        if t is None:
            out = torch.full((D, L), default_val, dtype=dtype)
            if ref_fill is not None:
                rf = torch.tensor(ref_fill, dtype=dtype) if not isinstance(ref_fill, torch.Tensor) else ref_fill.to(dtype)
                if rf.dim() == 0 or rf.shape[0] == 1:
                    out[0] = rf.expand(L) if rf.numel() == 1 else rf
                elif rf.dim() == 1:
                    out[0,:min(rf.shape[0],L)] = rf[:L] if rf.shape[0] >= L else torch.cat([rf, rf[-1:].expand(L-rf.shape[0])])
                else:
                    out[0,:L] = rf.flatten()[:L]
            return out
        t = torch.tensor(t, dtype=dtype)
        if t.dim() == 1: t_raw = t.unsqueeze(1)
        else: t_raw = t
        if ref_fill is not None:
            rf = torch.tensor(ref_fill, dtype=dtype) if not isinstance(ref_fill, torch.Tensor) else ref_fill.to(dtype)
            if rf.dim() == 0 or rf.numel() == 1:
                ref_row = rf.expand(L).clone()
            elif rf.dim() == 1:
                ref_row = rf[:L].clone() if rf.shape[0] >= L else torch.cat([rf, rf[-1:].expand(L-rf.shape[0])])
            else:
                ref_row = rf.flatten()[:L]
        else:
            ref_row = t_raw[-1].clone() if t_raw.shape[0] > 0 else torch.full(L, default_val, dtype=dtype)
        if t_raw.shape[0] >= n_reads: reads = t_raw[:n_reads]
        elif t_raw.shape[0] > 0: reads = torch.cat([t_raw, t_raw[-1:].expand(n_reads-t_raw.shape[0])])
        else: reads = torch.full((n_reads,L), default_val, dtype=dtype)
        if reads.shape[1] >= L: reads = reads[:,:L]
        else: reads = torch.cat([reads, torch.full((n_reads,L-reads.shape[1]), default_val, dtype=dtype)], dim=1)
        return torch.cat([ref_row.unsqueeze(0), reads], dim=0)

    _strand_ref = torch.zeros(len(ref_norm), dtype=torch.long)
    strand_t = _to_DL(strands, 0, torch.long, ref_fill=_strand_ref) if strands is not None else torch.full((D, L), 0, dtype=torch.long)
    _mq_ref = torch.tensor(mean_mq, dtype=torch.float) if mean_mq is not None else torch.zeros(len(ref_norm))
    _bq_ref = torch.tensor(mean_bq, dtype=torch.float) if mean_bq is not None else torch.zeros(len(ref_norm))
    mq_t = _to_DL(mq, 0.0, torch.float, ref_fill=_mq_ref)/100.0 if mean_mq is not None else torch.zeros(D, L)
    bq_t = _to_DL(bq, 0.0, torch.float, ref_fill=_bq_ref)/100.0 if mean_bq is not None else torch.zeros(D, L)
    _hap_ref = torch.zeros(len(ref_norm), dtype=torch.long)
    hap_t = _to_DL(haps, 0, torch.long, ref_fill=_hap_ref) if haps is not None else torch.zeros(D, L, dtype=torch.long)
    _cov = torch.tensor(base_counts, dtype=torch.float)
    if _cov.ndim >= 2:
        c_len = _cov.shape[0]
        if c_len >= L: cov_t = _cov[:L,:]
        elif c_len > 0: cov_t = torch.cat([_cov, _cov[-1:].expand(L-c_len, _cov.shape[1])])
        else: cov_t = torch.zeros(L, 28)
    else: cov_t = torch.zeros(L, 28)
    target_idx = int(meta["pos"]) - int(meta["start"])
    target_idx = max(0, min(msa_tokens.shape[1]-1, target_idx))
    tmask = torch.zeros(msa_tokens.shape[1], dtype=torch.bool); tmask[target_idx] = True
    gt_raw = meta.get("beta", meta.get("gt", 0.5))
    if isinstance(gt_raw, int): gt_cls = min(gt_raw, 2)
    elif gt_raw > 0.9: gt_cls = 1
    elif gt_raw < 0.1: gt_cls = 0
    else: gt_cls = 2  # intermediate -> ASM class for 3-class

    inputs = (msa_tokens, ins_tokens, mq_t, bq_t, strand_t)
    if record.get("haplotype") is not None: inputs += (hap_t,)
    inputs += (cov_t, tmask,)
    targets = (meta, torch.tensor(gt_cls, dtype=torch.long))
    return inputs, targets


def load_msa_records(msa_path):
    """Load MSA records, auto-detecting format."""
    idx_path = msa_path + '.idx'
    if os.path.isfile(idx_path):
        return _load_mmap_indexed(msa_path, idx_path)
    with open(msa_path, 'rb') as f:
        return pickle.load(f)


def _load_mmap_indexed(data_path, idx_path):
    if os.path.isfile(data_path):
        try:
            records = _load_pickle_streaming(data_path)
            if len(records) > 0 and 'msa' in records[0]:
                return records
        except Exception: pass
    if os.path.isfile(idx_path):
        return _load_via_index(data_path, idx_path)
    raise ValueError(f"Cannot load MSA: no valid format found for {data_path}")


def _load_pickle_streaming(data_path, max_records=None):
    records = []
    with open(data_path, 'rb') as f:
        count = 0
        while True:
            try:
                rec = pickle.load(f); count += 1
                if isinstance(rec, dict) and 'msa' in rec:
                    records.append(rec)
                    if max_records and len(records) >= max_records: break
            except EOFError: break
            except Exception: break
    return records


def _load_via_index(data_path, idx_path):
    dtypes_map = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
                  5: np.int64, 6: np.float32, 7: float, 8: np.uint16}
    with open(idx_path, 'rb') as f:
        raw_magic = f.read(8); magic = raw_magic.rstrip(b'\x00')
        if magic not in (b'MMIDIDX', b'TNTIDX'): raise ValueError(f'Bad index magic: {raw_magic}')
        version = struct.unpack('<Q', f.read(8))[0]
        dtype_code = struct.unpack('B', f.read(1))[0]
        num_items = struct.unpack('<Q', f.read(8))[0]
        doc_count = struct.unpack('<Q', f.read(8))[0]
        data_offset = f.tell()
    dtype = dtypes_map.get(dtype_code, np.uint8)
    idx_mmap = np.memmap(idx_path, mode='r', dtype=np.uint8, offset=data_offset)
    n_sizes = num_items * 4
    sizes = np.frombuffer(idx_mmap, dtype=np.int32, count=num_items, offset=0)
    pointers = np.frombuffer(idx_mmap, dtype=np.int64, count=num_items, offset=n_sizes)
    max_size = sizes.max() if len(sizes) > 0 else 0
    if max_size > 100 * 1024 * 1024:
        del idx_mmap; raise ValueError("Index parsing failed")
    records = []
    with open(data_path, 'rb') as f:
        for i in range(num_items):
            ptr, sz = int(pointers[i]), int(sizes[i])
            if sz <= 0 or sz > 100 * 1024 * 1024: continue
            f.seek(ptr); raw = f.read(sz)
            if not raw: break
            try: records.append(pickle.loads(raw))
            except: pass
    del idx_mmap; return records


KEY_MAP = {
    "embedding.weight": "embed.weight",
    "strand_embedding.weight": "strand_embed.weight",
    "quality_embedding.weight": "qual_embed.weight",
    "quality_embedding.bias": "qual_embed.bias",
    "coverage_embedding.weight": "cov_embed.weight",
    "coverage_embedding.bias": "cov_embed.bias",
    "haplotype_embedding.weight": "hap_embed.weight",
    "lm_head.weight": None,
    "gt_cls.out_proj.weight": "gt_cls.out.weight",
    "gt_cls.out_proj.bias": "gt_cls.out.bias",
}


def run_pytorch_inference(msa_file, model_path, output_csv, tokenizer, batch_size=64,
                          use_cpu=True, gt_class=3):
    """Pure PyTorch inference loop. Auto-detects features from checkpoint."""
    print(f"[INFO] Loading MSA data: {msa_file}")
    records = load_msa_records(msa_file)
    print(f"[INFO] Records: {len(records)}")
    print(f"[INFO] Loading model weights: {model_path}")
    raw_sd = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(raw_sd, dict) and 'module' in raw_sd: raw_sd = raw_sd['module']
    if isinstance(raw_sd, dict) and 'state_dict' in raw_sd:
        raw_sd = raw_sd['state_dict'] if isinstance(raw_sd['state_dict'], dict) else {}
    ckpt_keys = set(raw_sd.keys()) if isinstance(raw_sd, dict) else set()
    has_hap = any('haplotype' in k for k in ckpt_keys)
    has_strand = any('strand' in k for k in ckpt_keys)
    has_quality = any('quality' in k for k in ckpt_keys)
    detected_class = gt_class
    for k in ckpt_keys:
        if 'out_proj.bias' in k or 'out.bias' in k:
            try: detected_class = raw_sd[k].shape[0]
            except: pass
            break
    print(f"[INFO] Checkpoint: haplotype={has_hap}, strand={has_strand}, quality={has_quality}, gt_class={detected_class}")

    model = MethylationUSM(
        vocab_size=tokenizer.vocab_size, embedding_dim=384, num_layers=12, num_heads=12,
        coverage_dim=28, pretrained_coverage_dim=28,
        with_haplotype_embedding=has_hap, with_ins=True,
        with_quality=has_quality, with_strand=has_strand,
        gt_class=detected_class, tokenizer=tokenizer,
    )
    sd = {}
    for k, v in raw_sd.items():
        nk = k; nk = nk.replace("module.", "").replace("backbone.", "")
        for old_pfx, new_pfx in [("norm_1","n1"), ("norm_2","n2"), ("norm_3","n3")]:
            nk = nk.replace(old_pfx, new_pfx)
        if k in KEY_MAP:
            mapped = KEY_MAP[k]
            if mapped is None: continue
            nk = mapped
        sd[nk] = v
    model.load_model_weights(sd, strict=False)

    device = torch.device('cpu') if use_cpu else torch.device('cuda')
    model = model.to(device); model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model on {device}, params={param_count/1e6:.1f}M, gt_class={gt_class}")

    dataset = [prepare_single_record(r, tokenizer) for r in records]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    pad_id = tokenizer.pad; results = []

    for start_i in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
        batch = dataset[start_i:start_i+batch_size]; bx, by = zip(*batch); xs = list(zip(*bx))
        msa_ids = collate_dense_tensors(xs[0], pad_id); ins_ids = collate_dense_tensors(xs[1], pad_id)
        mq_t = collate_dense_tensors(xs[2], 0); bq_t = collate_dense_tensors(xs[3], 0)
        st_t = collate_dense_tensors(xs[4], 3)
        n_inputs = len(dataset[0][0]); has_hap = (n_inputs == 8)
        if has_hap:
            hap_t = collate_dense_tensors(xs[5], 0); cov_t = collate_dense_tensors(xs[6], 0); tm = collate_dense_tensors(xs[7], 0).bool()
        else:
            cov_t = collate_dense_tensors(xs[5], 0); tm = collate_dense_tensors(xs[6], 0).bool()
        info, gt = zip(*by); gt_t = collate_dense_tensors(gt, 0)
        msa_ids = msa_ids.to(device); ins_ids = ins_ids.to(device); mq_t = mq_t.to(device)
        bq_t = bq_t.to(device); st_t = st_t.to(device); cov_t = cov_t.to(device); tm = tm.to(device)
        kwargs = {"strand_ids": st_t, "mapping_qualities": mq_t, "base_qualities": bq_t,
                  "coverage_counts": cov_t, "target_mask": tm}
        if has_hap: kwargs["haplotypes"] = hap_t.to(device)
        else: kwargs["haplotypes"] = None
        with torch.no_grad():
            outputs = model(msa_ids, ins_ids=ins_ids, **kwargs)
        probs = outputs["gt_seq"].softmax(dim=-1)
        for i in range(len(info)):
            meta_item = info[i]; row = [meta_item["chrom"], int(meta_item["pos"])]
            if "depth_total" in meta_item:
                row.extend([meta_item.get("depth1",0), meta_item.get("depth2",0), meta_item.get("depth_total",0)])
            if "haplotype" in meta_item or "haplotype_idx" in meta_item:
                row.append(meta_item.get("haplotype", meta_item.get("haplotype_idx","")))
            row.append(int(gt_t[i].item()))
            probs_i = probs[i].tolist(); row.extend(probs_i)
            results.append(row)
    header = ['chrom','pos']
    if results and "depth_total" in (info[0] if info else {}):
        header.extend(['depth1','depth2','depth_total'])
    if results and ("haplotype" in (info[0] if info else {}) or "haplotype_idx" in (info[0] if info else {})):
        header.append('haplotype')
    header.extend(['gt_seq', 'p_unmethylated', 'p_full_methylated', 'p_haplotype'])
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(results)
    print(f"[INFO] Inference done: {output_csv} ({len(results)} rows)")
    return output_csv


import csv
