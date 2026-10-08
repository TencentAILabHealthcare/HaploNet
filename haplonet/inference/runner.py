"""Pure PyTorch inference loop for HaploNet methylation classification.

Supports auto-detection of 2-class vs 3-class models from checkpoint,
and auto-detection of feature availability (haplotype, strand, quality).
"""
import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from ..model.usm import MethylationUSM
from ..model.head import XNATokenizer
from .dataset import prepare_single_record, load_msa_records
from ..model.utils import collate_dense_tensors, masked_mean


# Checkpoint key mapping: original names -> our package names
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
    "gt_cls.bias": "gt_cls.out.bias",
}


def _map_state_dict(raw_sd):
    """Map checkpoint keys to package model keys."""
    sd = {}
    for k, v in raw_sd.items():
        nk = k
        nk = nk.replace("module.", "").replace("backbone.", "")
        for old_pfx, new_pfx in [("norm_1", "n1"), ("norm_2", "n2"), ("norm_3", "n3")]:
            nk = nk.replace(old_pfx, new_pfx)
        if k in KEY_MAP:
            mapped = KEY_MAP[k]
            if mapped is None:
                continue
            nk = mapped
        sd[nk] = v
    return sd


def _detect_checkpoint_features(raw_sd):
    """Auto-detect model features and gt_class from checkpoint keys."""
    ckpt_keys = set(raw_sd.keys()) if isinstance(raw_sd, dict) else set()
    has_hap = any("haplotype" in k for k in ckpt_keys)
    has_strand = any("strand" in k for k in ckpt_keys)
    has_quality = any("quality" in k for k in ckpt_keys)
    detected_class = 2  # default 2-class
    for k in ckpt_keys:
        if "out_proj.bias" in k or "out.bias" in k:
            try:
                detected_class = raw_sd[k].shape[0]
            except Exception:
                pass
            break
    return {
        "has_haplotype": has_hap,
        "has_strand": has_strand,
        "has_quality": has_quality,
        "gt_class": detected_class,
    }


def run_pytorch_inference(msa_file, model_path, output_csv, tokenizer=None,
                          batch_size=64, use_cpu=True, gt_class=None):
    """Run pure PyTorch inference on MSA records.

    Args:
        msa_file: Path to MSA pickle file (.pkl.bin)
        model_path: Path to .pt model weights
        output_csv: Output CSV path for predictions
        tokenizer: XNATokenizer instance (created if None)
        batch_size: Inference batch size
        use_cpu: Use CPU instead of GPU
        gt_class: Force gt_class (auto-detected from checkpoint if None)

    Returns:
        Path to output CSV file
    """
    if tokenizer is None:
        tokenizer = XNATokenizer()

    # Load MSA records
    print(f"[INFO] Loading MSA data: {msa_file}")
    records = load_msa_records(msa_file)
    print(f"[INFO] Records: {len(records)}")

    # Load weights and detect features
    print(f"[INFO] Loading model weights: {model_path}")
    raw_sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(raw_sd, dict) and "module" in raw_sd:
        raw_sd = raw_sd["module"]
    if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"] if isinstance(raw_sd["state_dict"], dict) else {}

    features = _detect_checkpoint_features(raw_sd)
    if gt_class is not None:
        features["gt_class"] = gt_class

    print(f"[INFO] Checkpoint features: haplotype={features['has_haplotype']}, "
          f"strand={features['has_strand']}, quality={features['has_quality']}, "
          f"gt_class={features['gt_class']}")

    # Build model
    model = MethylationUSM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=384,
        num_layers=12,
        num_heads=12,
        coverage_dim=28,
        pretrained_coverage_dim=28,
        with_haplotype_embedding=features["has_haplotype"],
        with_ins=True,
        with_quality=features["has_quality"],
        with_strand=features["has_strand"],
        gt_class=features["gt_class"],
        tokenizer=tokenizer,
    )

    # Load mapped weights
    sd = _map_state_dict(raw_sd)
    model.load_model_weights(sd, strict=False)

    device = torch.device("cpu") if use_cpu else torch.device("cuda")
    model = model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model on {device}, params={param_count / 1e6:.1f}M, gt_class={features['gt_class']}")

    # Prepare dataset
    dataset = [prepare_single_record(r, tokenizer) for r in records]

    # Inference loop
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pad_id = tokenizer.pad
    results = []
    gt_class_val = features["gt_class"]

    for start_i in tqdm(range(0, len(dataset), batch_size), desc="Inference", disable=None):
        batch = dataset[start_i:start_i + batch_size]
        bx, by = zip(*batch)
        xs = list(zip(*bx))

        msa_ids = collate_dense_tensors(xs[0], pad_id)
        ins_ids = collate_dense_tensors(xs[1], pad_id)
        mq_t = collate_dense_tensors(xs[2], 0)
        bq_t = collate_dense_tensors(xs[3], 0)
        st_t = collate_dense_tensors(xs[4], 3)

        n_inputs = len(dataset[0][0])
        has_hap = (n_inputs == 8)
        if has_hap:
            hap_t = collate_dense_tensors(xs[5], 0)
            cov_t = collate_dense_tensors(xs[6], 0)
            tm = collate_dense_tensors(xs[7], 0).bool()
        else:
            cov_t = collate_dense_tensors(xs[5], 0)
            tm = collate_dense_tensors(xs[6], 0).bool()

        info, gt = zip(*by)
        gt_t = collate_dense_tensors(gt, 0)

        # Move to device
        msa_ids = msa_ids.to(device)
        ins_ids = ins_ids.to(device)
        mq_t = mq_t.to(device)
        bq_t = bq_t.to(device)
        st_t = st_t.to(device)
        cov_t = cov_t.to(device)
        tm = tm.to(device)
        kwargs = {
            "strand_ids": st_t,
            "mapping_qualities": mq_t,
            "base_qualities": bq_t,
            "coverage_counts": cov_t,
            "target_mask": tm,
        }
        if has_hap:
            kwargs["haplotypes"] = hap_t.to(device)
        else:
            kwargs["haplotypes"] = None

        with torch.no_grad():
            outputs = model(msa_ids, ins_ids=ins_ids, **kwargs)

        probs = outputs["gt_seq"].softmax(dim=-1)  # [B, gt_class]

        for i in range(len(info)):
            meta_item = info[i]
            row = [meta_item["chrom"], int(meta_item["pos"])]
            if "depth_total" in meta_item:
                row.extend([meta_item.get("depth1", 0),
                            meta_item.get("depth2", 0),
                            meta_item.get("depth_total", 0)])
            if "haplotype" in meta_item or "haplotype_idx" in meta_item:
                row.append(meta_item.get("haplotype", meta_item.get("haplotype_idx", "")))
            row.append(int(gt_t[i].item()))
            probs_i = probs[i].tolist()
            row.extend(probs_i)
            results.append(row)

    # Write CSV header
    header = ["chrom", "pos"]
    if results and "depth_total" in (info[0] if info else {}):
        header.extend(["depth1", "depth2", "depth_total"])
    if results and ("haplotype" in (info[0] if info else {}) or "haplotype_idx" in (info[0] if info else {})):
        header.append("haplotype")
    # Probability column names depend on gt_class
    if gt_class_val == 3:
        header.extend(["gt_seq", "p_unmethylated", "p_full_methylated", "p_haplotype"])
    else:
        header.extend(["gt_seq", "prob_0", "prob_1"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            writer.writerow(r)

    # Print quick stats
    if results:
        total = len(results)
        from collections import defaultdict
        per_class = defaultdict(lambda: [0, 0])
        gt_col = header.index("gt_seq")
        if gt_class_val == 3:
            p_un_col = header.index("p_unmethylated")
            p_full_col = header.index("p_full_methylated")
            p_hap_col = header.index("p_haplotype")
        else:
            p_un_col = header.index("prob_0")
            p_full_col = header.index("prob_1")
            p_hap_col = None

        for r in results:
            gt_cls = r[gt_col]
            if gt_class_val == 3:
                probs_r = [r[p_un_col], r[p_full_col], r[p_hap_col]]
            else:
                probs_r = [r[p_un_col], r[p_full_col]]
            pred_cls = int(np.argmax(probs_r))
            per_class[gt_cls][1] += 1
            if gt_cls == pred_cls:
                per_class[gt_cls][0] += 1

        print(f"[INFO] Inference done: {output_csv} ({total} rows)")
        cls_names = {0: "unmethylated", 1: "methylated"}
        if gt_class_val == 3:
            cls_names[2] = "ASM/haplotype"
        for c in sorted(per_class.keys()):
            cc, ct = per_class[c]
            name = cls_names.get(c, f"class{c}")
            pct = cc / ct * 100 if ct > 0 else 0
            print(f"[INFO]   {name} (gt={c}): acc = {cc}/{ct} = {pct:.1f}%")

    return output_csv
