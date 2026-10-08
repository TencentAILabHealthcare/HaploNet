#!/usr/bin/env python3
# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""
HaploNet 3-Class Inference: Genome-wide methylation state classification.

Classifies each CpG site into one of three states:
  - Class 0: unmethylated (beta < 0.3)
  - Class 1: fully methylated (beta > 0.7)
  - Class 2: intermediate / ASM (0.3 <= beta <= 0.7)

Pipeline:
  Step 1:  Load GT candidates + Build complete MSA from BAM
  Step 2:  3-class model inference
  Step 3:  Ground truth evaluation (accuracy, F1, confusion matrix)

Usage:
    python infer_3class.py                              # Default: HG002 chr1:68M test region, CPU
    python infer_3class.py --region chr1:68000000-68100000 --cpu
    python infer_3class.py --bam /path/to/sample.bam --sample HG003 --cpu
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import namedtuple
from datetime import datetime

# Add project root to path for haplonet import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from haplonet.data.constants import Candidate, check_seq, parse_region
from haplonet.data.bam_reader import make_sequencing_msa_transbp, _padding_msa
from haplonet.data.builder import load_gt_candidates, build_complete_msa
from haplonet.model.head import XNATokenizer
from haplonet.inference.runner import run_pytorch_inference
from haplonet.evaluate.metrics import run_evaluation_inline

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weight")
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULTS = {
    "ref": os.path.join(WEIGHT_DIR, "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"),
    "model_weight": os.path.join(WEIGHT_DIR, "complete.pt"),
    "example_bam": os.path.join(WEIGHT_DIR, "phased.haplotag.bam"),
    "gt_csv": os.path.join(WEIGHT_DIR, "HG002_methy_labels.csv"),
    "gt_jsonl": os.path.join(WEIGHT_DIR, "HG002_gt_labels.jsonl"),
}
DEFAULT_REGION = "chr1:68000000-68100000"
COMPLETE_PADDING = 16
COMPLETE_MIN_DEPTH = 8


def log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{level}] {msg}", flush=True)


def check_dependencies(use_cpu=False):
    missing = []
    for name, mod in [("pysam", "pysam"), ("torch", "torch"), ("tqdm", "tqdm"), ("numpy", "numpy")]:
        try:
            __import__(mod)
            log(f"{name} OK")
        except ImportError:
            missing.append(name)
    try:
        from intervaltree import IntervalTree  # noqa: F401
    except ImportError:
        missing.append("intervaltree")
    if use_cpu:
        log("CPU mode: no GPU needed")
    if missing:
        log(f"Missing: {', '.join(missing)}. pip install {' '.join(missing)}", "ERROR")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="HaploNet 3-Class Methylation Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: HG002 chr1:68M test region, CPU, 3-class
  python %(prog)s

  # Custom region
  python %(prog)s --region chr1:68000000-68100000 --cpu

  # Custom BAM file
  python %(prog)s --bam /path/to/sample.bam --sample HG003 --cpu
        """,
    )
    inp = parser.add_argument_group("Input")
    inp.add_argument("--bam", default=None, help="Input phased BAM file (with MM/ML/HP tags)")
    inp.add_argument("--msa-file", default=None, help="Existing MSA pkl.bin file (skip MSA build)")
    inp.add_argument("--region", default=DEFAULT_REGION, help=f"Region (default: {DEFAULT_REGION})")
    inp.add_argument("--sample", default="HG002", help="Sample ID for GT lookup (default: HG002)")
    data = parser.add_argument_group("Data & Model")
    data.add_argument("--ref", default=None, help="Reference genome FASTA")
    data.add_argument("--model-weight", default=None, help="Path to .pt weights (default: complete.pt)")
    data.add_argument("--gt-csv", default=None, help="GT label CSV (default: HG002_methy_labels.csv)")
    data.add_argument("--gt-class", type=int, default=3, choices=[2, 3],
                      help="Model classes: 2=haplo, 3=complete (default: 3)")
    msa = parser.add_argument_group("MSA Parameters")
    msa.add_argument("--padding", type=int, default=COMPLETE_PADDING,
                     help=f"MSA half-window (default: {COMPLETE_PADDING})")
    msa.add_argument("--min-depth", type=int, default=COMPLETE_MIN_DEPTH,
                     help=f"Min depth (default: {COMPLETE_MIN_DEPTH})")
    inf = parser.add_argument_group("Inference")
    inf.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    inf.add_argument("--cpu", action="store_true", help="Use CPU")
    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", default=None, help="Output directory (default: ./store)")
    out.add_argument("--skip-eval", action="store_true", help="Skip GT evaluation step")
    out.add_argument("--skip-check", action="store_true", help="Skip dependency check")
    args = parser.parse_args()

    if not args.skip_check:
        log("Checking dependencies...")
        ok = check_dependencies(use_cpu=args.cpu)
        print()
        if not ok:
            raise SystemExit("Missing dependencies.")

    ref_path = args.ref or DEFAULTS["ref"]
    bam_path = args.bam or DEFAULTS["example_bam"]
    model_weight = args.model_weight or DEFAULTS["model_weight"]
    gt_csv = args.gt_csv or DEFAULTS["gt_csv"]

    contig, start, end = parse_region(args.region)

    # Normalize sample ID
    sample_id = args.sample.upper()
    if sample_id.startswith("HG") and len(sample_id) > 4:
        pass
    elif sample_id.isdigit():
        sample_id = f"HG{int(sample_id):03d}"

    output_base = args.output_dir or os.path.join(SCRIPT_DIR, "store")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"3class_{contig}_{start}_{end}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    log("=" * 60)
    log("HaploNet 3-Class Methylation Classification")
    log("=" * 60)
    log(f"Sample:      {sample_id}")
    log(f"Region:      {contig}:{start}-{end}")
    log(f"BAM:         {bam_path}")
    log(f"Ref:         {ref_path}")
    log(f"GT CSV:      {gt_csv}")
    log(f"Weights:     {model_weight}")
    log(f"gt_class:    {args.gt_class}")
    log(f"MSA params:  padding={args.padding}, min_depth={args.min_depth}")
    log(f"Output:      {output_dir}")
    log(f"Mode:        {'CPU' if args.cpu else 'GPU'}")
    log("=" * 60)
    print()

    t0 = time.time()

    # Step 1: Load GT Candidates + Build Complete MSA
    if args.msa_file and os.path.isfile(args.msa_file):
        log("-" * 60); log("[Step 1/3] Using existing MSA file"); log("-" * 60)
        msa_file = args.msa_file
        log(f"Skipping BAM->MSA build (using: {msa_file})")
    else:
        log("-" * 60); log("[Step 1/3] Load GT candidates + Build complete MSA"); log("-" * 60)
        msa_work = os.path.join(output_dir, "msa")
        os.makedirs(msa_work, exist_ok=True)

        candidates = load_gt_candidates(gt_csv, contig, start, end)
        if len(candidates) == 0:
            log("No GT candidates found. Falling back to CpG extraction...", "WARN")
            import pysam
            ref_fasta = pysam.FastaFile(ref_path)
            ref_seq = ref_fasta.fetch(contig, start - 1, end).upper()
            ref_fasta.close()
            candidates = []
            for i in range(len(ref_seq) - 1):
                if ref_seq[i] == "C" and ref_seq[i + 1] == "G":
                    candidates.append(Candidate(contig, start + i - 1, 1))
            log(f"Fallback: found {len(candidates)} CpG sites")

        msa_file = build_complete_msa(contig, bam_path, ref_path, candidates, msa_work,
                                      padding=args.padding, min_depth=args.min_depth)
    print()

    # Step 2: Model Inference
    log("-" * 60); log("[Step 2/3] 3-class model inference"); log("-" * 60)
    infer_out = os.path.join(output_dir, "inference")
    os.makedirs(infer_out, exist_ok=True)
    tok = XNATokenizer()
    pred_csv = run_pytorch_inference(
        msa_file, model_weight,
        os.path.join(infer_out, "predictions.csv"),
        tok, batch_size=args.batch_size, use_cpu=args.cpu, gt_class=args.gt_class,
    )
    if not pred_csv:
        raise SystemExit("Inference failed.")
    print()

    # Step 3: Ground Truth Evaluation
    has_eval = False
    if not args.skip_eval:
        log("-" * 60); log("[Step 3/3] Ground truth evaluation"); log("-" * 60)
        eval_dir = os.path.join(output_dir, "evaluation")
        has_eval = run_evaluation_inline(pred_csv, sample_id, contig, eval_dir,
                                         weight_dir=WEIGHT_DIR, workspace_root=WORKSPACE_ROOT)
        print()

    # Summary
    total = time.time() - t0
    log(""); log("=" * 60); log("Done!"); log("=" * 60)
    log(f"  Sample:        {sample_id}")
    log(f"  Region:        {contig}:{start}-{end}")
    log(f"  MSA file:      {msa_file}")
    log(f"  Predictions:   {pred_csv}")
    log(f"  Evaluated:     {'Yes' if has_eval else 'Skipped'}")
    log(f"  Time:          {total:.1f}s ({total / 60:.1f}min)")
    log("=" * 60)

    summary = {
        "mode": "3class_classification",
        "sample": sample_id, "contig": contig,
        "bam": bam_path, "region": f"{contig}:{start}-{end}",
        "msa_file": msa_file, "predictions": pred_csv,
        "has_evaluation": has_eval,
        "msa_params": {"padding": args.padding, "min_depth": args.min_depth},
        "gt_class": args.gt_class,
        "total_seconds": round(total, 2),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log(f"  Summary:       {os.path.join(output_dir, 'summary.json')}")
    return summary


if __name__ == "__main__":
    main()
