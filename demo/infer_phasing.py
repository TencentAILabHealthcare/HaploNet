#!/usr/bin/env python3
# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""
HaploNet Phasing Inference: Per-haplotype methylation classification + ASM detection.

Pipeline:
  Step 1:  CpG extraction + MSA build from BAM (per-haplotype)
  Step 2:  2-class model inference (methylated / unmethylated)
  Step 3:  ASM discordant filter
  Step 4:  CSV -> VCF conversion
  Step 5:  Graph consistency filter
  Step 6:  Depth ratio filter
  Step 7:  LongPhase phasing (optional, requires SNP VCF)

Usage:
    python infer_phasing.py --region chr1:68050000-68050500 --cpu
    python infer_phasing.py --bam /path/to/sample.bam --region chr1:1-5000000 --cpu
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
from collections import namedtuple
from datetime import datetime

# Add project root to path for haplonet import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from haplonet.data.constants import Candidate, check_seq, parse_region
from haplonet.data.builder import extract_cpg_sites, build_msa
from haplonet.model.head import XNATokenizer
from haplonet.inference.runner import run_pytorch_inference
from haplonet.pipeline.asm_filter import asm_filter
from haplonet.pipeline.vcf_utils import csv_to_vcf
from haplonet.pipeline.graph_filter import graph_filter_vcf
from haplonet.pipeline.ratio_filter import ratio_filter_vcf
from haplonet.pipeline.phasing import run_longphase

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weight")
DEFAULTS = {
    "ref": os.path.join(WEIGHT_DIR, "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"),
    "model_weight": os.path.join(WEIGHT_DIR, "haplotype.pt"),
    "example_bam": os.path.join(WEIGHT_DIR, "phased.haplotag.bam"),
    "snp_vcf": os.path.join(WEIGHT_DIR, "snp.vcf.gz"),
}
DEFAULT_REGION = "chr1:68050000-68050500"


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
    if use_cpu:
        log("CPU mode: no GPU needed")
    if missing:
        log(f"Missing: {', '.join(missing)}. pip install {' '.join(missing)}", "ERROR")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="HaploNet Phasing: Per-haplotype methylation + ASM detection + VCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --region chr1:68050000-68050500 --cpu
  python %(prog)s --bam sample.bam --region chr1:1-5000000 --snp-vcf snp.vcf.gz
        """,
    )
    inp = parser.add_argument_group("Input")
    inp.add_argument("--bam", default=None, help="Input phased BAM file (with MM/ML/HP tags)")
    inp.add_argument("--msa-file", default=None, help="Existing MSA pkl.bin file (skip MSA build)")
    inp.add_argument("--region", default=DEFAULT_REGION, help=f"Genomic region (default: {DEFAULT_REGION})")
    data = parser.add_argument_group("Data & Model")
    data.add_argument("--ref", default=None, help="Reference genome FASTA")
    data.add_argument("--model-weight", default=None, help="Path to .pt weights (default: haplotype.pt)")
    data.add_argument("--snp-vcf", default=None, help="SNP VCF for LongPhase phasing")
    data.add_argument("--sample-name", default="SAMPLE", help="Sample name for VCF output")
    msa = parser.add_argument_group("MSA Parameters")
    msa.add_argument("--padding", type=int, default=5, help="MSA half-window (default: 5)")
    msa.add_argument("--min-depth", type=int, default=4, help="Min depth per haplotype (default: 4)")
    inf = parser.add_argument_group("Inference")
    inf.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    inf.add_argument("--cpu", action="store_true", help="Use CPU")
    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", default=None, help="Output directory (default: ./store)")
    out.add_argument("--keep-temp", action="store_true", help="Keep temp files")
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
    snp_vc = args.snp_vcf or DEFAULTS["snp_vcf"]

    contig, start, end = parse_region(args.region)
    output_base = args.output_dir or os.path.join(SCRIPT_DIR, "store")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"phasing_{contig}_{start}_{end}_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    temp_workdir = tempfile.mkdtemp(prefix="usm_", dir=output_dir)

    log("=" * 60)
    log("HaploNet Phasing Inference")
    log("=" * 60)
    log(f"Sample:   {args.sample_name}")
    log(f"Region:   {contig}:{start}-{end}")
    log(f"BAM:      {bam_path}")
    log(f"Ref:      {ref_path}")
    log(f"Weights:  {model_weight}")
    log(f"Output:   {output_dir}")
    log(f"Mode:     {'CPU' if args.cpu else 'GPU'}")
    log("=" * 60)
    print()

    t0 = time.time()

    # Step 1: MSA Build (or use existing)
    if args.msa_file and os.path.isfile(args.msa_file):
        log("-" * 60); log("[Step 1/7] Using existing MSA file"); log("-" * 60)
        msa_file = args.msa_file
        log(f"Skipping BAM->MSA build (using: {msa_file})")
    else:
        log("-" * 60); log("[Step 1/7] CpG extraction + MSA build"); log("-" * 60)
        msa_work = os.path.join(temp_workdir, "msa")
        cpg_jsonl = os.path.join(msa_work, f"{contig}_cpg.jsonl")
        nc = extract_cpg_sites(bam_path, ref_path, contig, start, end, cpg_jsonl)
        if nc == 0:
            shutil.rmtree(temp_workdir, ignore_errors=True)
            raise SystemExit("No CpG sites.")
        msa_file = build_msa(contig, bam_path, ref_path, cpg_jsonl, msa_work,
                             padding=args.padding, min_depth=args.min_depth)
    print()

    # Step 2: Model Inference
    log("-" * 60); log("[Step 2/7] Model inference"); log("-" * 60)
    infer_out = os.path.join(temp_workdir, "inference")
    os.makedirs(infer_out, exist_ok=True)
    tok = XNATokenizer()
    pred_csv = run_pytorch_inference(
        msa_file, model_weight,
        os.path.join(infer_out, "methy_class.csv"),
        tok, batch_size=args.batch_size, use_cpu=args.cpu,
    )
    if not pred_csv:
        shutil.rmtree(temp_workdir, ignore_errors=True)
        raise SystemExit("Inference failed.")
    print()

    # Step 3: ASM Filter
    log("-" * 60); log("[Step 3/7] ASM discordant filter"); log("-" * 60)
    asm_work = os.path.join(temp_workdir, "asm")
    os.makedirs(asm_work, exist_ok=True)
    asm_csv = asm_filter(pred_csv, os.path.join(asm_work, "methy_class.asm_filtered.csv"),
                         high_prob=0.7, low_prob=0.3, min_hap_depth=0)
    if not asm_csv:
        log("ASM filter failed, using raw predictions", "WARN")
        asm_csv = pred_csv
    print()

    # Step 4: CSV -> VCF
    log("-" * 60); log("[Step 4/7] VCF generation"); log("-" * 60)
    vcfd = os.path.join(output_dir, "vcf")
    os.makedirs(vcfd, exist_ok=True)
    raw_vcf = csv_to_vcf(asm_csv, bam_path, ref_path,
                          os.path.join(vcfd, f"{args.sample_name}_mod.vcf"),
                          sample_name=args.sample_name)
    if not raw_vcf:
        log("VCF conversion failed", "WARN")
        perm_infer = os.path.join(output_dir, "inference")
        if os.path.exists(infer_out) and os.listdir(infer_out):
            shutil.copytree(infer_out, perm_infer, dirs_exist_ok=True)
        if not args.keep_temp:
            shutil.rmtree(temp_workdir, ignore_errors=True)
        raise SystemExit("VCF conversion failed.")
    print()

    # Step 5: Graph Consistency Filter
    log("-" * 60); log("[Step 5/7] Graph consistency filter"); log("-" * 60)
    graph_out = os.path.join(vcfd, "graph_filtered")
    os.makedirs(graph_out, exist_ok=True)
    gf_vcf = graph_filter_vcf(raw_vcf, os.path.join(graph_out, "filtered.vcf"),
                               connect_adjacent=5, connect_confidence=0.80, min_support_floor=4)
    if gf_vcf:
        log(f"Graph filtered: {gf_vcf}")
    else:
        log("Graph filter failed, using raw VCF", "WARN")
        gf_vcf = raw_vcf
    print()

    # Step 6: Ratio Filter
    log("-" * 60); log("[Step 6/7] Depth ratio filter"); log("-" * 60)
    rf_vcf = ratio_filter_vcf(gf_vcf, os.path.join(graph_out, "ratio_filtered.vcf.gz"),
                              ratio_low=0.25)
    final_mod_vcf = rf_vcf if rf_vcf else gf_vcf
    if final_mod_vcf:
        log(f"Ratio filtered: {final_mod_vcf}")
    else:
        log("Ratio filter failed", "WARN")
        final_mod_vcf = gf_vcf
    print()

    if not args.keep_temp:
        log("Cleaning up temp...")
        shutil.rmtree(temp_workdir, ignore_errors=True)
    else:
        log(f"Temp kept: {temp_workdir}")

    # Step 7: LongPhase Phasing
    phased = None
    bstats = None
    if args.snp_vcf and snp_vc and os.path.isfile(snp_vc):
        log(""); log("-" * 60); log("[Step 7/7] LongPhase phasing"); log("-" * 60)
        phased, bstats = run_longphase(final_mod_vcf, bam_path, snp_vc, ref_path, vcfd,
                                        args.sample_name)

    # Summary
    total = time.time() - t0
    log(""); log("=" * 60); log("Done!"); log("=" * 60)
    log(f"  VCF (raw):     {raw_vcf}")
    if gf_vcf and gf_vcf != raw_vcf:
        log(f"  VCF (graph):   {gf_vcf}")
    if final_mod_vcf:
        log(f"  VCF (final):   {final_mod_vcf}")
    if phased:
        log(f"  VCF (phased):  {phased}")
    if bstats:
        log(f"  Block N50:     {bstats['N50']:,} bp")
        log(f"  Block count:   {bstats['block_count']}")
    log(f"\n  Total time:    {total:.1f}s ({total / 60:.1f}min)")
    log("=" * 60)

    summary = {
        "mode": "phasing",
        "bam": bam_path, "region": f"{contig}:{start}-{end}", "ref": ref_path,
        "sample": args.sample_name, "final_vcf": final_mod_vcf, "raw_vcf": raw_vcf,
        "phased_vcf": phased, "block_stats": bstats, "total_seconds": round(total, 2),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


if __name__ == "__main__":
    main()
