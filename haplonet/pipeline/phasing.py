# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""LongPhase joint SNP+methylation phasing."""

import json
import os
import subprocess
import sys


def run_longphase(mod_vcf_path, bam_path, snp_vcf_path, ref_path, output_dir, sample_name):
    """Run LongPhase phasing. Returns (phased_vcf_gz, block_stats_dict)."""
    for tool in ["longphase", "bcftools"]:
        try:
            subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"[WARN] Missing '{tool}', skipping LongPhase")
            return None, None
    out_prefix = os.path.join(output_dir, f"{sample_name}_phased")
    lp = ["longphase", "phase", "-s", snp_vcf_path, "--mod-file", mod_vcf_path,
         "-b", bam_path, "-r", ref_path, "-t", "16", "-o", out_prefix, "--ont"]
    print("[INFO] Running LongPhase...")
    r = subprocess.run(lp, capture_output=True, text=True, timeout=1800)
    pv = f"{out_prefix}.vcf"
    if not os.path.exists(pv):
        print("[WARN] LongPhase produced no output")
        return None, None
    gz = pv + ".gz"
    subprocess.run(["bgzip", "-c", pv], stdout=open(gz, "wb"), check=False)
    subprocess.run(["tabix", "-f", "-p", "vcf", gz], capture_output=True, check=False)
    return gz, _calc_block_n50(gz)


def _calc_block_n50(gz):
    """Calculate phase block N50/N90 stats from phased VCF."""
    try:
        out = subprocess.run(["bcftools", "view", "-H", gz], capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return {}
    blocks = {}
    for line in out.strip().split("\n"):
        if not line:
            continue
        f = line.split("\t")
        ff = f[9].split(":")
        ps = next((i for i, x in enumerate(ff) if x == "PS"), -1)
        gt = next((i for i, x in enumerate(ff) if x == "GT"), -1)
        if ps < 0 or gt < 0:
            continue
        sf = f[9].split(":")
        g = sf[gt]
        p = sf[ps] if ps < len(sf) else "."
        if g not in ("0|1", "1|0") or p == ".":
            continue
        blocks.setdefault(p, []).append(int(f[1]))
    blens = sorted((max(v) - min(v)) for v in blocks.values() if len(v) >= 2 and max(v) > min(v))
    tb = sum(blens)
    n = len(blens)

    def calc(pct):
        target = tb * pct / 100.0
        cum = 0
        for b in blens:
            cum += b
            if cum >= target:
                return b
        return blens[-1] if blens else 0

    s = {"block_count": n, "total_bp": tb, "N50": calc(50), "N90": calc(90),
         "max_block": max(blens) if blens else 0,
         "avg_block": tb / n if n else 0,
         "median_block": blens[n // 2] if blens else 0}
    print(f"[INFO] Block stats: N50={s['N50']:,}bp, count={n}")
    return s
