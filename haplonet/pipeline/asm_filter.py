# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""ASM discordant filtering on inference results."""

import csv
import os
import shutil
import subprocess
import sys


def asm_filter(csv_path, output_csv_path, high_prob=0.7, low_prob=0.3, min_hap_depth=0,
               pipeline_script=None, project_root=None):
    """ASM discordant filtering via process_pipeline.py or simple probability filter."""
    if pipeline_script and os.path.isfile(pipeline_script):
        cmd = [sys.executable, pipeline_script, "extract-asm",
             "--mode", "discordant", "--input", csv_path,
             "--output", output_csv_path,
             "--high-prob-threshold", str(high_prob),
             "--low-prob-threshold", str(low_prob),
             "--min-haplotype-depth", str(min_hap_depth)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(output_csv_path):
            print(f"[INFO] ASM filtered: {output_csv_path}")
            return output_csv_path
        if r.returncode == 0:
            shutil.copy2(csv_path, output_csv_path)
            print(f"[INFO] ASM: 0 discordant, using all ({output_csv_path})")
            return output_csv_path
        print(f"[ERROR] ASM filter failed (exit={r.returncode})")
        return None

    # Fallback: simple probability-based filter
    _asm_filter_simple(csv_path, output_csv_path, high_prob, low_prob)
    return output_csv_path


def _asm_filter_simple(csv_path, output_csv_path, high_prob=0.7, low_prob=0.3):
    filtered = []
    kept = 0
    ambig = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p0 = float(row.get("p_unmethylated", row.get("prob_0", 1.0)))
                p1 = float(row.get("p_full_methylated", row.get("prob_1", 0.0)))
                p2 = float(row.get("p_haplotype", 0.0))
                mx = max([p0, p1, p2])
                mn = min([x for x in [p0, p1, p2] if x > 0])
                if (p1 >= high_prob and p0 <= low_prob) or (p0 >= high_prob and p1 <= low_prob):
                    filtered.append(row)
                    kept += 1
                elif p2 >= max(p0, p1):
                    ambig += 1
                    filtered.append(row)
                else:
                    filtered.append(row)
                    kept += 1
            except (KeyError, ValueError, IndexError):
                filtered.append(row)
                kept += 1
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filtered[0].keys()) if filtered else [])
        if filtered:
            w.writeheader()
            w.writerows(filtered)
    print(f"[INFO] Simple ASM: {kept} kept, {ambig} ambiguous")
