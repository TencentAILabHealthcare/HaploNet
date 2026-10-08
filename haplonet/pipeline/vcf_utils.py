# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""VCF generation from inference CSV."""

import subprocess
import sys
import os


def csv_to_vcf(csv_path, bam_path, ref_path, vcf_output_path, sample_name="SAMPLE",
              threads=32, pipeline_script=None, project_root=None):
    """Convert inference CSV to VCF format via process_pipeline.py."""
    if pipeline_script and os.path.isfile(pipeline_script):
        cmd = [sys.executable, pipeline_script, "csv2vcf",
             "--csv", csv_path, "--bam", bam_path,
             "--output", vcf_output_path, "--fasta", ref_path,
             "--sample", sample_name, "--threads", str(threads),
             "--modification", "m", "--mode", "stage0",
             "--tag-style", "longphase_like", "--genotype-style", "fixed_het",
             "--mapq-threshold", "1",
             "--mod-threshold", "0.8", "--unmod-threshold", "0.2",
             "--heter-ratio-threshold", "0.6", "--noise-ratio-threshold", "0.2"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(vcf_output_path):
            return vcf_output_path
        print(f"[ERROR] VCF conversion failed (exit={r.returncode})"); return None
    print("[WARN] No pipeline script found for VCF conversion"); return None
