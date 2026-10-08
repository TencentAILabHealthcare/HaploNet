# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Graph consistency filter on VCF."""

import subprocess
import sys


def graph_filter_vcf(input_vcf, output_vcf, connect_adjacent=5, connect_confidence=0.80,
                     min_support_floor=4, pipeline_script=None, project_root=None):
    """Graph consistency filter on VCF."""
    if pipeline_script and os.path.isfile(pipeline_script):
        cmd = [sys.executable or "python", pipeline_script, "graph-filter",
             "--input", input_vcf, "--output", output_vcf,
             "--connect-adjacent", str(connect_adjacent),
             "--connect-confidence", str(connect_confidence),
             "--min-support-floor", str(min_support_floor),
             "--allow-single-sided", "--ignore-gt"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(output_vcf):
            return output_vcf
        print(f"[ERROR] Graph filter failed (exit={r.returncode})")
        return input_vcf
    print("[WARN] No pipeline script found for graph filter"); return input_vcf
