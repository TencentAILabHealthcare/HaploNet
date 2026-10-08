# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Depth ratio filter on VCF using bcftools+awk."""

import os
import shutil
import subprocess


def _is_gzip(path):
    try:
        with open(path, 'rb') as f: return f.read(2) == b'\x1f\x8b'
    except: return False


def ratio_filter_vcf(input_vcf, output_vcf, ratio_low=0.25, bcftools=None, bgzip=None, tabix=None):
    """Filter VCF sites by depth ratio min(MD,UD)/max(MD,UD) >= threshold."""
    def _find_tool(name, *extra_paths):
        p = shutil.which(name)
        if p: return p
        for ep in extra_paths:
            if os.path.isfile(ep): return ep
        return None
    _bc = _find_tool("bcftools") or bcftools
    _bg = _find_tool("bgzip") or bgzip
    _tb = _find_tool("tabix") or tabix
    if not all([_bc, _bg, _tb]):
        print("[WARN] Missing tools, skipping ratio_filter"); return None

    out_vcf = output_vcf.replace(".gz", "") if output_vcf.endswith(".gz") else output_vcf
    out_gz = out_vcf + ".gz"
    awk_cmd = f"""awk -v rlow={ratio_low} 'BEGIN {{ OFS="\\t"; total=0; kept=0; drop_ratio=0 }} {{
        total++
        n=split($9, fmt, ":") ; md_i=0; ud_i=0
        for(i=1;i<=n;i++){{ if(fmt[i]=="MD") md_i=i; else if(fmt[i]=="UD") ud_i=i }}
        if(md_i==0 || ud_i==0) {{ kept++; print; next }}
        split($10, s, ":") ; md=s[md_i]+0; ud=s[ud_i]+0
        mx=(md>ud?md:ud); mn=(md<ud?md:ud)
        if(mx<=0){{ kept++; print; next }}
        ratio=mn/mx
        if(ratio < rlow){{ drop_ratio++; next }}
        kept++; print
    }} END {{ printf "total\\t%d\\nkept\\t%d\\ndrop\\t%d\\n", total, kept, drop_ratio }}'"""

    hdr = subprocess.run([_bc, "view", "-h", input_vcf], capture_output=True, text=True, timeout=120).stdout
    body = subprocess.run([_bc, "view", "-H", input_vcf], capture_output=True, text=True, timeout=600).stdout
    awk_proc = subprocess.run(awk_cmd, input=body.stdout, capture_output=True, text=True, timeout=600, shell=True, executable="/bin/awk")
    with open(out_vcf, 'w') as f: f.write(hdr.stdout + awk_proc.stdout)
    subprocess.run([_bg, "-f", "-c", out_vcf], stdout=open(out_gz,'wb'), check=False, timeout=120)
    subprocess.run([_tb, "-f", "-p", "vcf", out_gz], capture_output=True, check=False, timeout=60)
    return out_gz
