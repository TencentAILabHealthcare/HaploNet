# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""High-level MSA building functions: CpG extraction, candidate loading, BAM->MSA."""

import json
import os
import pickle
import pysam

from .bam_reader import (
    make_sequencing_msa_transbp,
    make_sequencing_msa_haplotype_transbp,
    padding_msa,
)
from .constants import Candidate, check_seq


def extract_cpg_sites(bam_path, ref_path, contig, start, end, output_jsonl):
    """Extract CpG sites from reference genome within region. Writes JSONL."""
    ref_fasta = pysam.FastaFile(ref_path)
    ref_seq = ref_fasta.fetch(contig, start-1, end).upper()
    ref_fasta.close()
    cpg_sites = []
    pos_in_ref = start
    for i in range(len(ref_seq)-1):
        if ref_seq[i] == 'C' and ref_seq[i+1] == 'G':
            cpg_sites.append({"chr": contig, "pos": pos_in_ref+i-1, "beta_mean": 0.5})
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w') as f:
        for s in cpg_sites:
            f.write(json.dumps(s)+'\n')
    return len(cpg_sites)


def load_gt_candidates(gt_csv, contig, start, end):
    """Load candidate methylation sites from GT CSV within region."""
    import csv
    candidates = []
    if gt_csv and os.path.isfile(gt_csv):
        with open(gt_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 3:
                    continue
                chrom = row[0].strip()
                if chrom != contig:
                    continue
                pos = int(row[1])
                gt_val = int(row[2])
                if start <= pos < end:
                    candidates.append(Candidate(chrom, pos, gt_val))
        candidates.sort(key=lambda c: c.pos)
    return candidates


def build_msa(contig, bam_path, ref_path, cpg_jsonl, msa_output_dir,
               padding=5, min_depth=4, phasing=True):
    """Build per-haplotype MSA for Mode A (2-class ASM detection)."""
    from intervaltree import IntervalTree, Interval
    candidates = []
    with open(cpg_jsonl) as f:
        for line in f:
            d = json.loads(line.strip())
            candidates.append(Candidate(d["chr"], d["pos"], d.get("beta_mean", 0.5)))

    ff = pysam.FastaFile(ref_path); ref_chr_seq = ff.fetch(contig); seq_len = len(ref_chr_seq)
    af = pysam.AlignmentFile(bam_path, "rb"); records = []

    for can in candidates:
        pos = can.pos - 1; ws = pos - padding; we = pos + padding + 1
        lp = max(0, -ws); rp = max(0, we - seq_len); ws = max(ws, 0); we = min(we, seq_len)
        ref_seq = ("."*lp) + ref_chr_seq[ws:we] + "."*rp; ref_seq = check_seq(ref_seq)

        msa1, msa2, msa_u = make_sequencing_msa_haplotype_transbp(
            af, contig, ws, we, max_depth=200, phasing=phasing)
        if msa1 is None or "seqs" not in msa1:
            continue
        d1 = len(msa1["seqs"]); d2 = len(msa2["seqs"])
        if d1 < min_depth or d2 < min_depth:
            continue
        for hi, all_msa in enumerate([msa1, msa2], 1):
            if lp > 0 or rp > 0:
                wl = 2*padding+1; all_msa = padding_msa(all_msa, lp, rp, wl)
            data = {
                "ref_seq": ref_seq, "msa": all_msa["seqs"], "ins": all_msa["insertions"],
                "strands": all_msa["strands"], "mapping_qualities": all_msa["mapping_qualities"],
                "base_qualities": all_msa["base_qualities"], "base_counts": all_msa["base_counts"],
                "breakpoints": all_msa["breakpoints"], "depth": len(all_msa["seqs"]),
            }
            if phasing:
                data["haplotype"] = all_msa["haplotype"]
            data["meta"] = {
                "id": f"{contig}:{ws}:{hi}", "chrom": contig, "pos": pos, "start": ws,
                "gt": can.beta, "haplotype_idx": hi, "depth1": d1, "depth2": d2,
            }
            records.append(data)
    af.close()
    msa_file = os.path.join(msa_output_dir, f"{contig}_msa.pkl.bin")
    with open(msa_file, 'wb') as f:
        pickle.dump(records, f)
    return msa_file


def build_complete_msa(contig, bam_path, ref_path, candidates, msa_output_dir,
                        padding=16, min_depth=8, phasing=True):
    """Build single aggregated MSA for Mode B (3-class classification)."""
    from intervaltree import IntervalTree
    ff = pysam.FastaFile(ref_path); ref_chr_seq = ff.fetch(contig); seq_len = len(ref_chr_seq)
    af = pysam.AlignmentFile(bam_path, "rb"); records = []

    for can in candidates:
        pos = can.pos; ws = pos - padding; we = pos + padding + 1
        lp = max(0, -ws); rp = max(0, we - seq_len); ws = max(ws, 0); we = min(we, seq_len)
        ref_seq = ("."*lp) + ref_chr_seq[ws:we] + "."*rp; ref_seq = check_seq(ref_seq)
        all_msa = make_sequencing_msa_transbp(af, contig, ws, we, max_depth=200, phasing=phasing)
        if all_msa is None:
            continue
        depth = len(all_msa.get("seqs", []))
        if depth < min_depth:
            continue
        if lp > 0 or rp > 0:
            wl = 2*padding+1; all_msa = padding_msa(all_msa, lp, rp, wl)
        data = {
            "ref_seq": ref_seq, "msa": all_msa["seqs"], "ins": all_msa.get("insertions", []),
            "strands": all_msa.get("strands", []),
            "mapping_qualities": all_msa.get("mapping_qualities", []),
            "base_qualities": all_msa.get("base_qualities", []),
            "base_counts": all_msa.get("base_counts", {}),
            "breakpoints": all_msa.get("breakpoints", []),
            "depth": depth,
        }
        if phasing and "haplotype" in all_msa:
            data["haplotype"] = all_msa["haplotype"]
        data["meta"] = {
            "id": f"{contig}:{pos}", "chrom": contig, "pos": pos, "start": ws,
            "gt": can.gt, "beta": can.gt, "depth": depth,
        }
        if phasing:
            data["meta"]["depth1"] = all_msa.get("depth1", 0)
            data["meta"]["depth2"] = all_msa.get("depth2", 0)
        records.append(data)
    af.close(); ff.close()
    msa_file = os.path.join(msa_output_dir, f"{contig}_msa.pkl.bin")
    os.makedirs(msa_output_dir, exist_ok=True)
    with open(msa_file, 'wb') as f:
        pickle.dump(records, f)
    return msa_file
