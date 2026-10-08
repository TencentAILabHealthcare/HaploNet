# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""
BAM read parsing and MSA matrix construction.

Contains two MSA builders:
- make_sequencing_msa_transbp: single aggregated MSA (used by complete/3-class mode)
- make_sequencing_msa_haplotype_transbp: split by HP1/HP2/unphased (used by haplotype/2-class mode)
"""

import numpy as np
import pysam

from .constants import (BASE28_TYPES, DefaultOrderedDict, IUPAC_TO_BASE,
                       check_base, check_seq, is_interval_in)


def parse_align_read(read, start, end, phasing=False, with_ml=False):
    """Parse a single aligned read into base sequence, qualities, and optional ML tags."""
    ref_pos = read.reference_start
    if ref_pos >= end:
        return None
    cigars = read.cigartuples
    if cigars is None or read.query_sequence is None:
        return None
    if phasing:
        hap = read.get_tag("HP") if read.has_tag("HP") else 0
    query_seq = check_seq(read.query_sequence.upper())
    base_qualities = list(read.query_qualities)

    ml_values = []
    methylated_refs = set()
    if with_ml:
        qlen = len(query_seq)
        ml_values = [0] * qlen
        if read.has_tag("MM"):
            try:
                mod_info = read.modified_bases
                try:
                    ref_positions = read.get_reference_positions(full_length=True)
                except Exception:
                    ref_positions = None
                for key, positions in mod_info.items():
                    mod_base, strand, mod_type = key
                    if mod_base.upper() == 'C' and mod_type == 'm':
                        for pos_in_read, likelihood in positions:
                            if pos_in_read < qlen:
                                ml_values[pos_in_read] = likelihood
                                if likelihood > 127 and ref_positions is not None:
                                    if pos_in_read < len(ref_positions):
                                        ref_p = ref_positions[pos_in_read]
                                        if ref_p is not None and ref_p >= 0:
                                            methylated_refs.add(ref_p)
            except (KeyError, IndexError, AttributeError):
                pass

    seq = []; bq = []; ml_out = []; insertions = []
    query_pos = 0; last_pos = ref_pos
    for op, length in cigars:
        if op in (pysam.CSOFT_CLIP,):
            query_pos += length
        elif op in (pysam.CREF_SKIP,):
            ref_pos += length
        elif op in (pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF):
            length = min(ref_pos + length, end) - ref_pos
            if start < ref_pos + length:
                p = query_pos + max(ref_pos, start) - ref_pos
                seq.extend(query_seq[p:p+length])
                bq.extend(base_qualities[p:p+length])
                if with_ml:
                    ml_out.extend(ml_values[p:p+length])
            ref_pos += length; query_pos += length; last_pos = ref_pos
        elif op == pysam.CINS:
            if ref_pos >= start:
                ins_len = min(ref_pos + length, end) - ref_pos
                ins = [ref_pos - start, query_seq[query_pos:query_pos+ins_len],
                      base_qualities[query_pos:query_pos+ins_len]]
                if with_ml:
                    ins.append(ml_values[query_pos:query_pos+ins_len])
                insertions.append(ins)
            query_pos += length
        elif op == pysam.CDEL:
            length = min(ref_pos + length, end) - ref_pos
            if start < ref_pos + length:
                try: qual = base_qualities[query_pos]
                except IndexError: qual = 0
                seq.extend(["-"] * length); bq.extend([qual] * length)
                if with_ml:
                    ml_out.extend([0] * length)
            ref_pos += length; last_pos = ref_pos
        if ref_pos >= end:
            break
    if len(seq) == 0:
        return None
    output = {"ref_pos": last_pos - len(seq), "bases": seq, "qualities": bq, "insertions": insertions}
    if with_ml:
        output["ml"] = ml_values
        if methylated_refs:
            output["methylated_refs"] = methylated_refs
    if phasing:
        output["haplotype"] = hap
    return output


def _format_msa(msa_obj, start, window):
    """Format raw MSA dict into standardized output."""
    if len(msa_obj) == 0:
        return None
    names = list(msa_obj.keys()); all_msa = {}
    for n in names:
        lines = msa_obj[n]
        for k, v in lines.items():
            if k == "intervals":
                continue
            if v and isinstance(v[0], str):
                all_msa[k] = "".join(v)
            else:
                all_msa[k] = v
    all_msa = dict(all_msa)
    intervals = [msa_obj[n].get("intervals") for n in names]
    breakpoints = []
    for ts in intervals:
        if ts:
            breakpoints.append([(t.begin - start, t.end - start) for t in ts])
        else:
            breakpoints.append([])
    all_msa["breakpoints"] = breakpoints
    all_msa["base_counts"] = {t: [] for t in BASE28_TYPES}
    all_msa["depth"] = len(all_msa.get("seqs", []))
    return all_msa


def make_sequencing_msa_haplotype_transbp(af, contig, start, end, min_bq=0, min_mq=0,
                                          max_depth=200, phasing=True):
    """Build MSA split by haplotype (HP1 / HP2 / unphased). Used by Mode A."""
    aln_reads = af.fetch(contig, start, end)
    window = end - start

    def msa_line():
        d = {"base_qualities":[0]*window, "mapping_qualities":[0]*window,
           "strands":[3]*window, "seqs":["."]*window, "insertions":[],
           "intervals":[]}
        if phasing: d["haplotype"] = [0]*window
        return d
    msa1 = DefaultOrderedDict(msa_line)
    msa2 = DefaultOrderedDict(msa_line)
    msa_u = DefaultOrderedDict(msa_line)
    msas = {1: msa1, 2: msa2, 0: msa_u}

    for cnt, align in enumerate(aln_reads):
        if align.is_unmapped or align.query_sequence is None:
            continue
        if align.is_secondary or align.is_supplementary:
            continue
        if align.mapping_quality < min_mq:
            continue
        output = parse_align_read(align, start, end, phasing=phasing, with_ml=True)
        if output is None:
            continue
        hap_val = output.get("haplotype")
        if hap_val not in (1, 2):
            hap_val = 0
        msa = msas[hap_val]
        ref_start = output["ref_pos"]; read_len = len(output["bases"])
        fragment_id = align.query_name or f"read{cnt}"
        if fragment_id not in msa:
            for rid in list(msa.keys()):
                ints = msa[rid]["intervals"]
                if (not ints) or (isinstance(ints, list) and len(ints) == 0):
                    msa[fragment_id] = msa.pop(rid); break
        line = msa[fragment_id]
        strand = 0 if align.is_forward else 1
        bases = line["seqs"]; bqs = line["base_qualities"]
        mqs = line["mapping_qualities"]; strands = line["strands"]
        line["insertions"].extend(output["insertions"])
        haps = line.get("haplotype")
        obases = output["bases"]; oquals = output["qualities"]
        omls = output.get("ml", [])
        methy_refs = output.get("methylated_refs", set())
        ref_j_start = ref_start
        for j in range(len(obases)):
            i = ref_start - start + j
            if i < 0 or i >= window:
                continue
            b2 = obases[j]; q2 = oquals[j]; ml2 = omls[j] if j < len(omls) else 0
            b1 = bases[i]; q1 = bqs[i]
            if q2 < min_bq:
                continue
            if b2 == 'C':
                this_ref_pos = ref_j_start + j
                is_methylated = False
                if methy_refs and this_ref_pos in methy_refs:
                    is_methylated = True
                elif ml2 > 127:
                    is_methylated = True
                if is_methylated:
                    b2 = 'M'
            if q2 >= q1 or b1 in (".", "N"):
                bases[i] = b2; bqs[i] = q2
                if haps is not None:
                    haps[i] = hap_val
            if strands[i] == 3:
                strands[i] = strand; mqs[i] = align.mapping_quality
            else:
                strands[i] = 2; mqs[i] = max(align.mapping_quality, mqs[i])
        if max_depth is not None and len(msa) > max_depth:
            break

    return (_format_msa(msa1, start, window),
            _format_msa(msa2, start, window),
            _format_msa(msa_u, start, window))


def make_sequencing_msa_transbp(af, contig, start, end, min_bq=0, min_mq=0,
                                  max_depth=200, phasing=True, combine_fragment=True):
    """Build single aggregated MSA (NOT split by haplotype). Used by Mode B (3-class)."""
    af = af if hasattr(af, 'fetch') else af
    try:
        aln_reads = af.fetch(contig, start, end, multiple_iterators=False)
    except TypeError:
        aln_reads = af.fetch(contig, start, end)
    window = end - start
    _base28 = list(BASE28_TYPES)
    base_counts = {t: [0]*window for t in _base28}

    def msa_line():
        d = {"base_qualities":[0]*window, "mapping_qualities":[0]*window,
           "strands":[3]*window, "seqs":["."]*window, "insertions":[],
           "intervals":[]}
        if phasing: d["haplotype"] = [0]*window
        return d
    msa = DefaultOrderedDict(msa_line)

    for cnt, align in enumerate(aln_reads):
        if align.is_unmapped or align.query_sequence is None:
            continue
        if align.is_secondary or align.is_supplementary:
            continue
        mapping_quality = align.mapping_quality
        if mapping_quality < min_mq:
            continue
        output = parse_align_read(align, start, end, phasing=phasing, with_ml=True)
        if output is None:
            continue
        hap_val = output.get("haplotype")
        ref_start = output["ref_pos"]; read_len = len(output["bases"]); ref_end = ref_start + read_len
        fragment_id = align.query_name or f"read{cnt}"
        if fragment_id is None or not combine_fragment:
            fragment_id = f"read{cnt}"
        if fragment_id not in msa and combine_fragment:
            for read_id in list(msa.keys()):
                if not is_interval_in(msa[read_id]["intervals"], ref_start, ref_end):
                    msa[fragment_id] = msa.pop(read_id); break
        line = msa[fragment_id]
        strand = 0 if align.is_forward else 1
        bases = line["seqs"]; bqs = line["base_qualities"]
        mqs = line["mapping_qualities"]; strands = line["strands"]
        line["insertions"].extend(output["insertions"])
        haps = line.get("haplotype") if phasing else None
        obases = output["bases"]; oquals = output["qualities"]
        omls = output.get("ml", [])
        methy_refs = output.get("methylated_refs", set())
        ref_j_start = ref_start

        for j in range(len(obases)):
            i = ref_start - start + j
            if i < 0 or i >= window:
                continue
            base2 = obases[j]; qual2 = oquals[j]
            ml2 = omls[j] if j < len(omls) else 0
            base1 = bases[i]; qual1 = bqs[i]
            if qual2 < min_bq:
                continue
            if base2 == 'C':
                this_ref_pos = ref_j_start + j
                is_methylated = False
                if methy_refs and this_ref_pos in methy_refs:
                    is_methylated = True
                elif ml2 > 127:
                    is_methylated = True
                if is_methylated:
                    base2 = 'M'
            if qual2 >= qual1 or base1 in (".", "N"):
                bases[i] = base2; bqs[i] = qual2
                if haps is not None:
                    haps[i] = hap_val
            if strands[i] == 3:
                strands[i] = strand; mqs[i] = mapping_quality
            else:
                strands[i] = 2; mqs[i] = max(mapping_quality, mqs[i])
            cbase = base2
            if cbase == "-": cbase = "D"
            cbase = cbase if strand else cbase.lower()
            base_counts[cbase][i] += 1
        for ins in output["insertions"]:
            if len(ins) == 4:
                ipos, ibases, iquals, isignal = ins
            elif len(ins) >= 2:
                ipos, ibases = ins[0], ins[1]
            else:
                continue
            if strand:
                base_counts["I"][ipos] += 1
            else:
                base_counts["i"][ipos] += 1
            for ii, ibase in enumerate(ibases, start=ipos):
                base_counts[f"{ibase}+"][ii] += 1
        if max_depth is not None and len(msa) > max_depth:
            break

    if len(msa) == 0:
        return None
    names = list(msa.keys()); all_msa = {}
    for n in names:
        lines = msa[n]
        for k, v in lines.items():
            if len(v) > 0 and isinstance(v[0], str):
                all_msa[k] = "".join(v)
            else:
                all_msa[k] = v
    all_msa = dict(all_msa)
    intervals = all_msa.pop("intervals"); breakpoints = []
    for ts in intervals:
        if ts:
            breakpoints.append([(t.begin-start, t.end-start) for t in ts])
        else:
            breakpoints.append([])
    all_msa["breakpoints"] = breakpoints
    all_msa["base_counts"] = base_counts
    all_msa["depth"] = len(all_msa.get("seqs", []))
    if phasing:
        depth1 = depth2 = 0
        for hap_list in all_msa.get('haplotype', []):
            read_hap = next((h for h in hap_list if h != 0), 0)
            if read_hap == 1: depth1 += 1
            elif read_hap == 2: depth2 += 1
        all_msa['depth1'] = depth1; all_msa['depth2'] = depth2
    return all_msa


def padding_msa(msa, left_pad, right_pad, window_len):
    """Pad MSA with left/right reference context for boundary positions."""
    out = {}; seqs = msa.get("seqs", []); n = len(seqs)
    seg_len = len(seqs[0]) if n > 0 and seqs[0] else 0
    out["seqs"] = ["."*left_pad + (seqs[i] if i < n else "") + "."*right_pad for i in range(n)]
    ins = msa.get("insertions", []); padded_ins = []
    for i in range(n):
        il = ins[i] if i < len(ins) else []
        padded_ins.append([[item[0]+left_pad] + item[1:] for item in il])

    def pad_list(data, pad_val):
        padded = []
        for i in range(n):
            arr = data[i] if i < len(data) else []
            arr = list(arr) + [pad_val]*(seg_len-len(arr)) if len(arr)<seg_len else arr[:seg_len]
            padded.append([pad_val]*left_pad + arr + [pad_val]*right_pad)
        return padded

    out["ins"] = padded_ins
    out["base_qualities"] = pad_list(msa.get("base_qualities", []), 0)
    out["mapping_qualities"] = pad_list(msa.get("mapping_qualities", []), 0)
    out["strands"] = pad_list(msa.get("strands", []), 3)
    if "haplotype" in msa:
        out["haplotype"] = pad_list(msa.get("haplotype", []), 0)
    out["base_counts"] = {}
    base_counts = msa.get("base_counts", {})
    for k in BASE28_TYPES:
        arr = np.zeros(window_len, dtype=np.int32); v = base_counts.get(k, [])
        v = list(v) + [0]*(seg_len-len(v)) if len(v)<seg_len else v[:seg_len]
        if seg_len > 0: arr[left_pad:left_pad+seg_len] = v
        out["base_counts"][k] = arr
    bps = msa.get("breakpoints", []); padded_bps = []
    for i in range(n):
        bl = bps[i] if i < len(bps) else []
        padded_bps.append([(s+left_pad, e+left_pad) for s,e in bl])
    out["breakpoints"] = padded_bps
    if "depth" in msa: out["depth"] = msa["depth"]
    return out
