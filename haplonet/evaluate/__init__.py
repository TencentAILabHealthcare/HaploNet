# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""Ground truth evaluation metrics."""

import os
import json
import csv

import numpy as np
try:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
except ImportError:
    pass  # evaluation will be skipped


def run_evaluation_inline(pred_csv, sample_id, contig, eval_output_dir,
                          gt_jsonl_candidates=None, weight_dir=None, workspace_root=None):
    """Compare predictions vs GT labels (JSONL). Returns True if eval ran."""
    if gt_jsonl_candidates is None:
        gt_jsonl_candidates = []
        if weight_dir:
            gt_jsonl_candidates = [
                os.path.join(weight_dir, f"{sample_id}_gt_labels.jsonl"),
            ]
        if workspace_root:
            gt_jsonl_candidates.append(
                os.path.join(workspace_root, "data/lvm_data/methy_gt", sample_id,
                             "training_labels_with_beta.jsonl")
            )
    truth_jsonl = None
    for c in gt_jsonl_candidates:
        if os.path.isfile(c):
            truth_jsonl = c; break
    if not truth_jsonl:
        print("[WARN] Ground truth JSONL not found, skipping evaluation"); return False

    print(f"[INFO] Loading GT from: {truth_jsonl}")
    gt_dict = {}; gt_count = 0
    with open(truth_jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                key = (rec.get("chrom", rec.get("chr", "")), rec.get("pos", rec.get("position", 0)))
                gt_dict[key] = rec; gt_count += 1
            except (json.JSONDecodeError, KeyError): pass
    print(f"[INFO] Loaded {gt_count} GT entries")

    os.makedirs(eval_output_dir, exist_ok=True)
    y_true_success, y_pred_success = [], []; y_true_comprehensive, y_pred_comprehensive = [], []
    sites_not_found = 0; sites_ambiguous = 0; total_pred = 0

    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_pred += 1; chrom = row["chrom"]; pos = int(row["pos"]); gt_key = (chrom, pos)
            true_label = None
            if gt_key in gt_dict:
                gt_rec = gt_dict[gt_key]; beta = gt_rec.get("beta", gt_rec.get("beta_mean", gt_rec.get("methylation", 0.5)))
                if isinstance(beta, (list, tuple)): beta = beta[0] if beta else 0.5
                if beta > 0.9: true_label = 1
                elif beta < 0.1: true_label = 0
            if true_label is None: continue
            y_true_comprehensive.append(true_label)
            pred_label = -1
            try:
                p0 = float(row["p_unmethylated"]); p1 = float(row["p_full_methylated"]); p2 = float(row["p_haplotype"])
                max_idx = int(np.argmax([p0, p1, p2]))
                if max_idx == 1: pred_label = 1
                elif max_idx == 0: pred_label = 0
            except (KeyError, ValueError): sites_not_found += 1; y_pred_comprehensive.append(0); continue

            if pred_label != -1:
                y_true_success.append(true_label); y_pred_success.append(pred_label); y_pred_comprehensive.append(pred_label)
            else: sites_ambiguous += 1; y_pred_comprehensive.append(1 - true_label)

    total_truth = len(y_true_comprehensive)
    if total_truth == 0:
        print("[WARN] No matched GT entries"); return False

    # Compute metrics
    acc_succ = accuracy_score(y_true_success, y_pred_success) if y_true_success else 0.0
    cm_succ = confusion_matrix(y_true_success, y_pred_success, labels=[0,1]) if y_true_success else None
    prec_succ, rec_succ, f1_succ = 0.0, 0.0, 0.0
    if cm_succ is not None and cm_succ.size == 4:
        tn, fp, fn, tp = cm_succ.ravel()
        prec_succ = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec_succ = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1_succ = f1_score(y_true_success, y_pred_success, average='macro') if y_true_success else 0.0

    acc_comp = accuracy_score(y_true_comprehensive, y_pred_comprehensive)
    f1_comp = f1_score(y_true_comprehensive, y_pred_comprehensive, average='macro')
    cm_all = confusion_matrix(y_true_comprehensive, y_pred_comprehensive, labels=[0,1])
    prec_comp, rec_comp = 0.0, 0.0
    if cm_all.size == 4:
        tn_a, fp_a, fn_a, tp_a = cm_all.ravel()
        prec_comp = tp_a/(tp_a+fp_a) if (tp_a+fp_a) > 0 else 0.0
        rec_comp = tp_a/(tp_a+fn_a) if (tp_a+fn_a) > 0 else 0.0

    report_path = os.path.join(eval_output_dir, "eval_report.txt")
    cls_names = {0:"unmethylated", 1:"methylated"}
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report: {sample_id} {contig}\n{'='*60}\n\n")
        f.write(f"Total predictions: {total_pred}\nMatched GT: {total_truth}\n\n")
        f.write(f"--- Coverage ---\nTotal GT: {total_truth}\n")
        f.write(f"Called & Classified: {len(y_true_success)}\nAmbiguous: {sites_ambiguous}\nNot Found: {sites_not_found}\n\n")
        f.write(f"--- Accuracy (Called) ---\nAcc={acc_succ:.4f}, Prec={prec_succ:.4f}\nF1={f1_succ:.4f}\n")
        if cm_succ and cm_succ.size==4:
            tn_s,fp_s,fn_s,tp_s=cm_succ.ravel(); f.write(f"\nConfusion Matrix:\n{tn_s:>7d}{fp_s:>7d}\n{fn_s:>7d}{tp_s:>7d}\n")
        f.write(f"\n--- Comprehensive (All) ---\nAcc={acc_comp:.4f}, F1={f1_comp:.4f}\n")

    detail_path = os.path.join(eval_output_dir, "eval_details.csv")
    eval_results = []
    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        for row in f:
            chrom=row["chrom"]; pos=int(row["pos"]); gt_key=(chrom,pos)
            if gt_key not in gt_dict: continue
            gt_rec=gt_dict[gt_key]; beta=gt_rec.get("beta",gt_rec.get("beta_mean",0.5))
            if isinstance(beta,(list,tuple)): beta=beta[0] if beta else 0.5
            if not (beta>0.9 or beta<0.1): continue
            true_cls = 1 if beta>0.9 else 0
            try:
                p0=float(row["p_unmethylated"]); p1=float(row["p_full_methylated"]); p2=float(row["p_haplotype"])
                max_idx=int(np.argmax([p0,p1,p2]))
                pred_bin=1 if max_idx==1 else (0 if max_idx==0 else -1)
            except (KeyError, ValueError): continue
            eval_results.append({"chrom":chrom,"pos":pos,"true":true_cls,"pred_bin":pred_bin,"beta":beta,"p0":p0,"p1":p1,"p2":p2})
    if eval_results:
        with open(detail_path,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=["chrom","pos","true","pred_bin","beta","p0","p1","p2"])
            writer.writeheader(); writer.writerows(eval_results)

    print(f"[INFO] Evaluation done! Called: acc={acc_succ:.4f}, F1={f1_succ:.4f}")
    print(f"[INFO] All: acc={acc_comp:.4f}, F1={f1_comp:.4f}")
    return True
