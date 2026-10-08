#!/bin/bash

# run_phasing.sh - HaploNet Phasing Inference (full pipeline)
#
# Pipeline: BAM -> MSA -> 2-class inference -> ASM filter -> VCF -> Graph filter -> Ratio filter -> LongPhase
#
# Usage:
#   ./run_phasing.sh                         # Default region, CPU
#   ./run_phasing.sh --region chr1:1-5000000 # Custom region
#   ./run_phasing.sh --skip-inference        # Skip if CSV exists

set -uo pipefail

############################################
# Path Configuration
############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

WEIGHT_DIR="${PROJECT_ROOT}/weight"

REF_FASTA="${WEIGHT_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
MODEL_WEIGHT="${WEIGHT_DIR}/haplotype.pt"
EXAMPLE_BAM="${WEIGHT_DIR}/phased.haplotag.bam"
SNP_VCF="${WEIGHT_DIR}/snp.vcf.gz"

VCALL_ROOT="${WORKSPACE_ROOT}/variant_calling"
PIPELINE_PY="${PROJECT_ROOT}/evalute/process_pipeline.py"
INFER_PY="${SCRIPT_DIR}/infer_phasing.py"
EVAL_SCRIPT="${VCALL_ROOT}/evaluate_dorado_methy.py"
LONGPHASE_BIN="${WORKSPACE_ROOT}/buddy2/aigp/longphase_v1.7.3/longphase_linux-x64"

INFERENCE_CFG="${VCALL_ROOT}/experiments/delploy/haplo_infer.yaml"

DEMO_OUTPUT="${SCRIPT_DIR}/store"
MSA_WORK="${DEMO_OUTPUT}/msa"
INFER_OUT="${DEMO_OUTPUT}/inference"
VCF_DIR="${DEMO_OUTPUT}/vcf"
METRICS_DIR="${DEMO_OUTPUT}/metrics"

SAMPLE_NAME="HG002"
REGION="${REGION:-chr1:68049857-68051097}"
CONTIG="${REGION%%:*}"
START_POS="${REGION##*:}"
START_POS="${START_POS%%-*}"

THREADS=${THREADS:-8}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
GPU_COUNT=${GPU_COUNT:-1}

ASM_HIGH_PROB=0.7
ASM_LOW_PROB=0.3
GRAPH_CA=5
GRAPH_CC=0.80
GRAPH_MS=4
RATIO_LOW=0.25
QUAL_THRESHOLD=127
MODIFICATION=m
LP_THREADS=16

############################################
# Argument Parsing
############################################

SKIP_INFERENCE=false
START_STEP=1
SKIP_PHASE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --start-step) START_STEP="$2"; shift 2 ;;
        --region) REGION="$2"; CONTIG="${REGION%%:*}"; START_POS="${REGION##*-}"; shift 2 ;;
        --sample) SAMPLE_NAME="$2"; shift 2 ;;
        --skip-phase) SKIP_PHASE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "HaploNet Phasing Pipeline"
            echo ""
            echo "Options:"
            echo "  --region CHR:START-END  Genomic region"
            echo "  --sample NAME          Sample name"
            echo "  --skip-inference       Skip inference step"
            echo "  --start-step N         Start from step N (1-7)"
            echo "  --skip-phase           Skip LongPhase phasing"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $1"; }
log_error() { echo "[$(date '+%H:%M:%S')] ERROR: $1" >&2; }
check_file() { [ -f "$1" ] || { log_error "Not found: $1"; return 1; }; }
count_vcf_records() { bcftools view -H "$1" 2>/dev/null | awk 'END{print NR+0}' || echo 0; }

export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

############################################
# Pre-flight
############################################

log "============================================"
log "HaploNet Phasing Pipeline"
log "============================================"
log "Sample:     ${SAMPLE_NAME}"
log "Region:     ${REGION}"
log "Output:     ${DEMO_OUTPUT}"
echo ""

mkdir -p "${DEMO_OUTPUT}"/{msa,inference,vcf,graph_filtered,longphase,metrics}
for f in "${REF_FASTA}" "${EXAMPLE_BAM}"; do check_file "$f" || true; done
T_START=$(date +%s)

############################################
# Step 1: Model Inference
############################################

if [ ${START_STEP} -le 1 ]; then
    log "[Step 1/7] Model Inference"
    EXISTING_CSV=$(find "${INFER_OUT}" -name "*methy_class.csv" 2>/dev/null | head -n1)
    if [ "${SKIP_INFERENCE}" = true ] && [ -n "${EXISTING_CSV}" ]; then
        log "Skip (using existing): ${EXISTING_CSV}"
    elif [ -f "${INFER_PY}" ]; then
        python3 "${INFER_PY}" \
            --bam "${EXAMPLE_BAM}" --region "${REGION}" --ref "${REF_FASTA}" \
            --sample-name "${SAMPLE_NAME}" --output-dir "${DEMO_OUTPUT}" \
            --batch-size "${BATCH_SIZE}" --keep-temp --skip-check
        EXISTING_CSV=$(find "${INFER_OUT}" -name "*methy_class.csv" 2>/dev/null | head -n1)
    fi
    PRED_CSV="${EXISTING_CSV:-$(find ${INFER_OUT} -name '*methy_class.csv' | head -n1)}"
else
    PRED_CSV=$(find "${INFER_OUT}" -name "*methy_class.csv" 2>/dev/null | head -n1)
fi
log "Inference CSV: ${PRED_CSV}"
echo ""

############################################
# Step 2: ASM Filter
############################################

if [ ${START_STEP} -le 2 ]; then
    log "[Step 2/7] ASM Filtering"
    ASM_CSV="${VCF_DIR}/${SAMPLE_NAME}_asm_filtered.csv"
    python3 "${PIPELINE_PY}" extract-asm --mode discordant \
        --input "${PRED_CSV}" --output "${ASM_CSV}" \
        --high-prob-threshold "${ASM_HIGH_PROB}" --low-prob-threshold "${ASM_LOW_PROB}"
fi
echo ""

############################################
# Steps 3-7 (abbreviated — full logic in Python script)
############################################

if [ ${START_STEP} -le 3 ]; then
    log "[Step 3/7] CSV -> VCF conversion"
    RAW_VCF_GZ="${VCF_DIR}/${SAMPLE_NAME}_raw_mod.vcf.gz"
    python3 "${PIPELINE_PY}" csv2vcf --csv "${ASM_CSV:-${PRED_CSV}}" --bam "${EXAMPLE_BAM}" \
        --output "${RAW_VCF_GZ}" --fasta "${REF_FASTA}" --sample "${SAMPLE_NAME}" \
        --threads "${THREADS}" --qual-threshold "${QUAL_THRESHOLD}" --mode stage0
fi

if [ ${START_STEP} -le 4 ]; then
    log "[Step 4/7] Graph Consistency Filter"
    FILTERED_VCF_GZ="${DEMO_OUTPUT}/graph_filtered/filtered.vcf.gz"
    mkdir -p "$(dirname "${FILTERED_VCF_GZ}")"
    python3 "${PIPELINE_PY}" graph-filter --input "${RAW_VCF_GZ}" --output "${FILTERED_VCF_GZ}" \
        --connect-adjacent "${GRAPH_CA}" --connect-confidence "${GRAPH_CC}" --min-support-floor "${GRAPH_MS}"
fi

if [ ${START_STEP} -le 5 ]; then
    log "[Step 5/7] Ratio Filter"
    RATIO_VCF_GZ="${DEMO_OUTPUT}/graph_filtered/ratio_filtered.vcf.gz"
    # Inline ratio filter via bcftools + awk (same as original infer.sh)
    { bcftools view -h "${FILTERED_VCF_GZ}"; bcftools view -H "${FILTERED_VCF_GZ}" | awk -v rlow="${RATIO_LOW}" '
      BEGIN{OFS="\t"} {total++; n=split($9,fmt,":"); md_i=ud_i=0;
       for(i=1;i<=n;i++){if(fmt[i]=="MD")md_i=i; else if(fmt[i]=="UD")ud_i=i}
       if(md_i==0||ud_i==0){kept++; print; next}
       split($10,s,":"); md=s[md_i]+0; ud=s[ud_i]+0; mx=(md>ud?md:ud); mn=(md<ud?md:ud);
       if(mx<=0){kept++; print;next}; ratio=mn/mx; if(ratio<rlow){drop++;next}; kept++; print}
       END{printf "total\t%d\nkept\t%d\ndropped\t%d\n",total,kept,drop+0}'
    } > "${RATIO_VCF_GZ%.gz}" && bgzip -f -c "${RATIO_VCF_GZ%.gz}" > "${RATIO_VCF_GZ}" && tabix -f -p vcf "${RATIO_VCF_GZ}"; }
fi

LP_VCF_GZ=""
if [ ${START_STEP} -le 6 ] && [ "${SKIP_PHASE}" != true ]; then
    log "[Step 6/7] LongPhase Phasing"
    LP_OUT_PREFIX="${DEMO_OUTPUT}/longphase/${SAMPLE_NAME}_${CONTIG}"
    if [ -f "${SNP_VCF}" ] && [ -f "${LONGPHASE_BIN}" ]; then
        "${LONGPHASE_BIN}" phase -s "${SNP_VCF}" --mod-file "${RATIO_VCF_GZ}" -b "${EXAMPLE_BAM}" \
            -r "${REF_FASTA}" -t "${LP_THREADS}" -o "${LP_OUT_PREFIX}" --ont
        LP_VCF_GZ=$(ls "${DEMO_OUTPUT}/longphase/"*.vcf.gz 2>/dev/null | head -n1)
    fi
fi

if [ ${START_STEP} -le 7 ]; then
    log "[Step 7/7] Metrics"
    # Summary stats written by Python script already
    :
fi

T_END=$(date +%s); ELAPSED=$(( T_END - T_START ))
log ""; log "Done! Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))min)"
