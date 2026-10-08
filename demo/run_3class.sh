#!/bin/bash

# run_3class.sh - HaploNet 3-Class Methylation Classification
#
# Pipeline: BAM -> Complete MSA (padding=16) -> 3-class inference -> GT evaluation
#
# Usage:
#   ./run_3class.sh                    # HG002, chr1, CPU
#   ./run_3class.sh HG003 chr2         # Custom sample + chromosome
#   ./run_3class.sh --skip-inference    # Use existing MSA/CSV

set -uo pipefail

############################################
# Path Configuration
############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

WEIGHT_DIR="${PROJECT_ROOT}/weight"

REF_FASTA="${WEIGHT_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
EXAMPLE_BAM="${WEIGHT_DIR}/phased.haplotag.bam"

VCALL_ROOT="${WORKSPACE_ROOT}/variant_calling"
INFER_PY="${SCRIPT_DIR}/infer_3class.py"
EVAL_SCRIPT="${VCALL_ROOT}/evaluate_dorado_methy.py"

COMPLETE_PADDING=${COMPLETE_PADDING:-16}
MIN_DEPTH=${MIN_DEPTH:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
SAMPLE_ID=${SAMPLE_ID:-HG002}
CONTIG=${CONTIG:-chr1}

DEMO_OUTPUT="${SCRIPT_DIR}/store"
MSA_OUT="${DEMO_OUTPUT}/msa"
INFER_OUT="${DEMO_OUTPUT}/inference"
EVAL_OUT="${DEMO_OUTPUT}/evaluation"

export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

############################################
# Argument Parsing
############################################

SKIP_INF=false; SKIP_EVAL=false; BAM_OVR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-inference) SKIP_INF=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --contig) CONTIG="$2"; shift 2 ;;
        --sample) SAMPLE_ID="$2"; shift 2 ;;
        --bam) BAM_OVR="$2"; shift 2 ;;
        --padding) COMPLETE_PADDING="$2"; shift 2 ;;
        --min-depth) MIN_DEPTH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [SAMPLE] [CONTIG]"
            echo ""
            echo "HaploNet 3-Class Methylation Classification"
            echo "Options: --bam, --contig, --sample, --padding, --min-depth, --skip-inference, --skip-eval"
            exit 0 ;;
        *) shift ;; esac
done

[[ "${SAMPLE_ID}" =~ ^[0-9]+$ ]] && SAMPLE_ID=$(printf "HG%03d" "$SAMPLE_ID") || SAMPLE_ID=${SAMPLE_ID^^}

log() { echo "[$(date '+%H:%M:%S')] $1"; }

############################################
# Run
############################################

log "============================================"
log "HaploNet 3-Class Classification"
log "============================================"
log "Sample:   ${SAMPLE_ID}"
log "Contig:   ${CONTIG}"
log "Padding:  ${COMPLETE_PADDING}  MinDepth: ${MIN_DEPTH}"
echo ""

mkdir -p "${DEMO_OUTPUT}"/{msa,inference,evaluation}

if [ -f "${INFER_PY}" ]; then
    python3 "${INFER_PY}" \
        --bam "${BAM_OVR:-${EXAMPLE_BAM}}" \
        --region "${CONTIG}:1-248956422" \
        --ref "${REF_FASTA}" \
        --sample "${SAMPLE_ID}" \
        --output-dir "${DEMO_OUTPUT}" \
        --padding "${COMPLETE_PADDING}" \
        --min-depth "${MIN_DEPTH}" \
        --batch-size "${BATCH_SIZE}" \
        ${SKIP_EVAL:+--skip-eval} \
        --skip-check
fi

echo ""; log "Outputs in: ${DEMO_OUTPUT}/"
