# HaploNet Demo Scripts

HaploNet provides **two inference modes** for DNA methylation analysis from ONT sequencing data.

## Quick Start

```bash
cd demo/

# Mode 1: Phasing (per-haplotype binary + ASM + VCF + LongPhase)
python infer_phasing.py --region chr1:68050000-68050500 --cpu --skip-check

# Mode 2: 3-Class Classification (unmethylated / methylated / intermediate)
python infer_3class.py --region chr1:68000000-68100000 --cpu --skip-check
```

Shell wrappers are also available:
```bash
./run_phasing.sh    # Full phasing pipeline
./run_3class.sh     # 3-class classification pipeline
```

---

## Mode 1: `infer_phasing.py` — Per-Haplotype Methylation & Phasing

Performs **per-haplotype binary methylation classification**, identifies **ASM sites**, and optionally runs **LongPhase joint SNP+methylation phasing**.

### What it does

| Step | Action | Output |
|------|--------|--------|
| 1 | CpG extraction + MSA build from BAM (per-haplotype) | `msa/*.pkl.bin` |
| 2 | 2-class model inference (methylated / unmethylated) | `inference/*methy_class.csv` |
| 3 | ASM discordant filter | `vcf/*_asm_filtered.csv` |
| 4 | CSV → VCF conversion | `vcf/*_raw_mod.vcf.gz` |
| 5 | Graph consistency filter | `graph_filtered/filtered.vcf.gz` |
| 6 | Depth ratio filter | `graph_filtered/ratio_filtered.vcf.gz` |
| 7 | LongPhase phasing (optional) | `longphase/*.vcf.gz` |

### Input

| File | Required | Description |
|------|:-------:|-------------|
| Phased BAM (`phased.haplotag.bam`) | Yes | BAM with MM/ML/HP tags from Dorado + WhatsHap |
| Reference FASTA (`*.fna` + `*.fai`) | Yes | GRCh38 no-alt analysis set |
| Model weights (`haplotype.pt`) | Yes | 2-class model (~120 MB) |
| SNP VCF (`snp.vcf.gz`) | Optional | For LongPhase phasing; without it, phasing is skipped |

### Output

| File | Description |
|------|-------------|
| `vcf/graph_filtered/ratio_filtered.vcf.gz` | Final filtered mod VCF (ASM sites) |
| `longphase/{sample}_phased.vcf.gz` | Phased VCF with PS tags (requires SNP VCF) |
| `summary.json` | Phase block stats (N50, N90, count, switch error) |

### Key parameters

| Parameter | Default | Note |
|-----------|---------|------|
| MSA window | 11 bp (padding=5) | Small window for per-haplotype resolution |
| Min depth | 4 | Per haplotype |
| Classes | 2 | methylated / unmethylated |
| Output | `store/phasing_{region}_{ts}/` | Auto-created timestamped directory |

### Command-line arguments

```bash
python infer_phasing.py \
    --bam ../weight/phased.haplotag.bam \
    --ref ../weight/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
    --model-weight ../weight/haplotype.pt \
    --region chr1:68050000-68050500 \
    --sample-name HG002 \
    --snp-vcf ../weight/snp.vcf.gz \
    --padding 5 --min-depth 4 \
    --batch-size 64 --cpu \
    --output-dir ./store
```

### Whole-chromosome example

```bash
# Infer entire chr1
python infer_phasing.py --region chr1 --cpu --skip-check

# With custom params
python infer_phasing.py --region chr1 --padding 5 --min-depth 4 --batch-size 1024 --skip-check
```

---

## Mode 2: `infer_3class.py` — Three-Class Methylation Classification

Classifies each CpG site into one of three methylation states:

| Class | State | Beta value range |
|:-----:|-------|-----------------|
| 0 | Unmethylated | beta < 0.3 |
| 1 | Fully methylated | beta > 0.7 |
| 2 | Intermediate / ASM | 0.3 <= beta <= 0.7 |

### What it does

| Step | Action | Output |
|------|--------|--------|
| 1 | Load GT candidates + Build complete MSA (all reads pooled) | `msa/*.pkl.bin` |
| 2 | 3-class model inference | `inference/predictions.csv` |
| 3 | Ground truth evaluation (if GT labels available) | `evaluation/eval_report.txt` |

### Input

| File | Required | Description |
|------|:-------:|-------------|
| Phased BAM (`phased.haplotag.bam`) | Yes | BAM with MM/ML/HP tags |
| Reference FASTA (`*.fna` + `*.fai`) | Yes | GRCh38 no-alt analysis set |
| Model weights (`complete.pt`) | Yes | 3-class model (~120 MB) |
| GT labels (`HG002_methy_labels.csv` or `.jsonl`) | Optional | For accuracy evaluation |

### Key differences from Mode 1

| Parameter | Mode 1 (Phasing) | Mode 2 (3-Class) |
|-----------|-------------------|------------------|
| MSA window | 11 bp (padding=5) | 33 bp (**padding=16**) |
| Min depth | 4 | **8** (stricter) |
| Classes | 2 | **3** |
| Read source | Per-haplotype (HP-split) | All reads aggregated |
| Output | VCF + phased VCF | Predictions CSV + eval report |
| Post-processing | ASM/VCF/graph/ratio/phasing | None (optional GT eval) |

### Command-line arguments

```bash
python infer_3class.py \
    --bam ../weight/phased.haplotag.bam \
    --ref ../weight/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
    --model-weight ../weight/complete.pt \
    --gt-csv ../weight/HG002_methy_labels.csv \
    --region chr1:68000000-68100000 \
    --sample HG002 \
    --padding 16 --min-depth 8 \
    --batch-size 64 --cpu \
    --skip-eval \
    --output-dir ./store
```

### Whole-chromosome example

```bash
python infer_3class.py --contig chr1 --cpu --skip-check
```

---

## Download Weights & Data

Model weights are **NOT included** in the repository. Download before running:

**Link**: [Baidu Netdisk](https://pan.baidu.com/s/1b9Ev9Ux18zENpY7PKTnwow) (extraction code: `ntj3`)

```bash
cd HaploNet/
# Download HaploNet_weights.zip, then:
unzip HaploNet_weights.zip -d ./
ls weight/*.pt weight/*.fna   # Verify
```

Minimum files to run inference: `haplotype.pt` or `complete.pt`, `*.fna`, `*.fai`, `phased.haplotag.bam`, `phased.haplotag.bam.bai`.

For full phasing pipeline: also need `snp.vcf.gz`.
For accuracy evaluation: additionally need `HG002_gt_labels.jsonl`.

See the main [README.md](../README.md) for full data sources and preprocessing instructions.
