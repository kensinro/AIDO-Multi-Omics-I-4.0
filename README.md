# AIDO-Multi-Omics-I-4.0

Representation-first, endpoint-specific, robustness-tested pathway-level multi-omics benchmark for cancer cohorts.

This repository contains the analysis scripts used to generate the AIDO-Multi-Omics-I-4.0 evidence package for the NARGAB-oriented manuscript draft.

## Study concept

The analysis does **not** assume that multi-omics integration is automatically better than gene expression alone. Instead, it compares matched pathway-level representations:

- `GE`
- `GE_CN`
- `GE_MU`
- `GE_CN_MU`

For each cohort, biological process, representation, and endpoint, the workflow computes endpoint discriminability:

```text
D = -log10(p)
```

Integrated representations are compared against the matched `GE` baseline using Delta-D:

```text
DeltaD = D_integrated - D_GE
```

The framework labels biological processes as GE-sufficient, CN-informative, MU-informative, multi-layer-informative, endpoint-weak, or integration-fragile.

## Repository structure

```text
AIDO-Multi-Omics-I-4.0-GitHub/
├── README.md
├── requirements.txt
├── VERSION_NOTES.md
├── MANIFEST.md
├── .gitignore
├── src/
│   ├── 01_internal_benchmark_v3_NARGAB.py
│   ├── 02_external_validation_v5_CLINICAL_OVERRIDES.py
│   └── 03_evidence_packaging_v1.py
├── docs/
│   ├── DATA_LAYOUT.md
│   └── OUTPUTS.md
└── legacy/
    ├── AIDO-Multi-Omics-I-4.0-V1_legacy_utf8.py
    └── AIDO-Multi-Omics-I-4.0-V2_legacy_utf8.py
```

## Recommended execution order

Run the scripts from the repository root after preparing the expected local data folders.

```bash
python src/01_internal_benchmark_v3_NARGAB.py
python src/02_external_validation_v5_CLINICAL_OVERRIDES.py
python src/03_evidence_packaging_v1.py
```

The scripts currently use Windows-style local paths in their CONFIG sections, for example:

```text
D:/AIDO-Data/UCSC_XENA/
D:/AIDO-Data/GSEA/
D:/AIDO-Data/External/
D:/AIDO-Temp/
```

Before running on another machine, edit the CONFIG block at the top of each script.

## Main scripts

### 1. Internal benchmark

`src/01_internal_benchmark_v3_NARGAB.py`

This is the formal internal TCGA benchmark backbone. It includes:

- robust UTF-8 / UTF-16 / UTF-8-SIG / Latin1 table reader;
- clinical patient-ID suffix remapping;
- GE, GE+CN, GE+MU, and GE+CN+MU representation construction;
- OS and STAGE endpoint discriminability;
- Delta-D over GE baseline;
- integration fragility, strong fragility, and signal-lost flags;
- FDR sensitivity by endpoint and by endpoint-cohort;
- true observation-readiness K sensitivity: `K = 5, 10, 15, 20`;
- size-matched random gene-set baseline;
- random BP-burden baseline;
- repeated train/test split validation.

### 2. External validation / stress test

`src/02_external_validation_v5_CLINICAL_OVERRIDES.py`

This script uses selected TCGA V3 records and evaluates external transportability / stress-test behavior in external or cBioPortal-format datasets. The intended interpretation is cautious: this is not clinical-grade validation.

Supported default datasets include:

- METABRIC BRCA;
- CPTAC UCEC;
- CPTAC LUAD;
- RCC CPTAC GDC;
- KIRP TCGA cBioPortal backup, included as a reproducibility/sanity check rather than independent external validation.

### 3. Evidence packaging

`src/03_evidence_packaging_v1.py`

This script does not rerun the full benchmark. It consumes already generated internal and external outputs and creates manuscript-ready summary tables and figures.

## Python environment

A minimal environment is listed in `requirements.txt`. Example setup:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell/CMD style may vary
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input data note

Large TCGA, GSEA, METABRIC, CPTAC, or cBioPortal data files are not included in this code package. Keep raw data outside the repository and document their local paths. The `.gitignore` is configured to avoid accidentally committing large data/output folders.

## Output note

The scripts write outputs to timestamped folders under `D:/AIDO-Temp/` by default. Most CSV outputs use UTF-8-SIG encoding for Excel compatibility.

## Interpretation boundaries

- `D = -log10(p)` is used as a task-discriminability statistic, not a direct effect-size estimate.
- Repeated split validation is used for representation-selection stability, not clinical prediction.
- BP-burden analyses are secondary stress tests, not clinical deployment models.
- Small external cohorts should not be over-interpreted.
- KIRP cBioPortal backup is a reproducibility/sanity check, not an independent external validation dataset.

## Suggested GitHub description

> Code for AIDO-Multi-Omics-I-4.0: a representation-first benchmark evaluating endpoint-specific pathway-level multi-omics gain and integration fragility across cancer cohorts.
