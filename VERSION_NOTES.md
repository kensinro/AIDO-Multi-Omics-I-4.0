# VERSION_NOTES

## Formal analysis line for AIDO-Multi-Omics-I-4.0

Use the following scripts as the manuscript-supporting workflow:

1. `src/01_internal_benchmark_v3_NARGAB.py`
2. `src/02_external_validation_v5_CLINICAL_OVERRIDES.py`
3. `src/03_evidence_packaging_v1.py`

## Why V3 is the main internal benchmark

V3 is the formal NARGAB-upgrade pipeline. It extends earlier internal versions with:

- FDR-aware sensitivity;
- per-representation fragility flags;
- signal-lost flags;
- true readiness-cutoff sensitivity using K = 5, 10, 15, 20;
- size-matched random gene-set baseline;
- random BP-burden baseline;
- repeated split validation;
- optional additional GMT support.

## Legacy scripts

The `legacy/` folder contains converted UTF-8 copies of older V1/V2 scripts. They are retained for provenance only and should not be used as the formal manuscript analysis unless explicitly stated.

## External validation interpretation

The V5 clinical-overrides script should be described as external validation / external stress testing. METABRIC BRCA is the most usable non-TCGA validation anchor. Small CPTAC datasets should be interpreted cautiously. KIRP TCGA cBioPortal backup is not independent external validation.
