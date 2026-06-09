# -*- coding: utf-8 -*-
"""
AIDO-Multi-Omics-I-4.0
External validation / external stress-test pipeline v1

Purpose
-------
Use selected TCGA V3 representation records to test external transportability
in non-TCGA / external cBioPortal-format cohorts.

External datasets supported by default
--------------------------------------
1. METABRIC BRCA
   D:/AIDO-Data/External/brca_metabric/

2. CPTAC UCEC
   D:/AIDO-Data/External/ucec_cptac_2020/

3. CPTAC LUAD
   D:/AIDO-Data/External/luad_cptac_2020/

4. RCC CPTAC GDC
   D:/AIDO-Data/External/rcc_cptac_gdc/

5. Optional KIRP TCGA PanCancer Atlas cBioPortal-format backup
   D:/AIDO-Data/External/kirp_tcga_pan_can_atlas_2018/
   This is NOT external validation. It is included only as a format/reference check.

Main outputs
------------
D:/AIDO-Temp/AIDO_MultiOmics_I_4_external_validation_<timestamp>/

Key output files:
- external_validation_all_records.csv
- external_validation_summary_by_dataset_endpoint.csv
- external_validation_summary_by_dataset_sourcecohort_endpoint.csv
- external_validation_top_supported_records.csv
- external_validation_data_availability.csv

Interpretation
--------------
This script does not claim clinical-grade validation.
It performs selected external transport / stress-test analysis for
representation-first BP signals identified in TCGA V3.

The primary transport question is:
Do TCGA-selected BP representations retain endpoint-discriminability or
consistent direction in external datasets?
"""

from __future__ import annotations

import re
import math
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIG
# =============================================================================

AIDO_TEMP = Path("D:/AIDO-Temp")
EXTERNAL_ROOT = Path("D:/AIDO-Data/External")

# Optional METABRIC clinical override exported from cBioPortal Clinical Data tab.
# This fixes Tumor Stage parsing when cBioPortal staging files omit or rename stage fields.
# The file may be placed either in External/ or inside External/brca_metabric/.
METABRIC_CLINICAL_OVERRIDE_CANDIDATES = [
    EXTERNAL_ROOT / "brca_metabric_clinical_data.tsv",
    EXTERNAL_ROOT / "brca_metabric" / "brca_metabric_clinical_data.tsv",
]

UCEC_CLINICAL_OVERRIDE_CANDIDATES = [
    EXTERNAL_ROOT / "ucec_cptac_2020_clinical_data.tsv",
    EXTERNAL_ROOT / "ucec_cptac_2020" / "ucec_cptac_2020_clinical_data.tsv",
]

LUAD_CLINICAL_OVERRIDE_CANDIDATES = [
    EXTERNAL_ROOT / "luad_cptac_2020_clinical_data.tsv",
    EXTERNAL_ROOT / "luad_cptac_2020" / "luad_cptac_2020_clinical_data.tsv",
]

KIRP_CLINICAL_OVERRIDE_CANDIDATES = [
    EXTERNAL_ROOT / "kirp_tcga_pan_can_atlas_2018_clinical_data.tsv",
    EXTERNAL_ROOT / "kirp_tcga_pan_can_atlas_2018" / "kirp_tcga_pan_can_atlas_2018_clinical_data.tsv",
]

GSEA_DIR = Path("D:/AIDO-Data/GSEA")
HALLMARK_GMT = GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt"

# If None, auto-detect latest V3 output under D:/AIDO-Temp/.
TCGA_V3_OUT: Optional[Path] = None

OUT_DIR = AIDO_TEMP / f"AIDO_MultiOmics_I_4_external_validation_v5_CLINICAL_OVERRIDES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

PRIMARY_READINESS_K = 10
D_THRESHOLD = -math.log10(0.05)

# Limit records per source cohort/endpoint to avoid external validation being dominated by one cohort.
# Set None to use all endpoint-informative TCGA records.
MAX_TCGA_RECORDS_PER_COHORT_ENDPOINT: Optional[int] = 50

# Which TCGA selected records to send to external validation.
# Recommended: endpoint_informative == 1, primary K, Hallmark.
USE_Q_FILTER = False
Q_THRESHOLD = 0.10

# External support threshold.
EXTERNAL_D_SUPPORT = D_THRESHOLD

RANDOM_SEED = 20260604
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

REPRESENTATIONS = {
    "GE": ["GE"],
    "GE_CN": ["GE", "CN"],
    "GE_MU": ["GE", "MU"],
    "GE_CN_MU": ["GE", "CN", "MU"],
}

# Dataset definitions. Filenames are checked in priority order.
EXTERNAL_DATASETS = [
    {
        "dataset_id": "METABRIC_BRCA",
        "label": "BRCA / METABRIC",
        "dir": EXTERNAL_ROOT / "brca_metabric",
        "source_cohorts": ["BRCA"],
        "external_type": "non_TCGA_external",
        "expression_candidates": [
            "data_mrna_illumina_microarray.txt",
            "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        ],
        "cna_candidates": ["data_cna.txt"],
        "mutation_candidates": ["data_mutations.txt"],
        "clinical_patient_candidates": ["data_clinical_patient.txt"],
        "clinical_sample_candidates": ["data_clinical_sample.txt"],
        "external_endpoints": ["OS", "RFS", "PFS", "DSS", "STAGE"],
    },
    {
        "dataset_id": "CPTAC_UCEC",
        "label": "UCEC / CPTAC 2020",
        "dir": EXTERNAL_ROOT / "ucec_cptac_2020",
        "source_cohorts": ["UCEC"],
        "external_type": "non_TCGA_external",
        "expression_candidates": [
            "data_mrna_seq_rsem.txt",
            "data_mrna_seq_rsem_zscores_ref_all_samples.txt",
            "data_mrna_seq_rsem_zscores_ref_diploid_samples.txt",
        ],
        "cna_candidates": ["data_cna.txt", "data_log2cna.txt"],
        "mutation_candidates": ["data_mutations.txt"],
        "clinical_patient_candidates": ["data_clinical_patient.txt"],
        "clinical_sample_candidates": ["data_clinical_sample.txt"],
        "external_endpoints": ["OS", "RFS", "PFS", "DSS", "STAGE"],
    },
    {
        "dataset_id": "CPTAC_LUAD",
        "label": "LUAD / CPTAC 2020",
        "dir": EXTERNAL_ROOT / "luad_cptac_2020",
        "source_cohorts": ["LUAD"],
        "external_type": "non_TCGA_external",
        "expression_candidates": [
            "data_mrna_seq_rpkm.txt",
            "data_mrna_seq_rpkm_zscores_ref_all_samples.txt",
        ],
        "cna_candidates": ["data_cna.txt", "data_log2cna.txt"],
        "mutation_candidates": ["data_mutations.txt"],
        "clinical_patient_candidates": ["data_clinical_patient.txt"],
        "clinical_sample_candidates": ["data_clinical_sample.txt"],
        "external_endpoints": ["OS", "RFS", "PFS", "DSS", "STAGE"],
    },
    {
        "dataset_id": "CPTAC_RCC_GDC",
        "label": "RCC / CPTAC GDC",
        "dir": EXTERNAL_ROOT / "rcc_cptac_gdc",
        "source_cohorts": ["KIRC", "KIRP"],
        "external_type": "renal_lineage_external_stress_test",
        "expression_candidates": [
            "data_mrna_seq_tpm.txt",
            "data_mrna_seq_fpkm.txt",
            "data_mrna_seq_tpm_zscores_ref_all_samples.txt",
            "data_mrna_seq_fpkm_zscores_ref_all_samples.txt",
        ],
        "cna_candidates": ["data_cna.txt"],
        "mutation_candidates": ["data_mutations.txt"],
        "clinical_patient_candidates": ["data_clinical_patient.txt"],
        "clinical_sample_candidates": ["data_clinical_sample.txt"],
        "external_endpoints": ["OS", "RFS", "PFS", "DSS", "STAGE"],
    },
    {
        "dataset_id": "KIRP_TCGA_CBIO_BACKUP",
        "label": "KIRP / TCGA PanCancer Atlas cBioPortal backup",
        "dir": EXTERNAL_ROOT / "kirp_tcga_pan_can_atlas_2018",
        "source_cohorts": ["KIRP"],
        "external_type": "TCGA_backup_not_external",
        "expression_candidates": [
            "data_mrna_seq_v2_rsem.txt",
            "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
            "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
        ],
        "cna_candidates": ["data_cna.txt", "data_log2_cna.txt", "data_log2cna.txt"],
        "mutation_candidates": ["data_mutations.txt"],
        "clinical_patient_candidates": ["data_clinical_patient.txt"],
        "clinical_sample_candidates": ["data_clinical_sample.txt"],
        "external_endpoints": ["OS", "RFS", "PFS", "DSS", "STAGE"],
    },
]


# =============================================================================
# 1. UTILITIES
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_name(x: Any) -> str:
    x = str(x)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x).strip("_")


def neglog10_p(p: Any) -> float:
    if p is None or pd.isna(p):
        return np.nan
    try:
        p = float(p)
    except Exception:
        return np.nan
    if p <= 0:
        return 300.0
    return -math.log10(max(p, 1e-300))


def find_existing_file(folder: Path, candidates: List[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    for c in candidates:
        if c.lower() in files:
            return files[c.lower()]
    return None


def find_latest_tcga_v3_output(temp_root: Path) -> Path:
    candidates = []
    for p in temp_root.glob("AIDO_MultiOmics_I_4_internal_benchmark_v3_NARGAB_*"):
        if p.is_dir() and (p / "aggregate_tables" / "all_selected_representations.csv").exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No V3 output found under D:/AIDO-Temp/. Please set TCGA_V3_OUT manually.")
    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def read_table_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1", "utf-16", "utf-16-le"]
    seps = ["\t", ","]
    best = None
    best_score = -10**9
    best_info = None
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[0] == 0 or df.shape[1] == 0:
                    continue
                score = df.shape[1] * 10
                score -= sum(str(c).lower().startswith("unnamed") for c in df.columns)
                score -= ("�" in " ".join(map(str, df.columns))) * 100
                if df.shape[1] == 1:
                    score -= 50
                if score > best_score:
                    best = df
                    best_score = score
                    best_info = (enc, sep, df.shape)
            except Exception as e:
                last_err = e
    if best is None:
        raise RuntimeError(f"Cannot read {path}. Last error: {last_err}")
    best.columns = [str(c).replace("\ufeff", "").strip() for c in best.columns]
    log(f"Read {path.name}: encoding={best_info[0]}, sep={repr(best_info[1])}, shape={best_info[2]}")
    return best


def read_cbio_table(path: Path) -> pd.DataFrame:
    """
    Read cBioPortal staging table.
    Clinical files often have four metadata/comment rows starting with '#',
    followed by a real header line. pandas comment='#' handles this well.
    """
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False, encoding="utf-8-sig")
        if df.shape[1] > 1:
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df
    except Exception:
        pass

    # Fallback.
    return read_table_auto(path)


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def zscore_df_by_rows(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.apply(pd.to_numeric, errors="coerce")
    mean = arr.mean(axis=1)
    sd = arr.std(axis=1, ddof=0).replace(0, np.nan)
    return arr.sub(mean, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan)


# =============================================================================
# 2. GENE SETS
# =============================================================================

def load_hallmark_gmt(gmt_path: Path) -> Dict[str, Dict[str, Any]]:
    if not gmt_path.exists():
        raise FileNotFoundError(f"GMT not found: {gmt_path}")

    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = sorted(set(g.strip().upper() for g in parts[2:] if g.strip()))
            bp_key = f"HALLMARK::{name}"
            gene_sets[bp_key] = {
                "bp_key": bp_key,
                "bp": name,
                "collection": "HALLMARK",
                "genes": genes,
                "annotated_n": len(genes),
            }
    return gene_sets


def normalize_bp_key(x: str) -> str:
    s = str(x)
    if "::" in s:
        return s
    if s.startswith("HALLMARK_"):
        return f"HALLMARK::{s}"
    return s


# =============================================================================
# 3. EXTERNAL MOLECULAR LOADERS
# =============================================================================

def load_cbio_matrix(path: Path, layer: str) -> pd.DataFrame:
    """
    Convert cBioPortal matrix to genes x samples.

    Handles files like:
    Hugo_Symbol Entrez_Gene_Id SAMPLE1 SAMPLE2...
    """
    df = read_cbio_table(path)
    df.columns = [str(c).strip() for c in df.columns]

    gene_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["hugo_symbol", "gene", "genes", "symbol"]:
            gene_col = c
            break
    if gene_col is None:
        gene_col = df.columns[0]

    drop_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in ["entrez_gene_id", "entrez", "cytoband"] or c == gene_col:
            continue
        # sample columns are kept
    df[gene_col] = df[gene_col].astype(str).str.upper().str.strip()
    df = df[df[gene_col].notna() & (df[gene_col] != "") & (df[gene_col] != "NAN")]
    df = df.drop_duplicates(gene_col, keep="first").set_index(gene_col)

    for c in list(df.columns):
        cl = str(c).lower()
        if cl in ["entrez_gene_id", "entrez", "cytoband"]:
            df = df.drop(columns=[c], errors="ignore")

    mat = df.apply(pd.to_numeric, errors="coerce")
    mat.columns = [str(c).strip() for c in mat.columns]
    mat.index = mat.index.astype(str).str.upper()

    if layer == "MU":
        mat = mat.fillna(0)
        mat = (mat != 0).astype(float)

    return mat


def load_cbio_mutation_binary(path: Path) -> pd.DataFrame:
    df = read_cbio_table(path)
    df.columns = [str(c).strip() for c in df.columns]

    gene_col = None
    sample_col = None

    for c in df.columns:
        if str(c).lower() in ["hugo_symbol", "gene", "genes", "symbol"]:
            gene_col = c
            break

    sample_candidates = [
        "Tumor_Sample_Barcode", "Tumor_Sample_Id", "SAMPLE_ID",
        "Sample_ID", "sample", "Sample"
    ]
    for cand in sample_candidates:
        for c in df.columns:
            if str(c).lower() == cand.lower():
                sample_col = c
                break
        if sample_col is not None:
            break

    if gene_col is None or sample_col is None:
        raise RuntimeError(f"Cannot identify gene/sample columns in mutation file: {path}")

    tmp = df[[gene_col, sample_col]].dropna()
    tmp["gene"] = tmp[gene_col].astype(str).str.upper().str.strip()
    tmp["sample"] = tmp[sample_col].astype(str).str.strip()
    tmp = tmp[(tmp["gene"] != "") & (tmp["sample"] != "")]
    tmp["value"] = 1.0

    mat = tmp.drop_duplicates(["gene", "sample"]).pivot_table(
        index="gene", columns="sample", values="value", aggfunc="max", fill_value=0.0
    )
    mat.index = mat.index.astype(str).str.upper()
    mat.columns = [str(c).strip() for c in mat.columns]
    return mat.astype(float)


# =============================================================================
# 4. EXTERNAL CLINICAL PARSING
# =============================================================================

def find_col(df: pd.DataFrame, exact_or_contains: List[str]) -> Optional[str]:
    if df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for pat in exact_or_contains:
        pl = pat.lower()
        for cl, orig in lower.items():
            if cl == pl:
                return orig
    for pat in exact_or_contains:
        pl = pat.lower()
        for cl, orig in lower.items():
            if pl in cl:
                return orig
    return None


def parse_event_status(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    if s.startswith("1:") or s in ["1", "true", "dead", "deceased", "recurred", "relapsed", "progressed", "event", "yes"]:
        return 1
    if s.startswith("0:") or s in ["0", "false", "alive", "living", "diseasefree", "disease_free", "no", "censored"]:
        return 0
    if "deceased" in s or "dead" in s or "recur" in s or "relapse" in s or "progress" in s:
        return 1
    if "living" in s or "alive" in s or "disease free" in s or "diseasefree" in s:
        return 0
    return None


def parse_stage_group(x: Any) -> Optional[str]:
    """
    Convert heterogeneous clinical stage labels to EARLY / ADVANCED.

    Rules used in this external stress-test:
    - EARLY: stage 0 / I / II
    - ADVANCED: stage III / IV

    Handles numeric values such as 1, 1.0, 2.0 and text labels such as
    Stage IIA, IIIC, IVB.
    """
    if pd.isna(x):
        return None

    # Numeric stage support, including METABRIC Tumor Stage values 0.0-4.0.
    try:
        xf = float(x)
        if np.isfinite(xf):
            xi = int(round(xf))
            if abs(xf - xi) < 1e-6:
                if xi in [0, 1, 2]:
                    return "EARLY"
                if xi in [3, 4]:
                    return "ADVANCED"
    except Exception:
        pass

    s = str(x).strip().upper()
    if s in ["", "NAN", "NA", "NONE", "UNKNOWN", "NOT AVAILABLE", "NOT_APPLICABLE"]:
        return None
    if s in ["EARLY", "I/II", "I-II", "STAGE_EARLY"]:
        return "EARLY"
    if s in ["ADVANCED", "LATE", "III/IV", "III-IV", "STAGE_ADVANCED"]:
        return "ADVANCED"

    # Normalize common prefixes and punctuation.
    s2 = s
    for token in [
        "STAGE", "PATHOLOGIC", "PATHOLOGICAL", "CLINICAL",
        "AJCC", "TUMOR", "FIGO", "STAGE:"
    ]:
        s2 = s2.replace(token, "")
    s2 = s2.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")

    # Arabic numerals embedded in strings.
    if s2 in ["0", "0A", "0B", "1", "1A", "1B", "1C", "2", "2A", "2B", "2C"]:
        return "EARLY"
    if s2 in ["3", "3A", "3B", "3C", "4", "4A", "4B", "4C"]:
        return "ADVANCED"

    # Roman numerals.
    if s2.startswith("IV"):
        return "ADVANCED"
    if s2.startswith("III"):
        return "ADVANCED"
    if s2.startswith("II"):
        return "EARLY"
    if s2.startswith("I"):
        return "EARLY"

    return None


def load_external_clinical(patient_path: Optional[Path], sample_path: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient = read_cbio_table(patient_path) if patient_path is not None and patient_path.exists() else pd.DataFrame()
    sample = read_cbio_table(sample_path) if sample_path is not None and sample_path.exists() else pd.DataFrame()

    patient_id_col = find_col(patient, ["PATIENT_ID", "Patient Identifier", "patient"])
    sample_id_col = find_col(sample, ["SAMPLE_ID", "Sample Identifier", "sample"])
    sample_patient_col = find_col(sample, ["PATIENT_ID", "Patient Identifier", "patient"])

    if patient_id_col is None and not patient.empty:
        patient_id_col = patient.columns[0]
    if sample_id_col is None and not sample.empty:
        sample_id_col = sample.columns[0]

    if not patient.empty:
        patient = patient.copy()
        patient["PATIENT_ID_STD"] = patient[patient_id_col].astype(str).str.strip()
        patient = patient.drop_duplicates("PATIENT_ID_STD", keep="first").set_index("PATIENT_ID_STD", drop=False)

    if not sample.empty:
        sample = sample.copy()
        sample["SAMPLE_ID_STD"] = sample[sample_id_col].astype(str).str.strip()
        if sample_patient_col is not None:
            sample["PATIENT_ID_STD"] = sample[sample_patient_col].astype(str).str.strip()
        else:
            sample["PATIENT_ID_STD"] = sample["SAMPLE_ID_STD"]
        sample = sample.drop_duplicates("SAMPLE_ID_STD", keep="first").set_index("SAMPLE_ID_STD", drop=False)

    # Build sample-level clinical table.
    if not sample.empty:
        clin = sample.copy()
        if not patient.empty:
            p2 = patient.drop(columns=[c for c in ["PATIENT_ID_STD"] if c in patient.columns], errors="ignore")
            clin = clin.join(p2, on="PATIENT_ID_STD", rsuffix="_PATIENT")
    elif not patient.empty:
        clin = patient.copy()
        clin["SAMPLE_ID_STD"] = clin["PATIENT_ID_STD"]
        clin = clin.set_index("SAMPLE_ID_STD", drop=False)
    else:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(index=clin.index)
    out.index.name = "sample"
    out["sample"] = clin.index.astype(str)
    out["patient"] = clin["PATIENT_ID_STD"].astype(str) if "PATIENT_ID_STD" in clin.columns else clin.index.astype(str)

    # OS
    os_time_col = find_col(clin, [
        "OS_MONTHS", "Overall Survival (Months)", "OS.time", "OS_TIME",
        "OVERALL_SURVIVAL_MONTHS", "SURVIVAL_MONTHS", "MONTHS_TO_DEATH"
    ])
    os_status_col = find_col(clin, [
        "OS_STATUS", "Overall Survival Status", "VITAL_STATUS",
        "overall_survival_status"
    ])

    if os_time_col is not None:
        out["OS_time"] = pd.to_numeric(clin[os_time_col], errors="coerce")
    else:
        out["OS_time"] = np.nan
    if os_status_col is not None:
        out["OS_event"] = clin[os_status_col].map(parse_event_status)
    else:
        out["OS_event"] = np.nan

    # RFS / DFS / PFS
    rfs_time_col = find_col(clin, [
        "RFS_MONTHS", "Relapse Free Status (Months)", "DFS_MONTHS", "PFS_MONTHS",
        "DISEASE_FREE_MONTHS", "RELAPSE_FREE_MONTHS", "RECURRENCE_FREE_MONTHS"
    ])
    rfs_status_col = find_col(clin, [
        "RFS_STATUS", "Relapse Free Status", "DFS_STATUS", "PFS_STATUS",
        "DISEASE_FREE_STATUS", "RECURRENCE_STATUS"
    ])

    if rfs_time_col is not None:
        out["RFS_time"] = pd.to_numeric(clin[rfs_time_col], errors="coerce")
    else:
        out["RFS_time"] = np.nan
    if rfs_status_col is not None:
        out["RFS_event"] = clin[rfs_status_col].map(parse_event_status)
    else:
        out["RFS_event"] = np.nan

    # PFS
    pfs_time_col = find_col(clin, [
        "PFS_MONTHS", "Progress Free Survival (Months)", "Progression Free Survival (Months)",
        "PROGRESSION_FREE_MONTHS", "PROGRESS_FREE_SURVIVAL_MONTHS"
    ])
    pfs_status_col = find_col(clin, [
        "PFS_STATUS", "Progression Free Status", "Progress Free Survival Status",
        "Progression Free Survival Status"
    ])
    out["PFS_time"] = pd.to_numeric(clin[pfs_time_col], errors="coerce") if pfs_time_col is not None else np.nan
    out["PFS_event"] = clin[pfs_status_col].map(parse_event_status) if pfs_status_col is not None else np.nan

    # DSS
    dss_time_col = find_col(clin, [
        "DSS_MONTHS", "Disease-specific Survival (Months)", "Months of disease-specific survival",
        "DISEASE_SPECIFIC_SURVIVAL_MONTHS"
    ])
    dss_status_col = find_col(clin, [
        "DSS_STATUS", "Disease-specific Survival status", "Disease Specific Survival Status",
        "DISEASE_SPECIFIC_SURVIVAL_STATUS"
    ])
    out["DSS_time"] = pd.to_numeric(clin[dss_time_col], errors="coerce") if dss_time_col is not None else np.nan
    out["DSS_event"] = clin[dss_status_col].map(parse_event_status) if dss_status_col is not None else np.nan

    # Stage
    stage_col = find_col(clin, [
        "TUMOR_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATHOLOGIC_STAGE",
        "STAGE", "Tumor Stage", "Neoplasm Disease Stage American Joint Committee on Cancer Code",
        "CLINICAL_STAGE", "FIGO Stage", "Tumor Stage-Pathological"
    ])
    out["stage_group"] = clin[stage_col].map(parse_stage_group) if stage_col is not None else None

    # Useful context columns.
    for label, patterns in {
        "cancer_type": ["CANCER_TYPE", "Cancer Type"],
        "cancer_type_detailed": ["CANCER_TYPE_DETAILED", "Cancer Type Detailed", "HISTOLOGY", "DISEASE_TYPE"],
        "subtype": ["SUBTYPE", "PAM50", "CLAUDIN_SUBTYPE", "Molecular Subtype"],
        "er_status": ["ER_STATUS", "ER Status"],
        "pr_status": ["PR_STATUS", "PR Status"],
        "her2_status": ["HER2_STATUS", "HER2 Status"],
    }.items():
        c = find_col(clin, patterns)
        out[label] = clin[c].astype(str) if c is not None else np.nan

    out.loc[out["OS_time"] <= 0, "OS_time"] = np.nan
    out.loc[out["RFS_time"] <= 0, "RFS_time"] = np.nan
    if "PFS_time" in out.columns:
        out.loc[out["PFS_time"] <= 0, "PFS_time"] = np.nan
    if "DSS_time" in out.columns:
        out.loc[out["DSS_time"] <= 0, "DSS_time"] = np.nan
    out["OS_event"] = pd.to_numeric(out["OS_event"], errors="coerce")
    out["RFS_event"] = pd.to_numeric(out["RFS_event"], errors="coerce")
    if "PFS_event" in out.columns:
        out["PFS_event"] = pd.to_numeric(out["PFS_event"], errors="coerce")
    if "DSS_event" in out.columns:
        out["DSS_event"] = pd.to_numeric(out["DSS_event"], errors="coerce")

    return out, clin


# =============================================================================
# 5. SCORE CONSTRUCTION AND TESTS
# =============================================================================

def construct_bp_scores_for_layer(
    mat_gene_by_sample: pd.DataFrame,
    gene_sets: Dict[str, Dict[str, Any]],
    layer: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mat_gene_by_sample.empty:
        return pd.DataFrame(), pd.DataFrame()

    mat = mat_gene_by_sample.copy()
    mat.index = mat.index.astype(str).str.upper()

    if layer in ["GE", "CN"]:
        zmat = zscore_df_by_rows(mat)
    else:
        zmat = mat.astype(float)

    measured = set(zmat.index)
    scores = {}
    rows = []

    for bp_key, rec in gene_sets.items():
        genes = rec["genes"]
        matched = sorted(set(genes).intersection(measured))
        matched_n = len(matched)

        rows.append({
            "bp_key": bp_key,
            "bp": rec["bp"],
            "layer": layer,
            "annotated_n": rec["annotated_n"],
            "external_matched_n": matched_n,
            "external_matched_fraction": matched_n / rec["annotated_n"] if rec["annotated_n"] else np.nan,
        })

        if matched_n < 1:
            continue

        raw = zmat.loc[matched].mean(axis=0, skipna=True)
        scores[bp_key] = zscore_series(raw)

    if not scores:
        return pd.DataFrame(), pd.DataFrame(rows)

    score_df = pd.DataFrame(scores)
    score_df.index.name = "sample"
    return score_df, pd.DataFrame(rows)


def construct_representation_scores(layer_scores: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    rep_scores = {}
    for rep, layers in REPRESENTATIONS.items():
        if not all(layer in layer_scores and not layer_scores[layer].empty for layer in layers):
            continue

        common_samples = None
        common_bps = None
        for layer in layers:
            df = layer_scores[layer]
            common_samples = set(df.index) if common_samples is None else common_samples.intersection(df.index)
            common_bps = set(df.columns) if common_bps is None else common_bps.intersection(df.columns)

        common_samples = sorted(common_samples)
        common_bps = sorted(common_bps)

        if len(common_samples) < 20 or len(common_bps) == 0:
            continue

        avg = None
        for layer in layers:
            part = layer_scores[layer].loc[common_samples, common_bps]
            avg = part.copy() if avg is None else avg + part
        avg = avg / len(layers)
        avg = avg.apply(zscore_series, axis=0)
        avg.index.name = "sample"
        rep_scores[rep] = avg

    return rep_scores


def logrank_test_basic(time_arr: np.ndarray, event_arr: np.ndarray, group_arr: np.ndarray) -> Tuple[float, float]:
    df = pd.DataFrame({"time": time_arr, "event": event_arr, "group": group_arr}).dropna()
    df = df[df["time"] > 0]
    if df.shape[0] < 20 or df["group"].nunique() != 2 or df["event"].sum() < 3:
        return np.nan, np.nan

    groups = sorted(df["group"].unique())
    g1 = groups[1]
    event_times = np.sort(df.loc[df["event"] == 1, "time"].unique())
    O1 = E1 = V1 = 0.0

    for t in event_times:
        at_risk = df["time"] >= t
        events_t = (df["time"] == t) & (df["event"] == 1)
        n = at_risk.sum()
        if n <= 1:
            continue
        n1 = (at_risk & (df["group"] == g1)).sum()
        d = events_t.sum()
        d1 = (events_t & (df["group"] == g1)).sum()
        if d <= 0:
            continue
        e1 = d * n1 / n
        v1 = (n1 / n) * (1 - n1 / n) * d * (n - d) / max(n - 1, 1)
        O1 += d1
        E1 += e1
        V1 += v1

    if V1 <= 0:
        return np.nan, np.nan
    chi = (O1 - E1) ** 2 / V1
    p = 1 - stats.chi2.cdf(chi, df=1)
    return chi, p


def survival_test(score: pd.Series, clinical: pd.DataFrame, endpoint: str) -> Dict[str, Any]:
    if endpoint == "OS":
        time_col, event_col = "OS_time", "OS_event"
    elif endpoint == "RFS":
        time_col, event_col = "RFS_time", "RFS_event"
    elif endpoint == "PFS":
        time_col, event_col = "PFS_time", "PFS_event"
    elif endpoint == "DSS":
        time_col, event_col = "DSS_time", "DSS_event"
    else:
        raise ValueError(endpoint)

    common = sorted(set(score.dropna().index).intersection(clinical.index))
    if len(common) < 30:
        return {"p": np.nan, "D": np.nan, "n": len(common), "events": np.nan, "HR": np.nan, "C_index": np.nan}

    df = pd.DataFrame({
        "score": score.loc[common],
        "time": clinical.loc[common, time_col],
        "event": clinical.loc[common, event_col],
    }).dropna()
    df = df[df["time"] > 0]

    if df.shape[0] < 30 or df["event"].sum() < 5:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "events": df["event"].sum() if df.shape[0] else np.nan, "HR": np.nan, "C_index": np.nan}

    med = df["score"].median()
    df["group"] = (df["score"] >= med).astype(int)

    _, p = logrank_test_basic(df["time"].values, df["event"].values, df["group"].values)
    D = neglog10_p(p)

    hr = np.nan
    cidx = np.nan
    if HAS_LIFELINES:
        try:
            cph = CoxPHFitter()
            cph.fit(df[["time", "event", "group"]], duration_col="time", event_col="event")
            hr = float(np.exp(cph.params_["group"]))
        except Exception:
            pass
        try:
            cidx = float(concordance_index(df["time"], df["score"], df["event"]))
        except Exception:
            pass

    return {"p": p, "D": D, "n": int(df.shape[0]), "events": int(df["event"].sum()), "HR": hr, "C_index": cidx}


def stage_test(score: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    common = sorted(set(score.dropna().index).intersection(clinical.index))
    if len(common) < 30:
        return {"p": np.nan, "D": np.nan, "n": len(common), "OR": np.nan, "AUC": np.nan}

    df = pd.DataFrame({
        "score": score.loc[common],
        "stage": clinical.loc[common, "stage_group"],
    }).dropna()
    df = df[df["stage"].isin(["EARLY", "ADVANCED"])]

    if df.shape[0] < 30 or df["stage"].nunique() != 2:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan}

    med = df["score"].median()
    df["group"] = (df["score"] >= med).astype(int)
    df["advanced"] = (df["stage"] == "ADVANCED").astype(int)

    tab = pd.crosstab(df["group"], df["advanced"])
    if tab.shape != (2, 2):
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan}

    try:
        if (tab.values < 5).any():
            OR, p = fisher_exact(tab.values)
        else:
            _, p, _, _ = chi2_contingency(tab.values)
            a, b = tab.iloc[1, 1], tab.iloc[1, 0]
            c, d = tab.iloc[0, 1], tab.iloc[0, 0]
            OR = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    except Exception:
        p = np.nan
        OR = np.nan

    D = neglog10_p(p)

    AUC = np.nan
    try:
        pos = df.loc[df["advanced"] == 1, "score"]
        neg = df.loc[df["advanced"] == 0, "score"]
        U, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        AUC = float(U / (len(pos) * len(neg)))
        AUC = max(AUC, 1 - AUC)
    except Exception:
        pass

    return {"p": p, "D": D, "n": int(df.shape[0]), "OR": OR, "AUC": AUC}


def evaluate_score(score: pd.Series, clinical: pd.DataFrame, endpoint: str) -> Dict[str, Any]:
    if endpoint in ["OS", "RFS", "PFS", "DSS"]:
        return survival_test(score, clinical, endpoint)
    if endpoint == "STAGE":
        return stage_test(score, clinical)
    raise ValueError(endpoint)


def get_direction_value(res: Dict[str, Any], endpoint: str) -> float:
    if endpoint in ["OS", "RFS", "PFS", "DSS"]:
        return res.get("HR", np.nan)
    if endpoint == "STAGE":
        return res.get("OR", np.nan)
    return np.nan


def direction_consistent(tcga_value: Any, external_value: Any) -> Optional[int]:
    if pd.isna(tcga_value) or pd.isna(external_value):
        return None
    try:
        tcga_value = float(tcga_value)
        external_value = float(external_value)
    except Exception:
        return None
    if tcga_value == 1 or external_value == 1:
        return None
    return int((tcga_value > 1 and external_value > 1) or (tcga_value < 1 and external_value < 1))


# =============================================================================
# 6. TCGA SELECTED RECORDS
# =============================================================================

def load_tcga_selected(tcga_v3_dir: Path) -> pd.DataFrame:
    path = tcga_v3_dir / "aggregate_tables" / "all_selected_representations.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["bp_key"] = df["bp_key"].map(normalize_bp_key)
    return df


def filter_tcga_records(selected: pd.DataFrame, source_cohorts: List[str]) -> pd.DataFrame:
    df = selected.copy()
    df = df[df["cohort"].isin(source_cohorts)]
    if "readiness_K" in df.columns:
        df = df[df["readiness_K"] == PRIMARY_READINESS_K]
    if "collection" in df.columns:
        df = df[df["collection"] == "HALLMARK"]
    if "endpoint_informative" in df.columns:
        df = df[df["endpoint_informative"] == 1]
    if USE_Q_FILTER and "best_q_by_endpoint" in df.columns:
        df = df[pd.to_numeric(df["best_q_by_endpoint"], errors="coerce") <= Q_THRESHOLD]

    if MAX_TCGA_RECORDS_PER_COHORT_ENDPOINT is not None and not df.empty:
        keep = []
        for (cohort, endpoint), sub in df.groupby(["cohort", "endpoint"]):
            sub = sub.sort_values("best_D", ascending=False).head(MAX_TCGA_RECORDS_PER_COHORT_ENDPOINT)
            keep.append(sub)
        df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

    return df.sort_values(["cohort", "endpoint", "best_D"], ascending=[True, True, False])



def resolve_first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def apply_metabric_clinical_override(clinical: pd.DataFrame, override_path: Optional[Path]) -> pd.DataFrame:
    """
    Apply METABRIC clinical TSV exported from cBioPortal Clinical Data tab.

    Expected useful columns:
    - Sample ID
    - Patient ID
    - Overall Survival (Months)
    - Overall Survival Status
    - Relapse Free Status (Months)
    - Relapse Free Status
    - Tumor Stage
    - PAM50 / ER / PR / HER2 context columns

    This function updates missing OS/RFS/STAGE fields and preserves existing parsed values.
    """
    if clinical.empty:
        log("METABRIC clinical override skipped: clinical table is empty.")
        return clinical
    if override_path is None or not override_path.exists():
        log("METABRIC clinical override skipped: override TSV not found.")
        return clinical
    log(f"METABRIC clinical override found: {override_path}")

    try:
        meta = pd.read_csv(override_path, sep="\t", encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            meta = pd.read_csv(override_path, sep="\t", encoding="utf-8", low_memory=False)
        except Exception as e:
            log(f"METABRIC clinical override could not be read: {override_path} | {e}")
            return clinical

    sample_col = find_col(meta, ["Sample ID", "SAMPLE_ID", "sample"])
    patient_col = find_col(meta, ["Patient ID", "PATIENT_ID", "patient"])
    if sample_col is None:
        log("METABRIC clinical override skipped: no Sample ID column.")
        return clinical

    meta = meta.copy()
    meta["sample"] = meta[sample_col].astype(str).str.strip()
    meta["patient_override"] = meta[patient_col].astype(str).str.strip() if patient_col is not None else meta["sample"]
    meta = meta.drop_duplicates("sample", keep="first").set_index("sample", drop=False)

    out = clinical.copy()

    common = sorted(set(out.index).intersection(meta.index))
    if len(common) == 0:
        # Try patient matching if sample ids do not overlap.
        if "patient" in out.columns:
            meta_by_patient = meta.drop_duplicates("patient_override", keep="first").set_index("patient_override", drop=False)
            common_pat = sorted(set(out["patient"].astype(str)).intersection(meta_by_patient.index))
            if len(common_pat) > 0:
                # Convert to sample-index-aligned table.
                rows = []
                for sample_id, patient_id in out["patient"].astype(str).items():
                    if patient_id in meta_by_patient.index:
                        r = meta_by_patient.loc[patient_id].copy()
                        r.name = sample_id
                        rows.append(r)
                if rows:
                    meta = pd.DataFrame(rows)
                    common = sorted(set(out.index).intersection(meta.index))
        if len(common) == 0:
            log("METABRIC clinical override found no overlapping sample/patient IDs.")
            return clinical

    # OS
    os_time_col = find_col(meta, ["Overall Survival (Months)", "OS_MONTHS", "OS.time", "OS_TIME"])
    os_status_col = find_col(meta, ["Overall Survival Status", "OS_STATUS", "VITAL_STATUS"])
    if os_time_col is not None:
        vals = pd.to_numeric(meta.loc[common, os_time_col], errors="coerce")
        out.loc[common, "OS_time"] = out.loc[common, "OS_time"].combine_first(vals)
    if os_status_col is not None:
        vals = meta.loc[common, os_status_col].map(parse_event_status)
        out.loc[common, "OS_event"] = out.loc[common, "OS_event"].combine_first(vals)

    # RFS
    rfs_time_col = find_col(meta, ["Relapse Free Status (Months)", "RFS_MONTHS", "DFS_MONTHS", "PFS_MONTHS"])
    rfs_status_col = find_col(meta, ["Relapse Free Status", "RFS_STATUS", "DFS_STATUS", "PFS_STATUS"])
    if rfs_time_col is not None:
        vals = pd.to_numeric(meta.loc[common, rfs_time_col], errors="coerce")
        out.loc[common, "RFS_time"] = out.loc[common, "RFS_time"].combine_first(vals)
    if rfs_status_col is not None:
        vals = meta.loc[common, rfs_status_col].map(parse_event_status)
        out.loc[common, "RFS_event"] = out.loc[common, "RFS_event"].combine_first(vals)

    # Stage override. Here we overwrite missing/failed parsing.
    stage_col = find_col(meta, ["Tumor Stage", "TUMOR_STAGE", "STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATHOLOGIC_STAGE"])
    if stage_col is not None:
        raw_counts = meta.loc[common, stage_col].value_counts(dropna=False).head(10).to_dict()
        log(f"METABRIC Tumor Stage raw counts, top 10: {raw_counts}")
        parsed_stage = meta.loc[common, stage_col].map(parse_stage_group)
        before = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum() if "stage_group" in out.columns else 0
        if "stage_group" not in out.columns:
            out["stage_group"] = None
        # Use the cBioPortal clinical export as the authoritative METABRIC stage source.
        stage_mask = parsed_stage.isin(["EARLY", "ADVANCED"])
        out.loc[parsed_stage.index[stage_mask], "stage_group"] = parsed_stage.loc[stage_mask]
        after = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum()
        early = int((out["stage_group"] == "EARLY").sum())
        advanced = int((out["stage_group"] == "ADVANCED").sum())
        log(f"METABRIC clinical override applied: stage available {before} -> {after}; EARLY={early}, ADVANCED={advanced}")

    # Context columns
    context_map = {
        "subtype": ["Pam50 + Claudin-low subtype", "PAM50"],
        "er_status": ["ER Status", "ER_STATUS"],
        "pr_status": ["PR Status", "PR_STATUS"],
        "her2_status": ["HER2 Status", "HER2_STATUS"],
        "cancer_type_detailed": ["Cancer Type Detailed", "CANCER_TYPE_DETAILED"],
    }
    for out_col, pats in context_map.items():
        c = find_col(meta, pats)
        if c is not None:
            if out_col not in out.columns:
                out[out_col] = np.nan
            vals = meta.loc[common, c].astype(str)
            out.loc[common, out_col] = out.loc[common, out_col].combine_first(vals)

    out.loc[out["OS_time"] <= 0, "OS_time"] = np.nan
    out.loc[out["RFS_time"] <= 0, "RFS_time"] = np.nan
    if "PFS_time" in out.columns:
        out.loc[out["PFS_time"] <= 0, "PFS_time"] = np.nan
    if "DSS_time" in out.columns:
        out.loc[out["DSS_time"] <= 0, "DSS_time"] = np.nan
    out["OS_event"] = pd.to_numeric(out["OS_event"], errors="coerce")
    out["RFS_event"] = pd.to_numeric(out["RFS_event"], errors="coerce")
    if "PFS_event" in out.columns:
        out["PFS_event"] = pd.to_numeric(out["PFS_event"], errors="coerce")
    if "DSS_event" in out.columns:
        out["DSS_event"] = pd.to_numeric(out["DSS_event"], errors="coerce")
    return out



def apply_stage_only_clinical_override(
    clinical: pd.DataFrame,
    override_path: Optional[Path],
    dataset_id: str,
    stage_patterns: List[str],
    context_patterns: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Apply a cBioPortal Clinical Data tab export for datasets where the staging
    file omits or misnames stage fields.

    This function mainly repairs STAGE and selected context columns.
    """
    if clinical.empty:
        log(f"{dataset_id} clinical override skipped: clinical table is empty.")
        return clinical
    if override_path is None or not override_path.exists():
        log(f"{dataset_id} clinical override skipped: override TSV not found.")
        return clinical

    log(f"{dataset_id} clinical override found: {override_path}")
    try:
        meta = pd.read_csv(override_path, sep="\t", encoding="utf-8-sig", low_memory=False)
    except Exception:
        meta = pd.read_csv(override_path, sep="\t", encoding="utf-8", low_memory=False)

    sample_col = find_col(meta, ["Sample ID", "SAMPLE_ID", "sample"])
    patient_col = find_col(meta, ["Patient ID", "PATIENT_ID", "patient"])
    if sample_col is None:
        log(f"{dataset_id} clinical override skipped: no Sample ID column.")
        return clinical

    meta = meta.copy()
    meta["sample"] = meta[sample_col].astype(str).str.strip()
    meta["patient_override"] = meta[patient_col].astype(str).str.strip() if patient_col is not None else meta["sample"]
    meta = meta.drop_duplicates("sample", keep="first").set_index("sample", drop=False)

    out = clinical.copy()
    common = sorted(set(out.index).intersection(meta.index))

    if len(common) == 0 and "patient" in out.columns:
        meta_by_patient = meta.drop_duplicates("patient_override", keep="first").set_index("patient_override", drop=False)
        rows = []
        for sample_id, patient_id in out["patient"].astype(str).items():
            if patient_id in meta_by_patient.index:
                r = meta_by_patient.loc[patient_id].copy()
                r.name = sample_id
                rows.append(r)
        if rows:
            meta = pd.DataFrame(rows)
            common = sorted(set(out.index).intersection(meta.index))

    if len(common) == 0:
        log(f"{dataset_id} clinical override found no overlapping sample/patient IDs.")
        return clinical

    stage_col = find_col(meta, stage_patterns)
    if stage_col is not None:
        raw_counts = meta.loc[common, stage_col].value_counts(dropna=False).head(10).to_dict()
        log(f"{dataset_id} stage raw counts, top 10: {raw_counts}")
        parsed_stage = meta.loc[common, stage_col].map(parse_stage_group)
        before = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum() if "stage_group" in out.columns else 0
        if "stage_group" not in out.columns:
            out["stage_group"] = None
        stage_mask = parsed_stage.isin(["EARLY", "ADVANCED"])
        out.loc[parsed_stage.index[stage_mask], "stage_group"] = parsed_stage.loc[stage_mask]
        after = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum()
        early = int((out["stage_group"] == "EARLY").sum())
        advanced = int((out["stage_group"] == "ADVANCED").sum())
        log(f"{dataset_id} clinical override applied: stage available {before} -> {after}; EARLY={early}, ADVANCED={advanced}")

    if context_patterns is not None:
        for out_col, pats in context_patterns.items():
            c = find_col(meta, pats)
            if c is not None:
                if out_col not in out.columns:
                    out[out_col] = np.nan
                vals = meta.loc[common, c].astype(str)
                out.loc[common, out_col] = out.loc[common, out_col].combine_first(vals)

    return out


def apply_kirp_clinical_override(clinical: pd.DataFrame, override_path: Optional[Path]) -> pd.DataFrame:
    """
    Apply KIRP TCGA PanCancer Atlas clinical export.

    This is a TCGA-format backup/sanity-check dataset, not independent external
    validation. It repairs OS, DFS/RFS, PFS, DSS and STAGE endpoints.
    """
    dataset_id = "KIRP_TCGA_CBIO_BACKUP"
    if clinical.empty:
        log(f"{dataset_id} clinical override skipped: clinical table is empty.")
        return clinical
    if override_path is None or not override_path.exists():
        log(f"{dataset_id} clinical override skipped: override TSV not found.")
        return clinical

    log(f"{dataset_id} clinical override found: {override_path}")
    try:
        meta = pd.read_csv(override_path, sep="\t", encoding="utf-8-sig", low_memory=False)
    except Exception:
        meta = pd.read_csv(override_path, sep="\t", encoding="utf-8", low_memory=False)

    sample_col = find_col(meta, ["Sample ID", "SAMPLE_ID", "sample"])
    patient_col = find_col(meta, ["Patient ID", "PATIENT_ID", "patient"])
    if sample_col is None:
        log(f"{dataset_id} clinical override skipped: no Sample ID column.")
        return clinical

    meta = meta.copy()
    meta["sample"] = meta[sample_col].astype(str).str.strip()
    meta["patient_override"] = meta[patient_col].astype(str).str.strip() if patient_col is not None else meta["sample"]
    meta = meta.drop_duplicates("sample", keep="first").set_index("sample", drop=False)

    out = clinical.copy()
    common = sorted(set(out.index).intersection(meta.index))
    if len(common) == 0 and "patient" in out.columns:
        meta_by_patient = meta.drop_duplicates("patient_override", keep="first").set_index("patient_override", drop=False)
        rows = []
        for sample_id, patient_id in out["patient"].astype(str).items():
            if patient_id in meta_by_patient.index:
                r = meta_by_patient.loc[patient_id].copy()
                r.name = sample_id
                rows.append(r)
        if rows:
            meta = pd.DataFrame(rows)
            common = sorted(set(out.index).intersection(meta.index))

    if len(common) == 0:
        log(f"{dataset_id} clinical override found no overlapping sample/patient IDs.")
        return clinical

    def put_surv(prefix: str, time_patterns: List[str], status_patterns: List[str]):
        time_col = find_col(meta, time_patterns)
        status_col = find_col(meta, status_patterns)
        if f"{prefix}_time" not in out.columns:
            out[f"{prefix}_time"] = np.nan
        if f"{prefix}_event" not in out.columns:
            out[f"{prefix}_event"] = np.nan
        if time_col is not None:
            out.loc[common, f"{prefix}_time"] = pd.to_numeric(meta.loc[common, time_col], errors="coerce")
        if status_col is not None:
            out.loc[common, f"{prefix}_event"] = meta.loc[common, status_col].map(parse_event_status)
        n_both = out.loc[common, [f"{prefix}_time", f"{prefix}_event"]].dropna().shape[0]
        n_events = int(pd.to_numeric(out.loc[common, f"{prefix}_event"], errors="coerce").fillna(0).sum())
        log(f"{dataset_id} {prefix} override: n_both={n_both}, events={n_events}")

    put_surv(
        "OS",
        ["Overall Survival (Months)", "OS_MONTHS"],
        ["Overall Survival Status", "OS_STATUS"]
    )
    put_surv(
        "RFS",
        ["Disease Free (Months)", "DFS_MONTHS", "Disease Free Survival (Months)"],
        ["Disease Free Status", "DFS_STATUS", "Disease Free Survival Status"]
    )
    put_surv(
        "PFS",
        ["Progress Free Survival (Months)", "Progression Free Survival (Months)", "PFS_MONTHS"],
        ["Progression Free Status", "Progress Free Survival Status", "PFS_STATUS"]
    )
    put_surv(
        "DSS",
        ["Months of disease-specific survival", "Disease-specific Survival (Months)", "DSS_MONTHS"],
        ["Disease-specific Survival status", "Disease-specific Survival Status", "DSS_STATUS"]
    )

    stage_col = find_col(meta, [
        "Neoplasm Disease Stage American Joint Committee on Cancer Code",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "Tumor Stage",
        "Stage"
    ])
    if stage_col is not None:
        raw_counts = meta.loc[common, stage_col].value_counts(dropna=False).head(10).to_dict()
        log(f"{dataset_id} stage raw counts, top 10: {raw_counts}")
        parsed_stage = meta.loc[common, stage_col].map(parse_stage_group)
        before = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum() if "stage_group" in out.columns else 0
        if "stage_group" not in out.columns:
            out["stage_group"] = None
        stage_mask = parsed_stage.isin(["EARLY", "ADVANCED"])
        out.loc[parsed_stage.index[stage_mask], "stage_group"] = parsed_stage.loc[stage_mask]
        after = out["stage_group"].isin(["EARLY", "ADVANCED"]).sum()
        early = int((out["stage_group"] == "EARLY").sum())
        advanced = int((out["stage_group"] == "ADVANCED").sum())
        log(f"{dataset_id} stage override applied: {before} -> {after}; EARLY={early}, ADVANCED={advanced}")

    context_map = {
        "subtype": ["Subtype", "Molecular Subtype"],
        "mutation_count": ["Mutation Count"],
        "fraction_genome_altered": ["Fraction Genome Altered"],
        "msi_mantis": ["MSI MANTIS Score"],
        "msisensor": ["MSIsensor Score"],
    }
    for out_col, pats in context_map.items():
        c = find_col(meta, pats)
        if c is not None:
            out[out_col] = meta.loc[common, c].astype(str)

    for ep in ["OS", "RFS", "PFS", "DSS"]:
        if f"{ep}_time" in out.columns:
            out.loc[out[f"{ep}_time"] <= 0, f"{ep}_time"] = np.nan
        if f"{ep}_event" in out.columns:
            out[f"{ep}_event"] = pd.to_numeric(out[f"{ep}_event"], errors="coerce")

    return out


# =============================================================================
# 7. EXTERNAL DATASET PROCESSING
# =============================================================================

def load_external_dataset(ds: Dict[str, Any], gene_sets: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    folder = ds["dir"]
    info = {
        "dataset_id": ds["dataset_id"],
        "label": ds["label"],
        "dir": str(folder),
        "exists": folder.exists(),
    }

    if not folder.exists():
        return {}, {}, pd.DataFrame(), pd.DataFrame(), info

    expr_path = find_existing_file(folder, ds["expression_candidates"])
    cna_path = find_existing_file(folder, ds["cna_candidates"])
    mut_path = find_existing_file(folder, ds["mutation_candidates"])
    patient_path = find_existing_file(folder, ds["clinical_patient_candidates"])
    sample_path = find_existing_file(folder, ds["clinical_sample_candidates"])

    info.update({
        "expression_file": str(expr_path) if expr_path else None,
        "cna_file": str(cna_path) if cna_path else None,
        "mutation_file": str(mut_path) if mut_path else None,
        "clinical_patient_file": str(patient_path) if patient_path else None,
        "clinical_sample_file": str(sample_path) if sample_path else None,
    })

    clinical, raw_clinical = load_external_clinical(patient_path, sample_path)

    if ds.get("dataset_id") == "METABRIC_BRCA":
        metabric_override_path = resolve_first_existing_path(
            METABRIC_CLINICAL_OVERRIDE_CANDIDATES + [ds["dir"] / "brca_metabric_clinical_data.tsv"]
        )
        clinical = apply_metabric_clinical_override(clinical, metabric_override_path)

    if ds.get("dataset_id") == "CPTAC_UCEC":
        ucec_override_path = resolve_first_existing_path(
            UCEC_CLINICAL_OVERRIDE_CANDIDATES + [ds["dir"] / "ucec_cptac_2020_clinical_data.tsv"]
        )
        clinical = apply_stage_only_clinical_override(
            clinical,
            ucec_override_path,
            "CPTAC_UCEC",
            ["Tumor Stage-Pathological", "FIGO Stage", "Stage", "Tumor Stage"],
            context_patterns={
                "subtype": ["Genomics Subtype", "MSI Status", "POLE Subtype"],
                "cnv_class": ["CNV class"],
                "p53": ["p53"],
                "tmb": ["TMB"],
                "mutation_count": ["Mutation Count"],
            },
        )

    if ds.get("dataset_id") == "CPTAC_LUAD":
        luad_override_path = resolve_first_existing_path(
            LUAD_CLINICAL_OVERRIDE_CANDIDATES + [ds["dir"] / "luad_cptac_2020_clinical_data.tsv"]
        )
        clinical = apply_stage_only_clinical_override(
            clinical,
            luad_override_path,
            "CPTAC_LUAD",
            ["Stage", "Tumor Stage", "Pathologic Stage"],
            context_patterns={
                "subtype": ["mRNA Expression Subtype TCGA"],
                "smoking_status": ["Smoking Status"],
                "egfr_status": ["EGFR Mutation Status"],
                "kras_status": ["KRAS Mutation Status"],
                "tp53_status": ["TP53 Mutation Status"],
                "stk11_status": ["STK11 Mutation Status"],
                "keap1_status": ["KEAP1 Mutation Status"],
                "tmb": ["TMB"],
                "mutation_count": ["Mutation Count"],
            },
        )

    if ds.get("dataset_id") == "KIRP_TCGA_CBIO_BACKUP":
        kirp_override_path = resolve_first_existing_path(
            KIRP_CLINICAL_OVERRIDE_CANDIDATES + [ds["dir"] / "kirp_tcga_pan_can_atlas_2018_clinical_data.tsv"]
        )
        clinical = apply_kirp_clinical_override(clinical, kirp_override_path)

    info["n_clinical_samples"] = int(clinical.shape[0]) if not clinical.empty else 0
    info["n_OS_available"] = int(clinical[["OS_time", "OS_event"]].dropna().shape[0]) if not clinical.empty else 0
    info["n_OS_events"] = int(pd.to_numeric(clinical.get("OS_event", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not clinical.empty else 0
    info["n_RFS_available"] = int(clinical[["RFS_time", "RFS_event"]].dropna().shape[0]) if not clinical.empty and "RFS_time" in clinical.columns else 0
    info["n_RFS_events"] = int(pd.to_numeric(clinical.get("RFS_event", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not clinical.empty else 0
    info["n_PFS_available"] = int(clinical[["PFS_time", "PFS_event"]].dropna().shape[0]) if not clinical.empty and "PFS_time" in clinical.columns else 0
    info["n_PFS_events"] = int(pd.to_numeric(clinical.get("PFS_event", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not clinical.empty else 0
    info["n_DSS_available"] = int(clinical[["DSS_time", "DSS_event"]].dropna().shape[0]) if not clinical.empty and "DSS_time" in clinical.columns else 0
    info["n_DSS_events"] = int(pd.to_numeric(clinical.get("DSS_event", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not clinical.empty else 0
    info["n_STAGE_available"] = int(clinical.get("stage_group", pd.Series(dtype=object)).isin(["EARLY", "ADVANCED"]).sum()) if not clinical.empty else 0

    mats = {}
    if expr_path:
        mats["GE"] = load_cbio_matrix(expr_path, "GE")
    if cna_path:
        mats["CN"] = load_cbio_matrix(cna_path, "CN")
    if mut_path:
        mats["MU"] = load_cbio_mutation_binary(mut_path)

    # Harmonize sample IDs against clinical index where possible.
    for layer in list(mats.keys()):
        mat = mats[layer]
        common = sorted(set(mat.columns).intersection(clinical.index))
        if len(common) < 20:
            # Try patient-level matching via clinical patient ids.
            sample_to_patient = clinical["patient"].to_dict() if "patient" in clinical.columns else {}
            renamed = {}
            for col in mat.columns:
                if col in clinical.index:
                    renamed[col] = col
                elif col in set(clinical.get("patient", pd.Series(dtype=str)).astype(str)):
                    # Find first sample with this patient.
                    matches = clinical.index[clinical["patient"].astype(str) == str(col)].tolist()
                    if matches:
                        renamed[col] = matches[0]
                    else:
                        renamed[col] = col
                else:
                    renamed[col] = col
            mat = mat.rename(columns=renamed)
            mat = mat.T.groupby(level=0).mean().T
            common = sorted(set(mat.columns).intersection(clinical.index))
        mats[layer] = mat
        info[f"n_{layer}_genes"] = int(mat.shape[0])
        info[f"n_{layer}_samples"] = int(mat.shape[1])
        info[f"n_{layer}_clinical_overlap"] = int(len(common))

    layer_scores = {}
    readiness_rows = []

    for layer, mat in mats.items():
        score_df, ready_df = construct_bp_scores_for_layer(mat, gene_sets, layer)
        layer_scores[layer] = score_df
        ready_df["dataset_id"] = ds["dataset_id"]
        readiness_rows.append(ready_df)

    readiness = pd.concat(readiness_rows, ignore_index=True) if readiness_rows else pd.DataFrame()
    rep_scores = construct_representation_scores(layer_scores)

    for rep, df in rep_scores.items():
        info[f"n_{rep}_bp_scores"] = int(df.shape[1])
        info[f"n_{rep}_samples"] = int(df.shape[0])

    return layer_scores, rep_scores, clinical, readiness, info


def validate_dataset(
    ds: Dict[str, Any],
    tcga_selected: pd.DataFrame,
    gene_sets: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log("=" * 90)
    log(f"External dataset: {ds['dataset_id']} | {ds['label']}")

    dataset_out = out_dir / "dataset_outputs" / ds["dataset_id"]
    ensure_dir(dataset_out)

    layer_scores, rep_scores, clinical, readiness, info = load_external_dataset(ds, gene_sets)
    write_json(info, dataset_out / f"{ds['dataset_id']}_data_availability.json")
    if not readiness.empty:
        write_csv(readiness, dataset_out / f"{ds['dataset_id']}_external_readiness.csv")
    if not clinical.empty:
        write_csv(clinical.reset_index(drop=True), dataset_out / f"{ds['dataset_id']}_clinical_parsed.csv")

    records = filter_tcga_records(tcga_selected, ds["source_cohorts"])
    write_csv(records, dataset_out / f"{ds['dataset_id']}_tcga_records_sent_to_external_validation.csv")

    if records.empty or not rep_scores or clinical.empty:
        log(f"{ds['dataset_id']}: no records or no external data.")
        return pd.DataFrame(), readiness, info

    rows = []

    for _, rec in records.iterrows():
        bp_key = normalize_bp_key(rec["bp_key"])
        tcga_endpoint = rec["endpoint"]
        best_rep = rec["best_representation"]

        # Endpoint mapping:
        # TCGA OS records are tested on external OS and RFS when available.
        # TCGA STAGE records are tested on external STAGE.
        if tcga_endpoint == "OS":
            external_endpoints = [e for e in ["OS", "RFS", "PFS", "DSS"] if e in ds["external_endpoints"]]
        elif tcga_endpoint == "STAGE":
            external_endpoints = ["STAGE"] if "STAGE" in ds["external_endpoints"] else []
        else:
            external_endpoints = []

        for ext_endpoint in external_endpoints:
            d_by_rep = {}
            p_by_rep = {}
            dir_by_rep = {}
            n_by_rep = {}
            event_by_rep = {}

            for rep in REPRESENTATIONS.keys():
                if rep not in rep_scores or bp_key not in rep_scores[rep].columns:
                    d_by_rep[rep] = np.nan
                    p_by_rep[rep] = np.nan
                    dir_by_rep[rep] = np.nan
                    n_by_rep[rep] = np.nan
                    event_by_rep[rep] = np.nan
                    continue

                score = rep_scores[rep][bp_key]
                res = evaluate_score(score, clinical, ext_endpoint)
                d_by_rep[rep] = res.get("D", np.nan)
                p_by_rep[rep] = res.get("p", np.nan)
                dir_by_rep[rep] = get_direction_value(res, ext_endpoint)
                n_by_rep[rep] = res.get("n", np.nan)
                event_by_rep[rep] = res.get("events", np.nan)

            external_same_rep_D = d_by_rep.get(best_rep, np.nan)
            external_same_rep_p = p_by_rep.get(best_rep, np.nan)
            external_same_rep_direction = dir_by_rep.get(best_rep, np.nan)
            external_GE_D = d_by_rep.get("GE", np.nan)

            valid_d = {r: v for r, v in d_by_rep.items() if pd.notna(v)}
            if valid_d:
                external_best_rep = max(valid_d, key=valid_d.get)
                external_best_D = valid_d[external_best_rep]
                external_best_p = p_by_rep.get(external_best_rep, np.nan)
            else:
                external_best_rep = None
                external_best_D = np.nan
                external_best_p = np.nan

            external_delta_same_minus_GE = (
                external_same_rep_D - external_GE_D
                if pd.notna(external_same_rep_D) and pd.notna(external_GE_D)
                else np.nan
            )
            external_delta_best_minus_GE = (
                external_best_D - external_GE_D
                if pd.notna(external_best_D) and pd.notna(external_GE_D)
                else np.nan
            )

            tcga_dir_value = rec.get("HR", np.nan) if tcga_endpoint == "OS" else rec.get("OR", np.nan)
            dir_cons = direction_consistent(tcga_dir_value, external_same_rep_direction)

            rows.append({
                "dataset_id": ds["dataset_id"],
                "dataset_label": ds["label"],
                "external_type": ds["external_type"],
                "source_cohort": rec["cohort"],
                "tcga_endpoint": tcga_endpoint,
                "external_endpoint": ext_endpoint,
                "bp_key": bp_key,
                "bp": rec.get("bp", bp_key),
                "tcga_best_representation": best_rep,
                "external_best_representation": external_best_rep,
                "same_representation_as_tcga": int(external_best_rep == best_rep) if external_best_rep is not None else np.nan,

                "tcga_best_D": rec.get("best_D", np.nan),
                "tcga_best_p": rec.get("best_p", np.nan),
                "tcga_best_q_by_endpoint": rec.get("best_q_by_endpoint", np.nan),
                "tcga_deltaD_best_minus_GE": rec.get("deltaD_best_minus_GE", np.nan),
                "tcga_deltaD_GE_CN_minus_GE": rec.get("deltaD_GE_CN_minus_GE", np.nan),
                "tcga_deltaD_GE_MU_minus_GE": rec.get("deltaD_GE_MU_minus_GE", np.nan),
                "tcga_deltaD_GE_CN_MU_minus_GE": rec.get("deltaD_GE_CN_MU_minus_GE", np.nan),
                "tcga_representation_class": rec.get("representation_class", np.nan),
                "tcga_gain_class": rec.get("gain_class", np.nan),
                "tcga_any_fragile": rec.get("any_fragile", np.nan),
                "tcga_any_strong_fragile": rec.get("any_strong_fragile", np.nan),
                "tcga_any_signal_lost": rec.get("any_signal_lost", np.nan),

                "external_same_rep_D": external_same_rep_D,
                "external_same_rep_p": external_same_rep_p,
                "external_same_rep_direction_value": external_same_rep_direction,
                "external_best_D": external_best_D,
                "external_best_p": external_best_p,
                "external_GE_D": external_GE_D,
                "external_GE_CN_D": d_by_rep.get("GE_CN", np.nan),
                "external_GE_MU_D": d_by_rep.get("GE_MU", np.nan),
                "external_GE_CN_MU_D": d_by_rep.get("GE_CN_MU", np.nan),
                "external_delta_same_minus_GE": external_delta_same_minus_GE,
                "external_delta_best_minus_GE": external_delta_best_minus_GE,

                "external_same_rep_supported_D_ge_1p301": int(pd.notna(external_same_rep_D) and external_same_rep_D >= EXTERNAL_D_SUPPORT),
                "external_best_supported_D_ge_1p301": int(pd.notna(external_best_D) and external_best_D >= EXTERNAL_D_SUPPORT),
                "direction_consistent_same_rep": dir_cons,

                "external_n_same_rep": n_by_rep.get(best_rep, np.nan),
                "external_events_same_rep": event_by_rep.get(best_rep, np.nan),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        write_csv(result, dataset_out / f"{ds['dataset_id']}_external_validation_records.csv")

    return result, readiness, info


# =============================================================================
# 8. SUMMARIES AND FIGURES
# =============================================================================

def summarize_external_results(df: pd.DataFrame, out_dir: Path) -> None:
    agg = out_dir / "aggregate_tables"
    ensure_dir(agg)

    if df.empty:
        return

    summary1 = df.groupby(["dataset_id", "dataset_label", "external_endpoint"]).agg(
        n_records=("bp_key", "count"),
        n_external_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "sum"),
        frac_external_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "mean"),
        n_external_best_supported=("external_best_supported_D_ge_1p301", "sum"),
        frac_external_best_supported=("external_best_supported_D_ge_1p301", "mean"),
        median_tcga_best_D=("tcga_best_D", "median"),
        median_external_same_rep_D=("external_same_rep_D", "median"),
        median_external_best_D=("external_best_D", "median"),
        median_external_delta_same_GE=("external_delta_same_minus_GE", "median"),
        n_direction_evaluable=("direction_consistent_same_rep", lambda x: pd.Series(x).notna().sum()),
        n_direction_consistent=("direction_consistent_same_rep", lambda x: pd.Series(x).dropna().sum()),
    ).reset_index()
    summary1["frac_direction_consistent"] = summary1["n_direction_consistent"] / summary1["n_direction_evaluable"].replace(0, np.nan)
    write_csv(summary1, agg / "external_validation_summary_by_dataset_endpoint.csv")

    summary2 = df.groupby(["dataset_id", "source_cohort", "tcga_endpoint", "external_endpoint"]).agg(
        n_records=("bp_key", "count"),
        n_external_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "sum"),
        frac_external_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "mean"),
        n_external_best_supported=("external_best_supported_D_ge_1p301", "sum"),
        frac_external_best_supported=("external_best_supported_D_ge_1p301", "mean"),
        median_tcga_best_D=("tcga_best_D", "median"),
        median_external_same_rep_D=("external_same_rep_D", "median"),
        median_external_best_D=("external_best_D", "median"),
        median_external_delta_same_GE=("external_delta_same_minus_GE", "median"),
        n_direction_evaluable=("direction_consistent_same_rep", lambda x: pd.Series(x).notna().sum()),
        n_direction_consistent=("direction_consistent_same_rep", lambda x: pd.Series(x).dropna().sum()),
    ).reset_index()
    summary2["frac_direction_consistent"] = summary2["n_direction_consistent"] / summary2["n_direction_evaluable"].replace(0, np.nan)
    write_csv(summary2, agg / "external_validation_summary_by_dataset_sourcecohort_endpoint.csv")

    # Top supported records.
    top = df[
        (df["external_same_rep_supported_D_ge_1p301"] == 1)
        | (df["external_best_supported_D_ge_1p301"] == 1)
    ].copy()
    if not top.empty:
        top = top.sort_values(["external_same_rep_D", "external_best_D", "tcga_best_D"], ascending=False)
        write_csv(top.head(500), agg / "external_validation_top_supported_records.csv")

    # Potentially transported same representation and direction.
    transported = df[
        (df["external_same_rep_supported_D_ge_1p301"] == 1)
        & ((df["direction_consistent_same_rep"] == 1) | df["direction_consistent_same_rep"].isna())
    ].copy()
    if not transported.empty:
        transported = transported.sort_values(["external_same_rep_D", "tcga_best_D"], ascending=False)
        write_csv(transported, agg / "external_validation_same_rep_supported_direction_ok.csv")


def make_external_figures(df: pd.DataFrame, out_dir: Path) -> None:
    if not HAS_MPL or df.empty:
        return

    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    # Scatter by dataset and endpoint.
    for (dataset_id, ext_endpoint), sub in df.groupby(["dataset_id", "external_endpoint"]):
        sub = sub.dropna(subset=["tcga_best_D", "external_same_rep_D"])
        if sub.shape[0] < 3:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(sub["tcga_best_D"], sub["external_same_rep_D"])
        plt.axhline(D_THRESHOLD, linestyle="--")
        plt.axvline(D_THRESHOLD, linestyle="--")
        plt.xlabel("TCGA best D")
        plt.ylabel("External same-representation D")
        plt.title(f"{dataset_id}: TCGA vs external ({ext_endpoint})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"scatter_TCGA_vs_external_same_rep_{safe_name(dataset_id)}_{safe_name(ext_endpoint)}.png", dpi=300)
        plt.close()

    # Bar summary.
    summary = df.groupby(["dataset_id", "external_endpoint"])["external_same_rep_supported_D_ge_1p301"].mean().reset_index()
    if not summary.empty:
        summary["label"] = summary["dataset_id"] + " / " + summary["external_endpoint"]
        plt.figure(figsize=(10, 5))
        plt.bar(summary["label"], summary["external_same_rep_supported_D_ge_1p301"])
        plt.ylabel("Fraction same-representation external D >= 1.301")
        plt.xticks(rotation=45, ha="right")
        plt.title("External transport support by dataset and endpoint")
        plt.tight_layout()
        plt.savefig(fig_dir / "bar_external_transport_support_by_dataset_endpoint.png", dpi=300)
        plt.close()


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    start = time.time()
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "aggregate_tables")
    ensure_dir(OUT_DIR / "dataset_outputs")
    ensure_dir(OUT_DIR / "figures")

    tcga_dir = TCGA_V3_OUT if TCGA_V3_OUT is not None else find_latest_tcga_v3_output(AIDO_TEMP)

    config = {
        "OUT_DIR": str(OUT_DIR),
        "TCGA_V3_OUT": str(tcga_dir),
        "EXTERNAL_ROOT": str(EXTERNAL_ROOT),
        "METABRIC_CLINICAL_OVERRIDE_CANDIDATES": [str(p) for p in METABRIC_CLINICAL_OVERRIDE_CANDIDATES],
        "UCEC_CLINICAL_OVERRIDE_CANDIDATES": [str(p) for p in UCEC_CLINICAL_OVERRIDE_CANDIDATES],
        "LUAD_CLINICAL_OVERRIDE_CANDIDATES": [str(p) for p in LUAD_CLINICAL_OVERRIDE_CANDIDATES],
        "KIRP_CLINICAL_OVERRIDE_CANDIDATES": [str(p) for p in KIRP_CLINICAL_OVERRIDE_CANDIDATES],
        "HALLMARK_GMT": str(HALLMARK_GMT),
        "PRIMARY_READINESS_K": PRIMARY_READINESS_K,
        "D_THRESHOLD": D_THRESHOLD,
        "MAX_TCGA_RECORDS_PER_COHORT_ENDPOINT": MAX_TCGA_RECORDS_PER_COHORT_ENDPOINT,
        "USE_Q_FILTER": USE_Q_FILTER,
        "Q_THRESHOLD": Q_THRESHOLD,
        "HAS_LIFELINES": HAS_LIFELINES,
        "datasets": [
            {
                "dataset_id": d["dataset_id"],
                "dir": str(d["dir"]),
                "source_cohorts": d["source_cohorts"],
                "external_type": d["external_type"],
            }
            for d in EXTERNAL_DATASETS
        ],
    }
    write_json(config, OUT_DIR / "external_validation_run_config.json")

    log("AIDO-Multi-Omics-I-4.0 external validation started")
    log(f"Output: {OUT_DIR}")
    log(f"TCGA V3 output: {tcga_dir}")

    gene_sets = load_hallmark_gmt(HALLMARK_GMT)
    log(f"Loaded Hallmark gene sets: {len(gene_sets)}")

    tcga_selected = load_tcga_selected(tcga_dir)
    log(f"Loaded TCGA selected records: {tcga_selected.shape}")

    all_results = []
    all_readiness = []
    availability = []

    for ds in EXTERNAL_DATASETS:
        try:
            result, readiness, info = validate_dataset(ds, tcga_selected, gene_sets, OUT_DIR)
            availability.append(info)
            if not result.empty:
                all_results.append(result)
            if not readiness.empty:
                all_readiness.append(readiness)
        except Exception as e:
            log(f"ERROR in dataset {ds['dataset_id']}: {e}")
            availability.append({
                "dataset_id": ds["dataset_id"],
                "label": ds["label"],
                "dir": str(ds["dir"]),
                "error": str(e),
            })

    availability_df = pd.DataFrame(availability)
    write_csv(availability_df, OUT_DIR / "aggregate_tables" / "external_validation_data_availability.csv")

    if all_readiness:
        readiness_df = pd.concat(all_readiness, ignore_index=True)
        write_csv(readiness_df, OUT_DIR / "aggregate_tables" / "external_validation_all_readiness.csv")

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        write_csv(result_df, OUT_DIR / "aggregate_tables" / "external_validation_all_records.csv")
        summarize_external_results(result_df, OUT_DIR)
        make_external_figures(result_df, OUT_DIR)
    else:
        result_df = pd.DataFrame()
        log("No external validation records generated.")

    final_report = {
        "output_dir": str(OUT_DIR),
        "tcga_v3_out": str(tcga_dir),
        "elapsed_minutes": (time.time() - start) / 60,
        "n_external_records": int(result_df.shape[0]) if not result_df.empty else 0,
        "n_datasets_attempted": len(EXTERNAL_DATASETS),
    }
    write_json(final_report, OUT_DIR / "external_validation_final_report.json")

    log(f"External validation completed in {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
