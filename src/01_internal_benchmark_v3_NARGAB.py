# -*- coding: utf-8 -*-
"""
AIDO-Multi-Omics-I-4.0
Internal benchmark pipeline v3 NARGAB-UPGRADE

Purpose
-------
This script upgrades the v2 internal benchmark into a stronger V3 pipeline for
AIDO-Multi-Omics-I-4.0. It keeps the useful V2 backbone and adds the key
rigour modules needed for the NARGAB-oriented manuscript:

1. Robust UTF-8 / UTF-16 / UTF-8-SIG / Latin1 input reader.
2. Clinical patient-ID suffix remapping.
3. Representation-specific BP score construction:
   GE, GE+CN, GE+MU, GE+CN+MU.
4. Endpoint discriminability:
   OS and STAGE.
5. Delta-D over GE baseline:
   GE+CN minus GE, GE+MU minus GE, GE+CN+MU minus GE, best minus GE.
6. Per-representation integration fragility:
   CN-fragile, MU-fragile, full-fragile, strong-fragile, signal-lost.
7. FDR-aware sensitivity:
   q-values by endpoint and by endpoint+cohort.
8. TRUE readiness-cutoff sensitivity:
   matched-gene cutoff values are rerun: 5, 10, 15, 20.
9. Size-matched random gene-set baseline:
   real BP scores compared against random gene sets with the same matched-gene count.
10. Optional BP-burden downstream stress test:
   selected representation burden, random BP-burden baseline, repeated split validation.
11. All outputs written to D:/AIDO-Temp/.

Input
-----
D:/AIDO-Data/UCSC_XENA/
D:/AIDO-Data/GSEA/h.all.v2026.1.Hs.symbols.gmt

Optional additional GMTs:
D:/AIDO-Data/GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt
D:/AIDO-Data/GSEA/c2.cp.reactome.v2026.1.Hs.symbols.gmt

Output
------
D:/AIDO-Temp/AIDO_MultiOmics_I_4_internal_benchmark_v3_NARGAB_<timestamp>/

Important interpretation
------------------------
D = -log10(p) is used as a task-discriminability statistic, not as a direct
effect-size estimate. FDR and effect-size-oriented summaries are provided as
sensitivity outputs.

This script does not build a clinical prognostic model. Repeated split analysis
is used to evaluate representation-selection stability and reduce circularity.
"""

from __future__ import annotations

import re
import gc
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
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

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


# =============================================================================
# 0. CONFIG
# =============================================================================

BASE_DIR = Path("D:/AIDO-Data/UCSC_XENA")
GSEA_DIR = Path("D:/AIDO-Data/GSEA")
OUT_ROOT = Path("D:/AIDO-Temp")

GMT_FILES = {
    "HALLMARK": GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt",

    # Optional. If files do not exist, they are skipped automatically.
    # Keep HALLMARK as primary for speed and manuscript main analysis.
    "GOBP": GSEA_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt",
    "REACTOME": GSEA_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
}

PRIMARY_GENESET_COLLECTION = "HALLMARK"

# TRUE observation-readiness matched-gene cutoff sensitivity.
READINESS_K_VALUES = [5, 10, 15, 20]
PRIMARY_READINESS_K = 10

# BP-burden downstream stress-test settings.
BURDEN_TOPK_VALUES = [5, 8, 10, 15]
PRIMARY_BURDEN_TOPK = 8
ABNORMALITY_CUTOFFS = ["SD1", "EXT20"]

D_THRESHOLD = -math.log10(0.05)
STRONG_FRAGILITY_DELTA = -0.3
MODERATE_GAIN_DELTA = 0.5
STRONG_GAIN_DELTA = 1.0

RANDOM_GENESET_ITER = 300
RANDOM_BP_BURDEN_ITER = 300
SPLIT_ITER = 100
TRAIN_FRAC = 0.70

RANDOM_SEED = 20260604

REPRESENTATIONS = {
    "GE": ["GE"],
    "GE_CN": ["GE", "CN"],
    "GE_MU": ["GE", "MU"],
    "GE_CN_MU": ["GE", "CN", "MU"],
}

ENDPOINTS = ["OS", "STAGE"]

DEFAULT_EXCLUDE_CODES = {"COADREAD", "LUNG", "READ"}

# For debugging:
# COHORT_WHITELIST = ["BRCA", "BLCA", "KIRC", "KIRP", "LUAD", "SKCM"]
COHORT_WHITELIST: Optional[List[str]] = None

# These toggles help when debugging memory/time.
RUN_BURDEN_MODULE = True
RUN_RANDOM_GENESET_BASELINE = True
RUN_RANDOM_BP_BURDEN_BASELINE = True
RUN_SPLIT_VALIDATION = True
RUN_FIGURES = True

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / f"AIDO_MultiOmics_I_4_internal_benchmark_v3_NARGAB_{TIMESTAMP}"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


# =============================================================================
# 1. BASIC UTILITIES
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
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")


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


def tcga_patient_id(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace(".", "-")
    s = s.replace("\ufeff", "")
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s)
    if m:
        return m.group(1)
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return s if s else None


def read_table_auto(path: Path, index_col: Optional[int] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-8", "latin1"]
    seps = ["\t", ","]
    best_df = None
    best_score = -10**9
    best_info = None
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path, sep=sep, encoding=enc, index_col=index_col,
                    low_memory=False, nrows=nrows
                )
                if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                    continue

                col_text = " ".join([str(c) for c in df.columns[:30]])
                bad_chars = col_text.count("\x00") + col_text.count("�")
                unnamed = sum(str(c).lower().startswith("unnamed") for c in df.columns)
                score = df.shape[1] * 10 - bad_chars * 100 - unnamed
                if df.shape[1] == 1:
                    score -= 20

                if score > best_score:
                    best_df = df
                    best_score = score
                    best_info = (enc, sep, df.shape)
            except Exception as e:
                last_err = e

    if best_df is None:
        raise RuntimeError(f"Cannot read file: {path}. Last error: {last_err}")

    best_df.columns = [str(c).replace("\ufeff", "").strip() for c in best_df.columns]
    log(f"[read_table_auto] {path.name} | encoding={best_info[0]} | sep={repr(best_info[1])} | shape={best_info[2]}")
    return best_df


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
    z = arr.sub(mean, axis=0).div(sd, axis=0)
    return z.replace([np.inf, -np.inf], np.nan)


def bh_fdr(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if mask.sum() == 0:
        return out

    p = pd.to_numeric(pvals.loc[mask], errors="coerce")
    mask2 = p.notna()
    if mask2.sum() == 0:
        return out

    if HAS_STATSMODELS:
        q = multipletests(p.loc[mask2].values, method="fdr_bh")[1]
        out.loc[p.loc[mask2].index] = q
    else:
        # Manual Benjamini-Hochberg fallback.
        pv = p.loc[mask2].values
        order = np.argsort(pv)
        ranked = pv[order]
        m = len(ranked)
        q = ranked * m / (np.arange(1, m + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        q_back = np.empty_like(q)
        q_back[order] = q
        out.loc[p.loc[mask2].index] = q_back

    return out


# =============================================================================
# 2. COHORT DISCOVERY
# =============================================================================

def infer_code_from_folder(folder_name: str) -> Optional[str]:
    m = re.search(r"\(([A-Z0-9]+)\)", folder_name)
    return m.group(1) if m else None


def discover_cohorts(base_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        code = infer_code_from_folder(p.name)
        if code is None:
            continue
        if COHORT_WHITELIST is not None and code not in COHORT_WHITELIST:
            continue
        rows.append({"cohort": code, "folder_name": p.name, "path": str(p)})
    return pd.DataFrame(rows)


def find_file_case_insensitive(folder: Path, candidates: List[str]) -> Optional[Path]:
    files = {x.name.lower(): x for x in folder.iterdir() if x.is_file()}
    for c in candidates:
        if c.lower() in files:
            return files[c.lower()]
    return None


def get_cohort_files(folder: Path, cohort: str) -> Dict[str, Optional[Path]]:
    out = {}
    out["GE"] = find_file_case_insensitive(folder, ["GE.tsv", "ge.tsv"])
    out["CN"] = find_file_case_insensitive(folder, ["CN.tsv", "cn.tsv"])
    out["MU"] = find_file_case_insensitive(folder, ["MU_fixed.tsv", "MU.tsv", "mu_fixed.tsv", "mu.tsv"])
    out["PHENO"] = find_file_case_insensitive(folder, ["Phenotype.tsv", "phenotype.tsv"])
    out["STAGE_GROUP"] = find_file_case_insensitive(folder, [f"{cohort}_stage_groups_from_survival.tsv"])

    clinical_candidates = [
        f"TCGA.{cohort}.sampleMap_{cohort}_clinicalMatrix",
        f"TCGA.{cohort}.sampleMap_{cohort}_clinicalMatrix.tsv",
    ]
    out["CLINICAL"] = find_file_case_insensitive(folder, clinical_candidates)

    if out["CLINICAL"] is None:
        for x in folder.iterdir():
            if x.is_file() and "clinicalmatrix" in x.name.lower():
                out["CLINICAL"] = x
                break
    return out


# =============================================================================
# 3. GENE SETS
# =============================================================================

def load_gmt(gmt_path: Path, collection: str) -> Dict[str, Dict[str, Any]]:
    if not gmt_path.exists():
        raise FileNotFoundError(f"GMT not found: {gmt_path}")

    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ""
            genes = sorted(set(g.strip().upper() for g in parts[2:] if g.strip()))
            gene_sets[name] = {
                "bp": name,
                "collection": collection,
                "description": desc,
                "genes": genes,
                "annotated_n": len(genes),
            }
    return gene_sets


def load_gene_set_collections(gmt_files: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    all_sets = {}
    for collection, path in gmt_files.items():
        if not path.exists():
            log(f"Optional GMT skipped: {collection} | {path}")
            continue
        gs = load_gmt(path, collection)
        for bp, rec in gs.items():
            key = f"{collection}::{bp}"
            rec["bp_key"] = key
            all_sets[key] = rec
        log(f"Loaded {collection}: {len(gs)} gene sets")
    if not all_sets:
        raise RuntimeError("No GMT gene sets loaded.")
    return all_sets


def subset_gene_sets(gene_sets: Dict[str, Dict[str, Any]], collection: str) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in gene_sets.items() if v.get("collection") == collection}


# =============================================================================
# 4. MOLECULAR MATRICES
# =============================================================================

def normalize_matrix_gene_by_patient(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    lower_cols = {c.lower(): c for c in df.columns}

    # Long mutation table.
    if layer == "MU" and ("sample" in lower_cols) and ("gene" in lower_cols):
        sample_col = lower_cols["sample"]
        gene_col = lower_cols["gene"]
        tmp = df[[sample_col, gene_col]].dropna()
        tmp["patient"] = tmp[sample_col].map(tcga_patient_id)
        tmp["gene"] = tmp[gene_col].astype(str).str.upper().str.strip()
        tmp = tmp.dropna(subset=["patient", "gene"])
        tmp["value"] = 1.0
        mat = tmp.drop_duplicates(["gene", "patient"]).pivot_table(
            index="gene", columns="patient", values="value", aggfunc="max", fill_value=0.0
        )
        mat.index = mat.index.astype(str).str.upper()
        mat.columns = [tcga_patient_id(c) for c in mat.columns]
        mat = mat.loc[:, [c is not None for c in mat.columns]]
        return mat.astype(float)

    first_col = df.columns[0]
    first_lower = str(first_col).lower()

    if first_lower in ["gene", "genes", "symbol", "hugo_symbol", "id", "name", "sample"]:
        df = df.set_index(first_col)
    else:
        numeric_ratio = pd.to_numeric(df.iloc[:20, 0], errors="coerce").notna().mean()
        if numeric_ratio < 0.5:
            df = df.set_index(first_col)

    drop_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if cl in [
            "description", "gene_id", "entrez", "chrom", "chr", "start", "end",
            "reference", "alt", "effect", "amino_acid_change", "dna_vaf"
        ]:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    new_cols = [tcga_patient_id(c) for c in numeric.columns]
    numeric.columns = new_cols
    numeric = numeric.loc[:, [c is not None and str(c) != "" for c in numeric.columns]]

    numeric = numeric.T.groupby(level=0).mean().T
    numeric.index = numeric.index.astype(str).str.upper().str.strip()
    numeric = numeric[~numeric.index.duplicated(keep="first")]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    if layer == "MU":
        numeric = numeric.fillna(0)
        numeric = (numeric != 0).astype(float)

    return numeric


def load_layer_matrix(path: Path, layer: str) -> pd.DataFrame:
    df = read_table_auto(path)
    mat = normalize_matrix_gene_by_patient(df, layer)
    log(f"Loaded {layer}: {path.name} | genes x patients = {mat.shape}")
    return mat


# =============================================================================
# 5. CLINICAL PARSING + ID REMAPPING
# =============================================================================

def find_first_matching_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for pat in patterns:
        pl = pat.lower()
        for cl, orig in lower_map.items():
            if cl == pl:
                return orig

    for pat in patterns:
        pl = pat.lower()
        for cl, orig in lower_map.items():
            if pl in cl:
                return orig

    return None


def standardize_clinical_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    id_col = find_first_matching_col(
        df,
        ["sample", "patient", "patient_id", "bcr_patient_barcode", "submitter_id", "sampleid", "id"]
    )
    if id_col is None:
        id_col = df.columns[0]

    df["patient_raw"] = df[id_col].astype(str).str.strip()
    df["patient"] = df[id_col].map(tcga_patient_id)
    df = df.dropna(subset=["patient"])
    df = df.drop_duplicates("patient", keep="first")
    df = df.set_index("patient", drop=True)
    return df


def remap_clinical_ids_to_molecular(clinical: pd.DataFrame, molecular_patients: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clinical = clinical.copy()
    molecular_patients = sorted(set(str(x).upper() for x in molecular_patients if pd.notna(x)))

    suffix_map = {}
    suffix_count = {}

    for pid in molecular_patients:
        parts = pid.split("-")
        suffixes = []
        if len(parts) >= 3:
            suffixes.append(parts[-1])
            suffixes.append("-".join(parts[-2:]))
        suffixes.append(pid)
        for suf in suffixes:
            suffix_count[suf] = suffix_count.get(suf, 0) + 1
            suffix_map[suf] = pid

    new_index = []
    rows = []

    for old_id in clinical.index:
        old = str(old_id).upper().strip()
        if old in molecular_patients:
            new = old
            status = "already_full_match"
        elif old in suffix_map and suffix_count.get(old, 0) == 1:
            new = suffix_map[old]
            status = "suffix_remapped"
        else:
            old_clean = old.split("-")[-1]
            if old_clean in suffix_map and suffix_count.get(old_clean, 0) == 1:
                new = suffix_map[old_clean]
                status = "suffix_remapped_cleaned"
            else:
                new = old
                status = "unmapped"

        new_index.append(new)
        rows.append({"old_clinical_id": old, "new_patient_id": new, "mapping_status": status})

    mapping_df = pd.DataFrame(rows)
    clinical.index = new_index
    clinical.index.name = "patient"
    clinical = clinical[~clinical.index.duplicated(keep="first")]
    return clinical, mapping_df


def parse_os_event(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["1", "true", "dead", "deceased", "event", "yes", "death"]:
        return 1
    if s in ["0", "false", "alive", "living", "censored", "no"]:
        return 0
    if s.startswith("1"):
        return 1
    if s.startswith("0"):
        return 0
    if "dead" in s or "deceased" in s:
        return 1
    if "alive" in s or "living" in s:
        return 0
    return None


def parse_stage_group(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in ["EARLY", "I/II", "I-II", "STAGE_EARLY"]:
        return "EARLY"
    if s in ["ADVANCED", "LATE", "III/IV", "III-IV", "STAGE_ADVANCED"]:
        return "ADVANCED"

    s2 = s
    for token in ["STAGE", "PATHOLOGIC", "CLINICAL", "AJCC", "TUMOR"]:
        s2 = s2.replace(token, "")
    s2 = s2.replace(" ", "").replace("_", "")

    if s2.startswith("IV") or re.search(r"\bIV\b", s2):
        return "ADVANCED"
    if s2.startswith("III") or re.search(r"\bIII\b", s2):
        return "ADVANCED"
    if s2.startswith("II") or re.search(r"\bII\b", s2):
        return "EARLY"
    if s2.startswith("I") or re.search(r"\bI\b", s2):
        return "EARLY"
    return None


def load_clinical(files: Dict[str, Optional[Path]], molecular_patients: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    for key in ["PHENO", "CLINICAL", "STAGE_GROUP"]:
        p = files.get(key)
        if p is not None and p.exists():
            try:
                d = standardize_clinical_table(read_table_auto(p))
                d["_source_priority"] = key
                dfs.append(d)
                log(f"Clinical source loaded: {key} | {p.name} | shape={d.shape}")
            except Exception as e:
                log(f"Warning: failed clinical source {key}: {p} | {e}")

    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    clin = dfs[0].copy()
    for d in dfs[1:]:
        clin = clin.combine_first(d)

    if molecular_patients is not None:
        clin, mapping_df = remap_clinical_ids_to_molecular(clin, molecular_patients)
    else:
        mapping_df = pd.DataFrame()

    out = pd.DataFrame(index=clin.index)

    time_col = find_first_matching_col(
        clin,
        ["OS.time", "OS_time", "OS.time.days", "days_to_death",
         "days_to_last_followup", "days_to_last_follow_up", "DSS.time", "PFI.time", "time"]
    )

    event_col = find_first_matching_col(
        clin,
        ["OS", "OS.event", "OS_event", "vital_status", "death_event", "event", "overall_survival"]
    )

    if time_col is not None:
        out["OS_time"] = pd.to_numeric(clin[time_col], errors="coerce")
    else:
        out["OS_time"] = np.nan

    if event_col is not None:
        out["OS_event"] = clin[event_col].map(parse_os_event)
    else:
        out["OS_event"] = np.nan

    death_col = find_first_matching_col(clin, ["days_to_death"])
    follow_col = find_first_matching_col(clin, ["days_to_last_followup", "days_to_last_follow_up"])
    vital_col = find_first_matching_col(clin, ["vital_status"])

    if death_col is not None or follow_col is not None:
        death = pd.to_numeric(clin[death_col], errors="coerce") if death_col else pd.Series(np.nan, index=clin.index)
        follow = pd.to_numeric(clin[follow_col], errors="coerce") if follow_col else pd.Series(np.nan, index=clin.index)
        best_time = death.combine_first(follow)
        out["OS_time"] = out["OS_time"].combine_first(best_time)
        if vital_col is not None:
            out["OS_event"] = out["OS_event"].combine_first(clin[vital_col].map(parse_os_event))
        else:
            out["OS_event"] = out["OS_event"].combine_first(death.notna().astype(int))

    stage_col = find_first_matching_col(
        clin,
        ["stage_group", "stage_binary", "TNM_stage_group",
         "pathologic_stage", "ajcc_pathologic_tumor_stage",
         "clinical_stage", "tumor_stage", "stage"]
    )
    out["stage_group"] = clin[stage_col].map(parse_stage_group) if stage_col is not None else None

    age_col = find_first_matching_col(
        clin,
        ["age_at_initial_pathologic_diagnosis", "age_at_diagnosis", "age", "diagnosis_age"]
    )
    if age_col is not None:
        age = pd.to_numeric(clin[age_col], errors="coerce")
        if age.dropna().shape[0] > 0 and age.dropna().median() > 200:
            age = age / 365.25
        out["age"] = age
    else:
        out["age"] = np.nan

    out.loc[out["OS_time"] <= 0, "OS_time"] = np.nan
    out["OS_event"] = pd.to_numeric(out["OS_event"], errors="coerce")
    return out, mapping_df


# =============================================================================
# 6. BP SCORE CONSTRUCTION
# =============================================================================

def construct_layer_bp_scores(
    mat_gene_by_patient: pd.DataFrame,
    gene_sets: Dict[str, Dict[str, Any]],
    layer: str,
    min_genes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mat = mat_gene_by_patient.copy()
    mat.index = mat.index.astype(str).str.upper()

    if layer in ["GE", "CN"]:
        zmat = zscore_df_by_rows(mat)
    else:
        zmat = mat.astype(float)

    measured = set(zmat.index)
    scores = {}
    readiness_rows = []

    for bp_key, rec in gene_sets.items():
        genes = rec["genes"]
        matched = sorted(set(g.upper() for g in genes).intersection(measured))
        matched_n = len(matched)
        ready = matched_n >= min_genes

        readiness_rows.append({
            "bp_key": bp_key,
            "bp": rec["bp"],
            "collection": rec["collection"],
            "layer": layer,
            "annotated_n": rec["annotated_n"],
            "matched_n": matched_n,
            "matched_fraction": matched_n / rec["annotated_n"] if rec["annotated_n"] else np.nan,
            "readiness_K": min_genes,
            "ready": int(ready),
        })

        if not ready:
            continue

        raw = zmat.loc[matched].mean(axis=0, skipna=True)
        scores[bp_key] = zscore_series(raw)

    if not scores:
        return pd.DataFrame(), pd.DataFrame(readiness_rows)

    score_df = pd.DataFrame(scores)
    score_df.index.name = "patient"
    return score_df, pd.DataFrame(readiness_rows)


def construct_representation_scores(
    layer_scores: Dict[str, pd.DataFrame],
    representations: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    rep_scores = {}

    for rep, layers in representations.items():
        if not all(layer in layer_scores and not layer_scores[layer].empty for layer in layers):
            continue

        common_patients = None
        common_bps = None

        for layer in layers:
            df = layer_scores[layer]
            common_patients = set(df.index) if common_patients is None else common_patients.intersection(df.index)
            common_bps = set(df.columns) if common_bps is None else common_bps.intersection(df.columns)

        common_patients = sorted(common_patients)
        common_bps = sorted(common_bps)

        if len(common_patients) < 30 or len(common_bps) == 0:
            continue

        avg = None
        for layer in layers:
            part = layer_scores[layer].loc[common_patients, common_bps]
            avg = part.copy() if avg is None else avg + part

        avg = avg / len(layers)
        avg = avg.apply(zscore_series, axis=0)
        avg.index.name = "patient"
        rep_scores[rep] = avg

    return rep_scores


# =============================================================================
# 7. ENDPOINT TESTS
# =============================================================================

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


def survival_test_from_score(score: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    common = sorted(set(score.dropna().index).intersection(clinical.index))
    if len(common) < 30:
        return {"p": np.nan, "D": np.nan, "n": len(common), "events": np.nan, "hr": np.nan, "c_index": np.nan}

    df = pd.DataFrame({
        "score": score.loc[common],
        "time": clinical.loc[common, "OS_time"],
        "event": clinical.loc[common, "OS_event"],
    }).dropna()
    df = df[df["time"] > 0]

    if df.shape[0] < 30 or df["event"].sum() < 5:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "events": df["event"].sum() if df.shape[0] else np.nan, "hr": np.nan, "c_index": np.nan}

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

    return {"p": p, "D": D, "n": int(df.shape[0]), "events": int(df["event"].sum()), "hr": hr, "c_index": cidx}


def stage_test_from_score(score: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    common = sorted(set(score.dropna().index).intersection(clinical.index))
    if len(common) < 30 or "stage_group" not in clinical.columns:
        return {"p": np.nan, "D": np.nan, "n": len(common), "OR": np.nan, "AUC": np.nan, "cliffs_delta": np.nan}

    df = pd.DataFrame({
        "score": score.loc[common],
        "stage": clinical.loc[common, "stage_group"],
    }).dropna()
    df = df[df["stage"].isin(["EARLY", "ADVANCED"])]

    if df.shape[0] < 30 or df["stage"].nunique() != 2:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan, "cliffs_delta": np.nan}

    med = df["score"].median()
    df["group"] = (df["score"] >= med).astype(int)
    df["advanced"] = (df["stage"] == "ADVANCED").astype(int)

    tab = pd.crosstab(df["group"], df["advanced"])
    if tab.shape != (2, 2):
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan, "cliffs_delta": np.nan}

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
    cliffs_delta = np.nan
    try:
        pos = df.loc[df["advanced"] == 1, "score"]
        neg = df.loc[df["advanced"] == 0, "score"]
        if len(pos) > 0 and len(neg) > 0:
            U, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            auc_raw = float(U / (len(pos) * len(neg)))
            AUC = max(auc_raw, 1 - auc_raw)
            cliffs_delta = 2 * auc_raw - 1
    except Exception:
        pass

    return {"p": p, "D": D, "n": int(df.shape[0]), "OR": OR, "AUC": AUC, "cliffs_delta": cliffs_delta}


def survival_test_from_group(group: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    group = group.dropna()
    common = sorted(set(group.index).intersection(clinical.index))
    if len(common) < 30:
        return {"p": np.nan, "D": np.nan, "n": len(common), "events": np.nan, "hr": np.nan, "c_index": np.nan}

    df = pd.DataFrame({
        "group": group.loc[common].astype(int),
        "time": clinical.loc[common, "OS_time"],
        "event": clinical.loc[common, "OS_event"],
    }).dropna()
    df = df[df["time"] > 0]

    if df.shape[0] < 30 or df["event"].sum() < 5 or df["group"].nunique() != 2:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "events": df["event"].sum() if df.shape[0] else np.nan, "hr": np.nan, "c_index": np.nan}

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
            cidx = float(concordance_index(df["time"], df["group"], df["event"]))
        except Exception:
            pass

    return {"p": p, "D": D, "n": int(df.shape[0]), "events": int(df["event"].sum()), "hr": hr, "c_index": cidx}


def stage_test_from_group(group: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    group = group.dropna()
    common = sorted(set(group.index).intersection(clinical.index))
    if len(common) < 30 or "stage_group" not in clinical.columns:
        return {"p": np.nan, "D": np.nan, "n": len(common), "OR": np.nan, "AUC": np.nan}

    df = pd.DataFrame({
        "group": group.loc[common].astype(int),
        "stage": clinical.loc[common, "stage_group"],
    }).dropna()
    df = df[df["stage"].isin(["EARLY", "ADVANCED"])]

    if df.shape[0] < 30 or df["stage"].nunique() != 2 or df["group"].nunique() != 2:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan}

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
        pos = df.loc[df["advanced"] == 1, "group"]
        neg = df.loc[df["advanced"] == 0, "group"]
        if len(pos) > 0 and len(neg) > 0:
            U, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            AUC = float(U / (len(pos) * len(neg)))
            AUC = max(AUC, 1 - AUC)
    except Exception:
        pass

    return {"p": p, "D": D, "n": int(df.shape[0]), "OR": OR, "AUC": AUC}


# =============================================================================
# 8. REPRESENTATION MAPS + DELTA-D + FRAGILITY
# =============================================================================

def evaluate_representation_maps(
    cohort: str,
    collection: str,
    readiness_K: int,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    gene_sets: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    all_bps = sorted(set().union(*[set(df.columns) for df in rep_scores.values()]))

    for bp_key in all_bps:
        rec = gene_sets.get(bp_key, {})
        for rep, score_df in rep_scores.items():
            if bp_key not in score_df.columns:
                continue

            score = score_df[bp_key]

            os_res = survival_test_from_score(score, clinical)
            rows.append({
                "cohort": cohort,
                "collection": collection,
                "readiness_K": readiness_K,
                "bp_key": bp_key,
                "bp": rec.get("bp", bp_key),
                "representation": rep,
                "endpoint": "OS",
                "p": os_res["p"],
                "D": os_res["D"],
                "n": os_res["n"],
                "events": os_res.get("events", np.nan),
                "HR": os_res.get("hr", np.nan),
                "C_index": os_res.get("c_index", np.nan),
                "OR": np.nan,
                "AUC": np.nan,
                "cliffs_delta": np.nan,
            })

            st_res = stage_test_from_score(score, clinical)
            rows.append({
                "cohort": cohort,
                "collection": collection,
                "readiness_K": readiness_K,
                "bp_key": bp_key,
                "bp": rec.get("bp", bp_key),
                "representation": rep,
                "endpoint": "STAGE",
                "p": st_res["p"],
                "D": st_res["D"],
                "n": st_res["n"],
                "events": np.nan,
                "HR": np.nan,
                "C_index": np.nan,
                "OR": st_res.get("OR", np.nan),
                "AUC": st_res.get("AUC", np.nan),
                "cliffs_delta": st_res.get("cliffs_delta", np.nan),
            })

    return pd.DataFrame(rows)


def add_fdr_columns(rep_map: pd.DataFrame) -> pd.DataFrame:
    if rep_map.empty or "p" not in rep_map.columns:
        return rep_map

    df = rep_map.copy()
    df["q_by_endpoint"] = np.nan
    df["q_by_endpoint_cohort"] = np.nan

    for endpoint, idx in df.groupby("endpoint").groups.items():
        df.loc[idx, "q_by_endpoint"] = bh_fdr(df.loc[idx, "p"])

    for (endpoint, cohort), idx in df.groupby(["endpoint", "cohort"]).groups.items():
        df.loc[idx, "q_by_endpoint_cohort"] = bh_fdr(df.loc[idx, "p"])

    df["endpoint_informative_p05"] = (df["p"] <= 0.05).astype(int)
    df["endpoint_informative_q10_by_endpoint"] = (df["q_by_endpoint"] <= 0.10).astype(int)
    df["endpoint_informative_q05_by_endpoint"] = (df["q_by_endpoint"] <= 0.05).astype(int)
    df["endpoint_informative_q10_by_endpoint_cohort"] = (df["q_by_endpoint_cohort"] <= 0.10).astype(int)
    df["endpoint_informative_q05_by_endpoint_cohort"] = (df["q_by_endpoint_cohort"] <= 0.05).astype(int)
    return df


def classify_representation(d_by_rep: Dict[str, float], best_rep: str, best_D: float) -> str:
    if pd.isna(best_D) or best_D < D_THRESHOLD:
        return "endpoint_weak"
    if best_rep == "GE":
        return "GE_sufficient"
    if best_rep == "GE_CN":
        return "CN_informative"
    if best_rep == "GE_MU":
        return "MU_informative"
    if best_rep == "GE_CN_MU":
        return "multi_layer_informative"
    return "other"


def select_best_representations(rep_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = {"cohort", "collection", "readiness_K", "bp_key", "bp", "endpoint", "representation", "D", "p"}
    if rep_map.empty or not required.issubset(rep_map.columns):
        return pd.DataFrame()

    for (cohort, collection, readiness_K, bp_key, bp, endpoint), sub in rep_map.groupby(
        ["cohort", "collection", "readiness_K", "bp_key", "bp", "endpoint"]
    ):
        sub2 = sub.dropna(subset=["D"])
        if sub2.empty:
            continue

        d_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}
        p_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}
        q_endpoint_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}
        q_ec_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}

        for _, row in sub2.iterrows():
            rep = row["representation"]
            d_by_rep[rep] = row["D"]
            p_by_rep[rep] = row["p"]
            q_endpoint_by_rep[rep] = row.get("q_by_endpoint", np.nan)
            q_ec_by_rep[rep] = row.get("q_by_endpoint_cohort", np.nan)

        best = sub2.loc[sub2["D"].idxmax()]
        best_rep = best["representation"]
        D_best = best["D"]
        D_ge = d_by_rep.get("GE", np.nan)

        delta_best_ge = D_best - D_ge if pd.notna(D_best) and pd.notna(D_ge) else np.nan

        delta_cn_ge = d_by_rep["GE_CN"] - D_ge if pd.notna(d_by_rep["GE_CN"]) and pd.notna(D_ge) else np.nan
        delta_mu_ge = d_by_rep["GE_MU"] - D_ge if pd.notna(d_by_rep["GE_MU"]) and pd.notna(D_ge) else np.nan
        delta_full_ge = d_by_rep["GE_CN_MU"] - D_ge if pd.notna(d_by_rep["GE_CN_MU"]) and pd.notna(D_ge) else np.nan

        endpoint_informative = int(pd.notna(D_best) and D_best >= D_THRESHOLD)
        rep_class = classify_representation(d_by_rep, best_rep, D_best)

        def fragile(delta: float) -> int:
            return int(pd.notna(delta) and delta < 0)

        def strong_fragile(delta: float) -> int:
            return int(pd.notna(delta) and delta <= STRONG_FRAGILITY_DELTA)

        def signal_lost(D_int: float) -> int:
            return int(pd.notna(D_ge) and pd.notna(D_int) and D_ge >= D_THRESHOLD and D_int < D_THRESHOLD)

        fragile_cn = fragile(delta_cn_ge)
        fragile_mu = fragile(delta_mu_ge)
        fragile_full = fragile(delta_full_ge)

        strong_fragile_cn = strong_fragile(delta_cn_ge)
        strong_fragile_mu = strong_fragile(delta_mu_ge)
        strong_fragile_full = strong_fragile(delta_full_ge)

        signal_lost_cn = signal_lost(d_by_rep["GE_CN"])
        signal_lost_mu = signal_lost(d_by_rep["GE_MU"])
        signal_lost_full = signal_lost(d_by_rep["GE_CN_MU"])

        any_fragile = int(fragile_cn or fragile_mu or fragile_full)
        any_strong_fragile = int(strong_fragile_cn or strong_fragile_mu or strong_fragile_full)
        any_signal_lost = int(signal_lost_cn or signal_lost_mu or signal_lost_full)

        if pd.isna(delta_best_ge):
            gain_class = "not_evaluable"
        elif delta_best_ge >= STRONG_GAIN_DELTA:
            gain_class = "strong_gain"
        elif delta_best_ge >= MODERATE_GAIN_DELTA:
            gain_class = "moderate_gain"
        elif delta_best_ge > 0:
            gain_class = "weak_gain"
        else:
            gain_class = "no_gain"

        rows.append({
            "cohort": cohort,
            "collection": collection,
            "readiness_K": readiness_K,
            "bp_key": bp_key,
            "bp": bp,
            "endpoint": endpoint,
            "best_representation": best_rep,
            "best_D": D_best,
            "best_p": best["p"],
            "best_q_by_endpoint": best.get("q_by_endpoint", np.nan),
            "best_q_by_endpoint_cohort": best.get("q_by_endpoint_cohort", np.nan),

            "D_GE": D_ge,
            "D_GE_CN": d_by_rep["GE_CN"],
            "D_GE_MU": d_by_rep["GE_MU"],
            "D_GE_CN_MU": d_by_rep["GE_CN_MU"],

            "p_GE": p_by_rep["GE"],
            "p_GE_CN": p_by_rep["GE_CN"],
            "p_GE_MU": p_by_rep["GE_MU"],
            "p_GE_CN_MU": p_by_rep["GE_CN_MU"],

            "q_endpoint_GE": q_endpoint_by_rep["GE"],
            "q_endpoint_GE_CN": q_endpoint_by_rep["GE_CN"],
            "q_endpoint_GE_MU": q_endpoint_by_rep["GE_MU"],
            "q_endpoint_GE_CN_MU": q_endpoint_by_rep["GE_CN_MU"],

            "deltaD_best_minus_GE": delta_best_ge,
            "deltaD_GE_CN_minus_GE": delta_cn_ge,
            "deltaD_GE_MU_minus_GE": delta_mu_ge,
            "deltaD_GE_CN_MU_minus_GE": delta_full_ge,
            "deltaD_best_minus_full": D_best - d_by_rep["GE_CN_MU"] if pd.notna(D_best) and pd.notna(d_by_rep["GE_CN_MU"]) else np.nan,

            "endpoint_informative": endpoint_informative,
            "endpoint_informative_q10_by_endpoint": int(pd.notna(best.get("q_by_endpoint", np.nan)) and best.get("q_by_endpoint", np.nan) <= 0.10),
            "endpoint_informative_q05_by_endpoint": int(pd.notna(best.get("q_by_endpoint", np.nan)) and best.get("q_by_endpoint", np.nan) <= 0.05),
            "representation_class": rep_class,
            "gain_class": gain_class,

            "fragile_GE_CN": fragile_cn,
            "fragile_GE_MU": fragile_mu,
            "fragile_GE_CN_MU": fragile_full,
            "strong_fragile_GE_CN": strong_fragile_cn,
            "strong_fragile_GE_MU": strong_fragile_mu,
            "strong_fragile_GE_CN_MU": strong_fragile_full,
            "signal_lost_GE_CN": signal_lost_cn,
            "signal_lost_GE_MU": signal_lost_mu,
            "signal_lost_GE_CN_MU": signal_lost_full,
            "any_fragile": any_fragile,
            "any_strong_fragile": any_strong_fragile,
            "any_signal_lost": any_signal_lost,

            "n": best.get("n", np.nan),
            "events": best.get("events", np.nan),
            "HR": best.get("HR", np.nan),
            "C_index": best.get("C_index", np.nan),
            "OR": best.get("OR", np.nan),
            "AUC": best.get("AUC", np.nan),
            "cliffs_delta": best.get("cliffs_delta", np.nan),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 9. TRUE SIZE-MATCHED RANDOM GENE-SET BASELINE
# =============================================================================

def build_bp_score_from_gene_list(
    mat_gene_by_patient: pd.DataFrame,
    genes: List[str],
    layer: str,
) -> pd.Series:
    genes = sorted(set(g.upper() for g in genes).intersection(mat_gene_by_patient.index))
    if len(genes) == 0:
        return pd.Series(dtype=float)

    if layer in ["GE", "CN"]:
        zmat = zscore_df_by_rows(mat_gene_by_patient.loc[genes])
        raw = zmat.mean(axis=0, skipna=True)
    else:
        raw = mat_gene_by_patient.loc[genes].astype(float).mean(axis=0, skipna=True)

    return zscore_series(raw)


def evaluate_score_for_endpoint(score: pd.Series, endpoint: str, clinical: pd.DataFrame) -> Dict[str, Any]:
    if endpoint == "OS":
        return survival_test_from_score(score, clinical)
    if endpoint == "STAGE":
        return stage_test_from_score(score, clinical)
    raise ValueError(f"Unknown endpoint: {endpoint}")


def size_matched_random_gene_set_baseline(
    cohort: str,
    collection: str,
    readiness_K: int,
    layer_mats: Dict[str, pd.DataFrame],
    gene_sets: Dict[str, Dict[str, Any]],
    selected: pd.DataFrame,
    clinical: pd.DataFrame,
    n_iter: int,
) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()

    measured_genes_by_layer = {layer: sorted(set(mat.index)) for layer, mat in layer_mats.items()}

    # Only run for endpoint-informative rows to control runtime.
    selected2 = selected[
        (selected["endpoint_informative"] == 1)
        & (selected["readiness_K"] == readiness_K)
        & (selected["collection"] == collection)
    ].copy()

    if selected2.empty:
        return pd.DataFrame()

    for i, row in selected2.iterrows():
        bp_key = row["bp_key"]
        endpoint = row["endpoint"]
        best_rep = row["best_representation"]
        layers = REPRESENTATIONS.get(best_rep, ["GE"])
        rec = gene_sets.get(bp_key)
        if rec is None:
            continue

        real_genes = set(rec["genes"])

        # Matched genes must exist in every layer used by the selected representation.
        common_measured = None
        for layer in layers:
            common_measured = set(measured_genes_by_layer[layer]) if common_measured is None else common_measured.intersection(measured_genes_by_layer[layer])

        matched_genes = sorted(real_genes.intersection(common_measured))
        matched_n = len(matched_genes)

        if matched_n < readiness_K or len(common_measured) < matched_n + 5:
            continue

        random_D_values = []
        random_p_values = []

        common_measured_list = sorted(common_measured)

        for t in range(n_iter):
            sampled_genes = random.sample(common_measured_list, matched_n)

            layer_scores = []
            common_patients = None

            for layer in layers:
                score = build_bp_score_from_gene_list(layer_mats[layer], sampled_genes, layer)
                if score.empty:
                    continue
                layer_scores.append(score)
                common_patients = set(score.index) if common_patients is None else common_patients.intersection(score.index)

            if len(layer_scores) != len(layers) or common_patients is None or len(common_patients) < 30:
                continue

            common_patients = sorted(common_patients)
            mat = pd.concat([s.loc[common_patients] for s in layer_scores], axis=1)
            random_score = zscore_series(mat.mean(axis=1))
            res = evaluate_score_for_endpoint(random_score, endpoint, clinical)

            if pd.notna(res.get("D", np.nan)):
                random_D_values.append(res["D"])
                random_p_values.append(res["p"])

        rand = pd.Series(random_D_values).dropna()

        rows.append({
            "cohort": cohort,
            "collection": collection,
            "readiness_K": readiness_K,
            "bp_key": bp_key,
            "bp": row["bp"],
            "endpoint": endpoint,
            "best_representation": best_rep,
            "layers": "+".join(layers),
            "matched_gene_count": matched_n,
            "real_best_D": row["best_D"],
            "real_best_p": row["best_p"],
            "random_gene_set_iter_requested": n_iter,
            "random_gene_set_iter_valid": int(len(rand)),
            "random_gene_set_D_median": rand.median() if len(rand) else np.nan,
            "random_gene_set_D_95pct": rand.quantile(0.95) if len(rand) else np.nan,
            "real_exceeds_random95": int(pd.notna(row["best_D"]) and len(rand) and row["best_D"] > rand.quantile(0.95)),
            "empirical_p_ge_observed": ((rand >= row["best_D"]).sum() + 1) / (len(rand) + 1) if len(rand) and pd.notna(row["best_D"]) else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 10. BP BURDEN DOWNSTREAM STRESS TEST
# =============================================================================

def infer_risk_direction(score: pd.Series, clinical: pd.DataFrame) -> int:
    common = sorted(set(score.dropna().index).intersection(clinical.index))
    df = pd.DataFrame({
        "score": score.loc[common],
        "time": clinical.loc[common, "OS_time"],
        "event": clinical.loc[common, "OS_event"],
    }).dropna()
    df = df[df["time"] > 0]

    if df.shape[0] < 20 or df["event"].sum() < 3:
        return 1

    med = df["score"].median()
    high = df[df["score"] >= med]
    low = df[df["score"] < med]
    if high.empty or low.empty:
        return 1

    eh = high["event"].mean()
    el = low["event"].mean()
    if eh > el:
        return 1
    if eh < el:
        return -1

    th = high["time"].mean()
    tl = low["time"].mean()
    return 1 if th < tl else -1


def abnormal_indicator(
    score: pd.Series,
    direction: int,
    cutoff: str,
    reference_score: Optional[pd.Series] = None,
) -> pd.Series:
    s = score.copy()
    ref = s.dropna() if reference_score is None else reference_score.dropna()
    out = pd.Series(0, index=s.index, dtype=int)

    if cutoff == "SD1":
        if direction == 1:
            out.loc[s >= 1.0] = 1
        else:
            out.loc[s <= -1.0] = 1
    elif cutoff == "EXT20":
        if ref.shape[0] < 5:
            return out
        if direction == 1:
            q = ref.quantile(0.80)
            out.loc[s >= q] = 1
        else:
            q = ref.quantile(0.20)
            out.loc[s <= q] = 1
    else:
        raise ValueError(f"Unknown cutoff: {cutoff}")
    return out


def construct_bp_burden(
    cohort: str,
    collection: str,
    readiness_K: int,
    endpoint: str,
    selected_table: pd.DataFrame,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    topK: int,
    cutoff: str,
    patient_subset: Optional[List[str]] = None,
    training_reference_patients: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = selected_table[
        (selected_table["cohort"] == cohort)
        & (selected_table["collection"] == collection)
        & (selected_table["readiness_K"] == readiness_K)
        & (selected_table["endpoint"] == endpoint)
        & (selected_table["endpoint_informative"] == 1)
    ].copy()

    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = sub.sort_values("best_D", ascending=False).head(topK)
    selected_rows = []
    abnormal_cols = []
    common_patients = None

    for _, row in sub.iterrows():
        bp_key = row["bp_key"]
        rep = row["best_representation"]

        if rep not in rep_scores or bp_key not in rep_scores[rep].columns:
            continue

        score = rep_scores[rep][bp_key].dropna()

        if training_reference_patients is not None:
            ref_patients = sorted(set(training_reference_patients).intersection(score.index).intersection(clinical.index))
        else:
            ref_patients = sorted(set(score.index).intersection(clinical.index))

        if len(ref_patients) < 20:
            continue

        ref_score = score.loc[ref_patients]
        direction = infer_risk_direction(ref_score, clinical.loc[ref_patients])

        if patient_subset is not None:
            patients = sorted(set(patient_subset).intersection(score.index))
        else:
            patients = sorted(set(score.index).intersection(clinical.index))

        if len(patients) < 20:
            continue

        abn = abnormal_indicator(
            score.loc[patients],
            direction=direction,
            cutoff=cutoff,
            reference_score=ref_score if cutoff == "EXT20" else None,
        )

        abnormal_cols.append(abn.rename(bp_key))
        selected_rows.append({
            "cohort": cohort,
            "collection": collection,
            "readiness_K": readiness_K,
            "endpoint": endpoint,
            "topK": topK,
            "cutoff": cutoff,
            "bp_key": bp_key,
            "bp": row["bp"],
            "representation": rep,
            "best_D": row["best_D"],
            "D_GE": row["D_GE"],
            "D_GE_CN": row["D_GE_CN"],
            "D_GE_MU": row["D_GE_MU"],
            "D_GE_CN_MU": row["D_GE_CN_MU"],
            "deltaD_best_minus_GE": row["deltaD_best_minus_GE"],
            "risk_direction": "high_score_risk" if direction == 1 else "low_score_risk",
        })

        common_patients = set(patients) if common_patients is None else common_patients.intersection(patients)

    if not abnormal_cols or common_patients is None or len(common_patients) < 20:
        return pd.DataFrame(), pd.DataFrame()

    common_patients = sorted(common_patients)
    abn_df = pd.concat(abnormal_cols, axis=1).loc[common_patients]
    burden = abn_df.sum(axis=1)

    burden_df = pd.DataFrame({
        "patient": burden.index,
        "cohort": cohort,
        "collection": collection,
        "readiness_K": readiness_K,
        "endpoint_selection": endpoint,
        "topK": topK,
        "cutoff": cutoff,
        "burden": burden.values,
    })

    high_cutoff = burden.quantile(0.67)
    burden_df["high_burden"] = (burden_df["burden"] >= high_cutoff).astype(int)
    burden_df["burden_cutoff_q67"] = high_cutoff

    return burden_df, pd.DataFrame(selected_rows)


def evaluate_burden(burden_df: pd.DataFrame, clinical: pd.DataFrame) -> Dict[str, Any]:
    if burden_df.empty or "patient" not in burden_df.columns or "high_burden" not in burden_df.columns:
        return {
            "OS_p": np.nan, "OS_D": np.nan, "OS_HR": np.nan, "OS_C_index": np.nan,
            "STAGE_p": np.nan, "STAGE_D": np.nan, "STAGE_OR": np.nan, "STAGE_AUC": np.nan,
            "n_OS": np.nan, "events": np.nan, "n_STAGE": np.nan,
        }

    b = burden_df.set_index("patient")
    group = b["high_burden"].astype(int)
    os_res = survival_test_from_group(group, clinical)
    st_res = stage_test_from_group(group, clinical)

    return {
        "OS_p": os_res["p"],
        "OS_D": os_res["D"],
        "OS_HR": os_res["hr"],
        "OS_C_index": os_res["c_index"],
        "n_OS": os_res["n"],
        "events": os_res["events"],
        "STAGE_p": st_res["p"],
        "STAGE_D": st_res["D"],
        "STAGE_OR": st_res["OR"],
        "STAGE_AUC": st_res["AUC"],
        "n_STAGE": st_res["n"],
    }


def random_bp_burden_baseline(
    cohort: str,
    collection: str,
    readiness_K: int,
    endpoint: str,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    candidate_bps: List[str],
    topK: int,
    cutoff: str,
    n_iter: int,
) -> pd.DataFrame:
    rows = []
    if "GE" not in rep_scores:
        return pd.DataFrame()

    ge_scores = rep_scores["GE"]
    candidate_bps = sorted(set(candidate_bps).intersection(ge_scores.columns))
    if len(candidate_bps) < topK:
        return pd.DataFrame()

    for t in range(n_iter):
        sampled = random.sample(candidate_bps, topK)
        abnormal_cols = []
        common_patients = None

        for bp_key in sampled:
            score = ge_scores[bp_key].dropna()
            patients = sorted(set(score.index).intersection(clinical.index))
            if len(patients) < 20:
                continue
            direction = infer_risk_direction(score.loc[patients], clinical.loc[patients])
            abn = abnormal_indicator(score.loc[patients], direction=direction, cutoff=cutoff)
            abnormal_cols.append(abn.rename(bp_key))
            common_patients = set(patients) if common_patients is None else common_patients.intersection(patients)

        if not abnormal_cols or common_patients is None or len(common_patients) < 20:
            continue

        common_patients = sorted(common_patients)
        abn_df = pd.concat(abnormal_cols, axis=1).loc[common_patients]
        burden = abn_df.sum(axis=1)

        bdf = pd.DataFrame({
            "patient": burden.index,
            "cohort": cohort,
            "collection": collection,
            "readiness_K": readiness_K,
            "endpoint_selection": endpoint,
            "topK": topK,
            "cutoff": cutoff,
            "burden": burden.values,
        })
        q67 = burden.quantile(0.67)
        bdf["high_burden"] = (bdf["burden"] >= q67).astype(int)
        bdf["burden_cutoff_q67"] = q67

        eval_res = evaluate_burden(bdf, clinical)
        rows.append({
            "cohort": cohort,
            "collection": collection,
            "readiness_K": readiness_K,
            "endpoint_selection": endpoint,
            "topK": topK,
            "cutoff": cutoff,
            "iteration": t + 1,
            "random_OS_D": eval_res["OS_D"],
            "random_STAGE_D": eval_res["STAGE_D"],
            "random_OS_p": eval_res["OS_p"],
            "random_STAGE_p": eval_res["STAGE_p"],
            "random_OS_HR": eval_res["OS_HR"],
            "random_STAGE_OR": eval_res["STAGE_OR"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# 11. SPLIT VALIDATION
# =============================================================================

def train_selected_representations(
    cohort: str,
    collection: str,
    readiness_K: int,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    train_patients: List[str],
    gene_sets: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    clinical_train = clinical.loc[train_patients].copy()
    rep_scores_train = {}

    for rep, df in rep_scores.items():
        pats = sorted(set(train_patients).intersection(df.index))
        if len(pats) >= 20:
            rep_scores_train[rep] = df.loc[pats].copy()

    rep_map_train = evaluate_representation_maps(
        cohort=cohort,
        collection=collection,
        readiness_K=readiness_K,
        rep_scores=rep_scores_train,
        clinical=clinical_train,
        gene_sets=gene_sets,
    )
    rep_map_train = add_fdr_columns(rep_map_train)
    selected_train = select_best_representations(rep_map_train)
    return selected_train


def split_validation(
    cohort: str,
    collection: str,
    readiness_K: int,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    gene_sets: Dict[str, Dict[str, Any]],
    topK: int,
    cutoff: str,
    n_iter: int,
) -> pd.DataFrame:
    rows = []
    available_patients = set(clinical.dropna(subset=["OS_time", "OS_event"]).index)
    for rep_df in rep_scores.values():
        available_patients = available_patients.intersection(rep_df.index)
    available_patients = sorted(available_patients)

    if len(available_patients) < 80:
        return pd.DataFrame()

    for t in range(n_iter):
        pats = available_patients.copy()
        random.shuffle(pats)

        n_train = int(len(pats) * TRAIN_FRAC)
        train = sorted(pats[:n_train])
        test = sorted(pats[n_train:])

        if len(train) < 40 or len(test) < 20:
            continue

        try:
            selected_train = train_selected_representations(
                cohort, collection, readiness_K, rep_scores, clinical, train, gene_sets
            )
        except Exception as e:
            log(f"Split train selection failed: {cohort} K={readiness_K} iter {t + 1}: {e}")
            continue

        for endpoint in ENDPOINTS:
            bdf_test, selected_bps = construct_bp_burden(
                cohort=cohort,
                collection=collection,
                readiness_K=readiness_K,
                endpoint=endpoint,
                selected_table=selected_train,
                rep_scores=rep_scores,
                clinical=clinical,
                topK=topK,
                cutoff=cutoff,
                patient_subset=test,
                training_reference_patients=train,
            )

            eval_res = evaluate_burden(bdf_test, clinical)

            rows.append({
                "cohort": cohort,
                "collection": collection,
                "readiness_K": readiness_K,
                "iteration": t + 1,
                "endpoint_selection": endpoint,
                "topK": topK,
                "cutoff": cutoff,
                "n_train": len(train),
                "n_test": len(test),
                "n_selected_bp": selected_bps.shape[0] if not selected_bps.empty else 0,
                "test_OS_D": eval_res["OS_D"],
                "test_OS_p": eval_res["OS_p"],
                "test_OS_HR": eval_res["OS_HR"],
                "test_OS_C_index": eval_res["OS_C_index"],
                "test_STAGE_D": eval_res["STAGE_D"],
                "test_STAGE_p": eval_res["STAGE_p"],
                "test_STAGE_OR": eval_res["STAGE_OR"],
                "test_STAGE_AUC": eval_res["STAGE_AUC"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# 12. PROCESS ONE COHORT
# =============================================================================

def process_one_cohort(
    cohort: str,
    folder: Path,
    all_gene_sets: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    log("=" * 90)
    log(f"Processing cohort {cohort}: {folder}")

    cohort_out = out_dir / "cohort_outputs" / cohort
    ensure_dir(cohort_out)

    files = get_cohort_files(folder, cohort)
    write_json({k: str(v) if v is not None else None for k, v in files.items()}, cohort_out / f"{cohort}_input_files.json")

    if cohort in DEFAULT_EXCLUDE_CODES:
        log(f"Skipping {cohort}: default excluded")
        return {"cohort": cohort, "status": "skipped_default_exclude"}

    if files["GE"] is None or files["CN"] is None or files["MU"] is None:
        log(f"Skipping {cohort}: missing GE/CN/MU")
        return {"cohort": cohort, "status": "skipped_missing_molecular"}

    layer_mats = {}
    for layer in ["GE", "CN", "MU"]:
        try:
            layer_mats[layer] = load_layer_matrix(files[layer], layer)
        except Exception as e:
            log(f"Failed to load {layer} for {cohort}: {e}")
            return {"cohort": cohort, "status": f"skipped_failed_load_{layer}", "error": str(e)}

    molecular_patients = sorted(
        set(layer_mats["GE"].columns)
        .intersection(layer_mats["CN"].columns)
        .intersection(layer_mats["MU"].columns)
    )

    clinical, clinical_mapping = load_clinical(files, molecular_patients=molecular_patients)
    if clinical.empty:
        log(f"Skipping {cohort}: missing clinical")
        return {"cohort": cohort, "status": "skipped_missing_clinical"}

    write_csv(clinical.reset_index().rename(columns={"index": "patient"}), cohort_out / f"{cohort}_clinical_parsed.csv")
    if not clinical_mapping.empty:
        write_csv(clinical_mapping, cohort_out / f"{cohort}_clinical_id_mapping.csv")

    matched_clinical_molecular = sorted(set(clinical.index).intersection(molecular_patients))
    log(
        f"{cohort}: molecular common patients={len(molecular_patients)}, "
        f"clinical patients={clinical.shape[0]}, matched={len(matched_clinical_molecular)}, "
        f"OS_available={clinical.loc[matched_clinical_molecular, ['OS_time', 'OS_event']].dropna().shape[0] if matched_clinical_molecular else 0}, "
        f"stage_available={clinical.loc[matched_clinical_molecular, 'stage_group'].isin(['EARLY', 'ADVANCED']).sum() if matched_clinical_molecular else 0}"
    )

    if len(matched_clinical_molecular) < 30:
        return {"cohort": cohort, "status": "skipped_too_few_matched_patients"}

    # Keep only primary collection by default for full expensive analysis.
    primary_gene_sets = subset_gene_sets(all_gene_sets, PRIMARY_GENESET_COLLECTION)
    if not primary_gene_sets:
        raise RuntimeError(f"Primary collection not found: {PRIMARY_GENESET_COLLECTION}")

    status_parts = []
    all_readiness = []
    all_rep_map = []
    all_selected = []
    all_random_geneset = []
    all_burden_summary = []
    all_selected_burden_bp = []
    all_random_bp_burden_summary = []
    all_split = []

    for readiness_K in READINESS_K_VALUES:
        log(f"{cohort}: TRUE readiness matched-gene cutoff K={readiness_K}")

        layer_scores = {}
        readiness_k_all = []

        for layer in ["GE", "CN", "MU"]:
            score_df, ready_df = construct_layer_bp_scores(
                layer_mats[layer],
                gene_sets=primary_gene_sets,
                layer=layer,
                min_genes=readiness_K,
            )
            layer_scores[layer] = score_df
            ready_df["cohort"] = cohort
            readiness_k_all.append(ready_df)

            # Save BP scores only for primary K to reduce output size.
            if readiness_K == PRIMARY_READINESS_K:
                write_csv(score_df.reset_index(), cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_{layer}_BP_scores_K{readiness_K}.csv")

        readiness_df = pd.concat(readiness_k_all, ignore_index=True)
        all_readiness.append(readiness_df)
        write_csv(readiness_df, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_readiness_all_layers_K{readiness_K}.csv")

        rep_scores = construct_representation_scores(layer_scores, REPRESENTATIONS)
        if "GE" not in rep_scores:
            log(f"{cohort}: no GE representation at readiness_K={readiness_K}")
            continue

        if readiness_K == PRIMARY_READINESS_K:
            for rep, df in rep_scores.items():
                write_csv(df.reset_index(), cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_{rep}_representation_scores_K{readiness_K}.csv")

        rep_map = evaluate_representation_maps(
            cohort=cohort,
            collection=PRIMARY_GENESET_COLLECTION,
            readiness_K=readiness_K,
            rep_scores=rep_scores,
            clinical=clinical,
            gene_sets=primary_gene_sets,
        )
        rep_map = add_fdr_columns(rep_map)
        selected = select_best_representations(rep_map)

        all_rep_map.append(rep_map)
        all_selected.append(selected)

        write_csv(rep_map, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_representation_map_all_reps_K{readiness_K}.csv")
        write_csv(selected, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_selected_representations_deltaD_fragility_K{readiness_K}.csv")

        # Size-matched random gene-set baseline: primary readiness K only.
        if RUN_RANDOM_GENESET_BASELINE and readiness_K == PRIMARY_READINESS_K:
            log(f"{cohort}: size-matched random gene-set baseline K={readiness_K}")
            rand_gs = size_matched_random_gene_set_baseline(
                cohort=cohort,
                collection=PRIMARY_GENESET_COLLECTION,
                readiness_K=readiness_K,
                layer_mats=layer_mats,
                gene_sets=primary_gene_sets,
                selected=selected,
                clinical=clinical,
                n_iter=RANDOM_GENESET_ITER,
            )
            if not rand_gs.empty:
                all_random_geneset.append(rand_gs)
                write_csv(rand_gs, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_size_matched_random_geneset_K{readiness_K}.csv")

        # Burden and split modules: primary readiness K only.
        if RUN_BURDEN_MODULE and readiness_K == PRIMARY_READINESS_K:
            burden_rows = []
            selected_bp_rows = []

            for topK in BURDEN_TOPK_VALUES:
                for endpoint in ENDPOINTS:
                    for cutoff in ABNORMALITY_CUTOFFS:
                        bdf, sdf = construct_bp_burden(
                            cohort=cohort,
                            collection=PRIMARY_GENESET_COLLECTION,
                            readiness_K=readiness_K,
                            endpoint=endpoint,
                            selected_table=selected,
                            rep_scores=rep_scores,
                            clinical=clinical,
                            topK=topK,
                            cutoff=cutoff,
                        )

                        if not bdf.empty:
                            eval_res = evaluate_burden(bdf, clinical)
                            burden_rows.append({
                                "cohort": cohort,
                                "collection": PRIMARY_GENESET_COLLECTION,
                                "readiness_K": readiness_K,
                                "endpoint_selection": endpoint,
                                "topK": topK,
                                "cutoff": cutoff,
                                "n_patients": bdf.shape[0],
                                "mean_burden": bdf["burden"].mean(),
                                "median_burden": bdf["burden"].median(),
                                "high_burden_fraction": bdf["high_burden"].mean(),
                                **eval_res,
                            })
                            write_csv(bdf, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_burden_{endpoint}_K{readiness_K}_top{topK}_{cutoff}.csv")

                        if not sdf.empty:
                            selected_bp_rows.append(sdf)

            burden_summary = pd.DataFrame(burden_rows)
            selected_bp_summary = pd.concat(selected_bp_rows, ignore_index=True) if selected_bp_rows else pd.DataFrame()

            if not burden_summary.empty:
                all_burden_summary.append(burden_summary)
                write_csv(burden_summary, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_burden_topK_sensitivity_summary_K{readiness_K}.csv")

            if not selected_bp_summary.empty:
                all_selected_burden_bp.append(selected_bp_summary)
                write_csv(selected_bp_summary, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_selected_BP_for_burden_all_topK_K{readiness_K}.csv")

            if RUN_RANDOM_BP_BURDEN_BASELINE and not burden_summary.empty:
                candidate_bps = sorted(rep_scores["GE"].columns)
                observed_lookup = burden_summary[burden_summary["topK"] == PRIMARY_BURDEN_TOPK].copy()

                random_rows = []
                for endpoint in ENDPOINTS:
                    for cutoff in ABNORMALITY_CUTOFFS:
                        log(f"{cohort}: random BP-burden baseline endpoint={endpoint}, cutoff={cutoff}, topK={PRIMARY_BURDEN_TOPK}")
                        rand_df = random_bp_burden_baseline(
                            cohort=cohort,
                            collection=PRIMARY_GENESET_COLLECTION,
                            readiness_K=readiness_K,
                            endpoint=endpoint,
                            rep_scores=rep_scores,
                            clinical=clinical,
                            candidate_bps=candidate_bps,
                            topK=PRIMARY_BURDEN_TOPK,
                            cutoff=cutoff,
                            n_iter=RANDOM_BP_BURDEN_ITER,
                        )
                        if not rand_df.empty:
                            write_csv(rand_df, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_random_BP_burden_{endpoint}_K{readiness_K}_top{PRIMARY_BURDEN_TOPK}_{cutoff}.csv")

                            obs_sub = observed_lookup[
                                (observed_lookup["endpoint_selection"] == endpoint)
                                & (observed_lookup["cutoff"] == cutoff)
                            ]

                            obs_os = obs_sub["OS_D"].iloc[0] if not obs_sub.empty and "OS_D" in obs_sub.columns else np.nan
                            obs_stage = obs_sub["STAGE_D"].iloc[0] if not obs_sub.empty and "STAGE_D" in obs_sub.columns else np.nan

                            rand_os = rand_df["random_OS_D"].dropna()
                            rand_stage = rand_df["random_STAGE_D"].dropna()

                            random_rows.append({
                                "cohort": cohort,
                                "collection": PRIMARY_GENESET_COLLECTION,
                                "readiness_K": readiness_K,
                                "endpoint_selection": endpoint,
                                "topK": PRIMARY_BURDEN_TOPK,
                                "cutoff": cutoff,
                                "observed_OS_D": obs_os,
                                "random_OS_median": rand_os.median() if len(rand_os) else np.nan,
                                "random_OS_95pct": rand_os.quantile(0.95) if len(rand_os) else np.nan,
                                "empirical_OS_p_ge_observed": ((rand_os >= obs_os).sum() + 1) / (len(rand_os) + 1) if len(rand_os) and pd.notna(obs_os) else np.nan,
                                "observed_STAGE_D": obs_stage,
                                "random_STAGE_median": rand_stage.median() if len(rand_stage) else np.nan,
                                "random_STAGE_95pct": rand_stage.quantile(0.95) if len(rand_stage) else np.nan,
                                "empirical_STAGE_p_ge_observed": ((rand_stage >= obs_stage).sum() + 1) / (len(rand_stage) + 1) if len(rand_stage) and pd.notna(obs_stage) else np.nan,
                                "n_random_iter_OS": len(rand_os),
                                "n_random_iter_STAGE": len(rand_stage),
                            })

                random_bp_summary = pd.DataFrame(random_rows)
                if not random_bp_summary.empty:
                    all_random_bp_burden_summary.append(random_bp_summary)
                    write_csv(random_bp_summary, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_random_BP_burden_summary_K{readiness_K}.csv")

            if RUN_SPLIT_VALIDATION:
                log(f"{cohort}: split validation K={readiness_K}, topK={PRIMARY_BURDEN_TOPK}, cutoff=EXT20")
                split_df = split_validation(
                    cohort=cohort,
                    collection=PRIMARY_GENESET_COLLECTION,
                    readiness_K=readiness_K,
                    rep_scores=rep_scores,
                    clinical=clinical,
                    gene_sets=primary_gene_sets,
                    topK=PRIMARY_BURDEN_TOPK,
                    cutoff="EXT20",
                    n_iter=SPLIT_ITER,
                )
                if not split_df.empty:
                    all_split.append(split_df)
                    write_csv(split_df, cohort_out / f"{cohort}_{PRIMARY_GENESET_COLLECTION}_split_validation_K{readiness_K}_top{PRIMARY_BURDEN_TOPK}_EXT20.csv")

        status_parts.append({
            "cohort": cohort,
            "collection": PRIMARY_GENESET_COLLECTION,
            "readiness_K": readiness_K,
            "n_rep_scores": len(rep_scores),
            "n_rep_map_rows": int(rep_map.shape[0]),
            "n_selected_rows": int(selected.shape[0]),
        })

        del layer_scores, rep_scores, rep_map, selected
        gc.collect()

    # Save all cohort-level combined outputs.
    if all_readiness:
        write_csv(pd.concat(all_readiness, ignore_index=True), cohort_out / f"{cohort}_ALL_readiness_sensitivity.csv")
    if all_rep_map:
        write_csv(pd.concat(all_rep_map, ignore_index=True), cohort_out / f"{cohort}_ALL_representation_maps.csv")
    if all_selected:
        write_csv(pd.concat(all_selected, ignore_index=True), cohort_out / f"{cohort}_ALL_selected_representations_deltaD_fragility.csv")
    if all_random_geneset:
        write_csv(pd.concat(all_random_geneset, ignore_index=True), cohort_out / f"{cohort}_ALL_size_matched_random_geneset.csv")
    if all_burden_summary:
        write_csv(pd.concat(all_burden_summary, ignore_index=True), cohort_out / f"{cohort}_ALL_burden_summary.csv")
    if all_selected_burden_bp:
        write_csv(pd.concat(all_selected_burden_bp, ignore_index=True), cohort_out / f"{cohort}_ALL_selected_BP_for_burden.csv")
    if all_random_bp_burden_summary:
        write_csv(pd.concat(all_random_bp_burden_summary, ignore_index=True), cohort_out / f"{cohort}_ALL_random_BP_burden_summary.csv")
    if all_split:
        write_csv(pd.concat(all_split, ignore_index=True), cohort_out / f"{cohort}_ALL_split_validation.csv")

    status = {
        "cohort": cohort,
        "status": "completed",
        "n_molecular_common": int(len(molecular_patients)),
        "n_clinical": int(clinical.shape[0]),
        "n_matched_clinical_molecular": int(len(matched_clinical_molecular)),
        "n_OS_available_matched": int(clinical.loc[matched_clinical_molecular, ["OS_time", "OS_event"]].dropna().shape[0]),
        "n_OS_events_matched": int(pd.to_numeric(clinical.loc[matched_clinical_molecular, "OS_event"], errors="coerce").fillna(0).sum()),
        "n_stage_available_matched": int(clinical.loc[matched_clinical_molecular, "stage_group"].isin(["EARLY", "ADVANCED"]).sum()),
        "status_parts": status_parts,
    }

    write_json(status, cohort_out / f"{cohort}_status.json")

    del layer_mats
    gc.collect()
    return status


# =============================================================================
# 13. AGGREGATION, SUMMARY TABLES, FIGURES
# =============================================================================

def aggregate_outputs(out_dir: Path, cohort_status: pd.DataFrame) -> None:
    cohort_root = out_dir / "cohort_outputs"
    aggregate_dir = out_dir / "aggregate_tables"
    ensure_dir(aggregate_dir)

    patterns = {
        "all_readiness_sensitivity": "*_ALL_readiness_sensitivity.csv",
        "all_representation_maps": "*_ALL_representation_maps.csv",
        "all_selected_representations": "*_ALL_selected_representations_deltaD_fragility.csv",
        "all_size_matched_random_geneset": "*_ALL_size_matched_random_geneset.csv",
        "all_burden_summary": "*_ALL_burden_summary.csv",
        "all_selected_BP_for_burden": "*_ALL_selected_BP_for_burden.csv",
        "all_random_BP_burden_summary": "*_ALL_random_BP_burden_summary.csv",
        "all_split_validation": "*_ALL_split_validation.csv",
        "all_clinical_id_mapping": "*_clinical_id_mapping.csv",
    }

    for name, pattern in patterns.items():
        dfs = []
        for p in cohort_root.glob(f"*/{pattern}"):
            try:
                dfs.append(pd.read_csv(p, encoding="utf-8-sig"))
            except Exception as e:
                log(f"Aggregate read failed: {p} | {e}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            write_csv(df, aggregate_dir / f"{name}.csv")
            log(f"Aggregated {name}: {df.shape}")
        else:
            log(f"No aggregate files for {name}")

    write_csv(cohort_status, aggregate_dir / "cohort_status_summary.csv")


def make_summary_tables(out_dir: Path) -> None:
    agg = out_dir / "aggregate_tables"

    selected_path = agg / "all_selected_representations.csv"
    readiness_path = agg / "all_readiness_sensitivity.csv"
    rand_gs_path = agg / "all_size_matched_random_geneset.csv"
    burden_path = agg / "all_burden_summary.csv"
    rand_bp_path = agg / "all_random_BP_burden_summary.csv"
    split_path = agg / "all_split_validation.csv"

    if readiness_path.exists():
        df = pd.read_csv(readiness_path, encoding="utf-8-sig")
        if not df.empty:
            summary = df.groupby(["collection", "cohort", "layer", "readiness_K"]).agg(
                n_terms=("bp_key", "count"),
                n_ready=("ready", "sum"),
                median_matched_n=("matched_n", "median"),
                median_matched_fraction=("matched_fraction", "median"),
            ).reset_index()
            summary["ready_fraction"] = summary["n_ready"] / summary["n_terms"]
            write_csv(summary, agg / "summary_true_readiness_K_sensitivity.csv")

    if selected_path.exists():
        df = pd.read_csv(selected_path, encoding="utf-8-sig")
        if not df.empty:
            primary = df[df["readiness_K"] == PRIMARY_READINESS_K].copy()

            rep_counts = primary.groupby(["endpoint", "best_representation"]).size().reset_index(name="n")
            write_csv(rep_counts, agg / "summary_best_representation_counts_primaryK.csv")

            class_counts = primary.groupby(["endpoint", "representation_class"]).size().reset_index(name="n")
            write_csv(class_counts, agg / "summary_representation_class_counts_primaryK.csv")

            gain_counts = primary.groupby(["endpoint", "gain_class"]).size().reset_index(name="n")
            write_csv(gain_counts, agg / "summary_deltaD_gain_class_counts_primaryK.csv")

            frag_cols = [
                "fragile_GE_CN", "fragile_GE_MU", "fragile_GE_CN_MU",
                "strong_fragile_GE_CN", "strong_fragile_GE_MU", "strong_fragile_GE_CN_MU",
                "signal_lost_GE_CN", "signal_lost_GE_MU", "signal_lost_GE_CN_MU",
                "any_fragile", "any_strong_fragile", "any_signal_lost",
            ]

            frag_summary = primary.groupby("endpoint")[frag_cols].sum().reset_index()
            write_csv(frag_summary, agg / "summary_per_representation_fragility_primaryK.csv")

            endpoint_info = primary.groupby("endpoint")["endpoint_informative"].agg(["sum", "count"]).reset_index()
            endpoint_info["fraction"] = endpoint_info["sum"] / endpoint_info["count"]
            write_csv(endpoint_info, agg / "summary_endpoint_informative_counts_primaryK.csv")

            fdr_summary = primary.groupby("endpoint").agg(
                n_total=("bp_key", "count"),
                n_p05=("endpoint_informative", "sum"),
                n_q10_endpoint=("endpoint_informative_q10_by_endpoint", "sum"),
                n_q05_endpoint=("endpoint_informative_q05_by_endpoint", "sum"),
            ).reset_index()
            write_csv(fdr_summary, agg / "summary_FDR_sensitivity_primaryK.csv")

            k_sens = df.groupby(["readiness_K", "endpoint"]).agg(
                n_total=("bp_key", "count"),
                n_endpoint_informative=("endpoint_informative", "sum"),
                median_delta_best_GE=("deltaD_best_minus_GE", "median"),
                median_delta_full_GE=("deltaD_GE_CN_MU_minus_GE", "median"),
                n_any_fragile=("any_fragile", "sum"),
                n_any_strong_fragile=("any_strong_fragile", "sum"),
                n_any_signal_lost=("any_signal_lost", "sum"),
            ).reset_index()
            write_csv(k_sens, agg / "summary_selected_results_by_true_readiness_K.csv")

            top_gain = primary.sort_values("deltaD_best_minus_GE", ascending=False).head(200)
            write_csv(top_gain, agg / "top200_integration_gain_primaryK.csv")

            top_fragile = primary.sort_values("deltaD_GE_CN_MU_minus_GE", ascending=True).head(200)
            write_csv(top_fragile, agg / "top200_full_integration_fragile_primaryK.csv")

    if rand_gs_path.exists():
        df = pd.read_csv(rand_gs_path, encoding="utf-8-sig")
        if not df.empty:
            summary = df.groupby(["endpoint", "best_representation"]).agg(
                n_real_BP=("bp_key", "count"),
                n_exceeds_random95=("real_exceeds_random95", "sum"),
                median_real_D=("real_best_D", "median"),
                median_random95=("random_gene_set_D_95pct", "median"),
                median_empirical_p=("empirical_p_ge_observed", "median"),
            ).reset_index()
            summary["fraction_exceeds_random95"] = summary["n_exceeds_random95"] / summary["n_real_BP"]
            write_csv(summary, agg / "summary_size_matched_random_geneset_primaryK.csv")

    if burden_path.exists():
        df = pd.read_csv(burden_path, encoding="utf-8-sig")
        if not df.empty:
            k_summary = df.groupby(["endpoint_selection", "topK", "cutoff"]).agg(
                median_OS_D=("OS_D", "median"),
                median_STAGE_D=("STAGE_D", "median"),
                n_profiles=("cohort", "count"),
                n_OS_informative=("OS_D", lambda x: np.nansum(np.array(x) >= D_THRESHOLD)),
                n_STAGE_informative=("STAGE_D", lambda x: np.nansum(np.array(x) >= D_THRESHOLD)),
            ).reset_index()
            write_csv(k_summary, agg / "summary_burden_topK_sensitivity.csv")

    if rand_bp_path.exists():
        df = pd.read_csv(rand_bp_path, encoding="utf-8-sig")
        if not df.empty:
            df["OS_exceeds_random95"] = df["observed_OS_D"] > df["random_OS_95pct"]
            df["STAGE_exceeds_random95"] = df["observed_STAGE_D"] > df["random_STAGE_95pct"]
            write_csv(df, agg / "all_random_BP_burden_summary_with_flags.csv")

            summary = df.groupby(["endpoint_selection", "cutoff"]).agg(
                n_settings=("cohort", "count"),
                n_OS_exceeds_random95=("OS_exceeds_random95", "sum"),
                n_STAGE_exceeds_random95=("STAGE_exceeds_random95", "sum"),
                median_observed_OS_D=("observed_OS_D", "median"),
                median_random_OS_95pct=("random_OS_95pct", "median"),
                median_observed_STAGE_D=("observed_STAGE_D", "median"),
                median_random_STAGE_95pct=("random_STAGE_95pct", "median"),
            ).reset_index()
            write_csv(summary, agg / "summary_random_BP_burden_flags.csv")

    if split_path.exists():
        df = pd.read_csv(split_path, encoding="utf-8-sig")
        if not df.empty:
            split_summary = df.groupby(["cohort", "endpoint_selection", "topK", "cutoff"]).agg(
                n_splits=("iteration", "count"),
                median_test_OS_D=("test_OS_D", "median"),
                q05_test_OS_D=("test_OS_D", lambda x: np.nanquantile(x, 0.05) if np.isfinite(x).any() else np.nan),
                q95_test_OS_D=("test_OS_D", lambda x: np.nanquantile(x, 0.95) if np.isfinite(x).any() else np.nan),
                frac_test_OS_D_ge_1p301=("test_OS_D", lambda x: np.nanmean(np.array(x) >= D_THRESHOLD)),
                median_test_STAGE_D=("test_STAGE_D", "median"),
                q05_test_STAGE_D=("test_STAGE_D", lambda x: np.nanquantile(x, 0.05) if np.isfinite(x).any() else np.nan),
                q95_test_STAGE_D=("test_STAGE_D", lambda x: np.nanquantile(x, 0.95) if np.isfinite(x).any() else np.nan),
                frac_test_STAGE_D_ge_1p301=("test_STAGE_D", lambda x: np.nanmean(np.array(x) >= D_THRESHOLD)),
            ).reset_index()
            write_csv(split_summary, agg / "summary_split_validation_by_cohort.csv")


def make_simple_figures(out_dir: Path) -> None:
    if not HAS_MPL or not RUN_FIGURES:
        return

    agg = out_dir / "aggregate_tables"
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    selected_path = agg / "all_selected_representations.csv"
    rand_gs_path = agg / "all_size_matched_random_geneset.csv"

    if selected_path.exists():
        df = pd.read_csv(selected_path, encoding="utf-8-sig")
        df = df[df["readiness_K"] == PRIMARY_READINESS_K].copy()

        for endpoint in ENDPOINTS:
            sub = df[(df["endpoint"] == endpoint) & df["deltaD_best_minus_GE"].notna()]
            if not sub.empty:
                plt.figure(figsize=(8, 5))
                plt.hist(sub["deltaD_best_minus_GE"], bins=40)
                plt.axvline(0, linestyle="--")
                plt.axvline(MODERATE_GAIN_DELTA, linestyle="--")
                plt.axvline(STRONG_GAIN_DELTA, linestyle="--")
                plt.xlabel("Delta D: best representation minus GE")
                plt.ylabel("Cancer-BP observations")
                plt.title(f"{endpoint}: Delta D over GE baseline")
                plt.tight_layout()
                plt.savefig(fig_dir / f"DeltaD_best_minus_GE_{endpoint}_primaryK.png", dpi=300)
                plt.close()

            for col, label in [
                ("deltaD_GE_CN_minus_GE", "GE+CN minus GE"),
                ("deltaD_GE_MU_minus_GE", "GE+MU minus GE"),
                ("deltaD_GE_CN_MU_minus_GE", "GE+CN+MU minus GE"),
            ]:
                sub2 = df[(df["endpoint"] == endpoint) & df[col].notna()]
                if not sub2.empty:
                    plt.figure(figsize=(8, 5))
                    plt.hist(sub2[col], bins=40)
                    plt.axvline(0, linestyle="--")
                    plt.axvline(STRONG_FRAGILITY_DELTA, linestyle="--")
                    plt.xlabel(f"Delta D: {label}")
                    plt.ylabel("Cancer-BP observations")
                    plt.title(f"{endpoint}: {label}")
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"{safe_name(col)}_{endpoint}_primaryK.png", dpi=300)
                    plt.close()

    if rand_gs_path.exists():
        df = pd.read_csv(rand_gs_path, encoding="utf-8-sig")
        for endpoint in ENDPOINTS:
            sub = df[(df["endpoint"] == endpoint) & df["real_best_D"].notna() & df["random_gene_set_D_95pct"].notna()]
            if not sub.empty:
                plt.figure(figsize=(6, 6))
                plt.scatter(sub["random_gene_set_D_95pct"], sub["real_best_D"])
                lim = max(sub["random_gene_set_D_95pct"].max(), sub["real_best_D"].max()) + 0.5
                plt.plot([0, lim], [0, lim], linestyle="--")
                plt.xlabel(f"Random gene-set 95th percentile {endpoint} D")
                plt.ylabel(f"Real BP best {endpoint} D")
                plt.title(f"Size-matched random gene-set baseline: {endpoint}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"Size_matched_random_geneset_{endpoint}_primaryK.png", dpi=300)
                plt.close()


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    start = time.time()

    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "cohort_outputs")
    ensure_dir(OUT_DIR / "aggregate_tables")
    ensure_dir(OUT_DIR / "figures")

    config = {
        "BASE_DIR": str(BASE_DIR),
        "GSEA_DIR": str(GSEA_DIR),
        "OUT_DIR": str(OUT_DIR),
        "GMT_FILES": {k: str(v) for k, v in GMT_FILES.items()},
        "PRIMARY_GENESET_COLLECTION": PRIMARY_GENESET_COLLECTION,
        "READINESS_K_VALUES": READINESS_K_VALUES,
        "PRIMARY_READINESS_K": PRIMARY_READINESS_K,
        "BURDEN_TOPK_VALUES": BURDEN_TOPK_VALUES,
        "PRIMARY_BURDEN_TOPK": PRIMARY_BURDEN_TOPK,
        "D_THRESHOLD": D_THRESHOLD,
        "STRONG_FRAGILITY_DELTA": STRONG_FRAGILITY_DELTA,
        "MODERATE_GAIN_DELTA": MODERATE_GAIN_DELTA,
        "STRONG_GAIN_DELTA": STRONG_GAIN_DELTA,
        "RANDOM_GENESET_ITER": RANDOM_GENESET_ITER,
        "RANDOM_BP_BURDEN_ITER": RANDOM_BP_BURDEN_ITER,
        "SPLIT_ITER": SPLIT_ITER,
        "TRAIN_FRAC": TRAIN_FRAC,
        "RANDOM_SEED": RANDOM_SEED,
        "REPRESENTATIONS": REPRESENTATIONS,
        "DEFAULT_EXCLUDE_CODES": sorted(DEFAULT_EXCLUDE_CODES),
        "COHORT_WHITELIST": COHORT_WHITELIST,
        "HAS_STATSMODELS": HAS_STATSMODELS,
        "HAS_LIFELINES": HAS_LIFELINES,
        "RUN_BURDEN_MODULE": RUN_BURDEN_MODULE,
        "RUN_RANDOM_GENESET_BASELINE": RUN_RANDOM_GENESET_BASELINE,
        "RUN_RANDOM_BP_BURDEN_BASELINE": RUN_RANDOM_BP_BURDEN_BASELINE,
        "RUN_SPLIT_VALIDATION": RUN_SPLIT_VALIDATION,
        "RUN_FIGURES": RUN_FIGURES,
        "v3_major_upgrades": [
            "true matched-gene readiness K sensitivity",
            "per-representation Delta-D and fragility",
            "BH-FDR q-values by endpoint and endpoint+cohort",
            "size-matched random gene-set baseline",
            "retained fixed burden evaluation without second median split",
            "retained clinical suffix ID remapping",
        ],
    }
    write_json(config, OUT_DIR / "run_config.json")

    log("AIDO-Multi-Omics-I-4.0 internal benchmark v3 NARGAB-UPGRADE started")
    log(f"Output directory: {OUT_DIR}")

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")

    all_gene_sets = load_gene_set_collections(GMT_FILES)
    primary_sets = subset_gene_sets(all_gene_sets, PRIMARY_GENESET_COLLECTION)
    if not primary_sets:
        raise RuntimeError(f"No gene sets found for PRIMARY_GENESET_COLLECTION={PRIMARY_GENESET_COLLECTION}")
    log(f"Primary collection {PRIMARY_GENESET_COLLECTION}: {len(primary_sets)} gene sets")

    cohorts = discover_cohorts(BASE_DIR)
    write_csv(cohorts, OUT_DIR / "discovered_cohorts.csv")
    if cohorts.empty:
        raise RuntimeError("No cohorts discovered.")

    statuses = []
    for _, row in cohorts.iterrows():
        cohort = row["cohort"]
        folder = Path(row["path"])

        try:
            status = process_one_cohort(
                cohort=cohort,
                folder=folder,
                all_gene_sets=all_gene_sets,
                out_dir=OUT_DIR,
            )
        except Exception as e:
            log(f"ERROR processing {cohort}: {e}")
            status = {"cohort": cohort, "status": "error", "error": str(e)}

        statuses.append(status)
        write_csv(pd.DataFrame(statuses), OUT_DIR / "cohort_status_running.csv")

    cohort_status = pd.DataFrame(statuses)
    write_csv(cohort_status, OUT_DIR / "cohort_status_summary.csv")

    log("Aggregating outputs")
    aggregate_outputs(OUT_DIR, cohort_status)

    log("Making summary tables")
    make_summary_tables(OUT_DIR)

    log("Making simple figures")
    make_simple_figures(OUT_DIR)

    elapsed = time.time() - start
    final_report = {
        "output_dir": str(OUT_DIR),
        "elapsed_minutes": elapsed / 60,
        "n_cohorts_discovered": int(cohorts.shape[0]),
        "n_cohorts_completed": int((cohort_status["status"] == "completed").sum()) if "status" in cohort_status else 0,
        "n_cohorts_error": int((cohort_status["status"] == "error").sum()) if "status" in cohort_status else 0,
        "note": "All CSV outputs use UTF-8-SIG encoding for Excel compatibility.",
    }
    write_json(final_report, OUT_DIR / "final_report.json")

    log(f"Completed in {elapsed / 60:.2f} minutes")
    log("AIDO-Multi-Omics-I-4.0 internal benchmark v3 NARGAB-UPGRADE finished")


if __name__ == "__main__":
    main()
