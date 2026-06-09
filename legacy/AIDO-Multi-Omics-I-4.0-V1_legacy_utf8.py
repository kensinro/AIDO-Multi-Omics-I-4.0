# -*- coding: utf-8 -*-
"""
AIDO-Multi-Omics-I-4.0
Internal benchmark pipeline for NARGAB upgrade

Main goals
----------
1. Recalculate Hallmark BP scores for GE, CN, MU.
2. Evaluate GE, GE+CN, GE+MU, GE+CN+MU representations.
3. Calculate endpoint-specific D = -log10(p) for OS and STAGE.
4. Quantify Delta-D over GE baseline.
5. Quantify integration fragility.
6. Construct patient-level risk-aligned BP-burden profiles.
7. Run K sensitivity analysis.
8. Run all-cohort random BP-burden baseline.
9. Run repeated train/test split validation.
10. Export summary tables for NARGAB manuscript rebuilding.

Important design choices
------------------------
Input files may be UTF-8, UTF-8-SIG, UTF-16, UTF-16-LE, or Latin1.
This script uses read_table_auto() for robust encoding/separator detection.

Expected input structure
------------------------
D:/AIDO-Data/UCSC_XENA/
    Breast Cancer (BRCA)/
        GE.tsv
        CN.tsv
        MU.tsv or MU_fixed.tsv
        Phenotype.tsv
        TCGA.BRCA.sampleMap_BRCA_clinicalMatrix
        optional: BRCA_stage_groups_from_survival.tsv
    ...

D:/AIDO-Data/GSEA/
    h.all.v2026.1.Hs.symbols.gmt

Output
------
D:/AIDO-Temp/AIDO_MultiOmics_I_4_internal_benchmark_<timestamp>/
"""

from __future__ import annotations

import os
import re
import gc
import math
import time
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

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


# =============================================================================
# 0. CONFIG
# =============================================================================

BASE_DIR = Path("D:/AIDO-Data/UCSC_XENA")
GSEA_DIR = Path("D:/AIDO-Data/GSEA")
OUT_ROOT = Path("D:/AIDO-Temp")

HALLMARK_GMT = GSEA_DIR / "h.all.v2026.1.Hs.symbols.gmt"

MIN_MATCHED_GENES = 10
D_THRESHOLD = -math.log10(0.05)

PRIMARY_K = 8
K_VALUES = [5, 8, 10, 15]

RANDOM_ITER = 300
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
ABNORMALITY_CUTOFFS = ["SD1", "EXT20"]

DEFAULT_EXCLUDE_CODES = {"COADREAD", "LUNG", "READ"}

# Set to None for all cohorts.
# Example: COHORT_WHITELIST = ["BRCA", "SKCM", "KIRC"]
COHORT_WHITELIST: Optional[List[str]] = None

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / f"AIDO_MultiOmics_I_4_internal_benchmark_{TIMESTAMP}"

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


def safe_name(x: Any) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s)
    if m:
        return m.group(1)
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return s if s else None


def read_table_auto(path: Path, index_col: Optional[int] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Robust table reader for mixed UTF-8 / UTF-16 / UTF-8-SIG / Latin1 files.

    Designed for UCSC Xena / TCGA files where:
    - Some files are UTF-8.
    - Some files are UTF-16.
    - Some files have BOM.
    - Most files are tab-separated.
    - Some files may be comma-separated.
    - File extensions are not always reliable.
    """
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-8", "latin1"]
    seps = ["\t", ","]

    last_err = None
    best_df = None
    best_score = -10**9
    best_info = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    index_col=index_col,
                    low_memory=False,
                    nrows=nrows,
                )

                if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                    continue

                col_text = " ".join([str(c) for c in df.columns[:20]])
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
                continue

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

def load_gmt(gmt_path: Path) -> Dict[str, List[str]]:
    if not gmt_path.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt_path}")

    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = sorted(set(g.strip().upper() for g in parts[2:] if g.strip()))
            gene_sets[name] = genes

    return gene_sets


# =============================================================================
# 4. MOLECULAR MATRICES
# =============================================================================

def normalize_matrix_gene_by_patient(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    """
    Convert common TCGA/Xena formats into genes x patients matrix.

    GE/CN expected:
        gene rows x sample columns, first column gene symbol.

    MU expected:
        either gene rows x sample columns,
        or long mutation table with sample and gene columns.
    """
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    lower_cols = {c.lower(): c for c in df.columns}

    # Long mutation table
    if layer == "MU" and ("sample" in lower_cols) and ("gene" in lower_cols):
        sample_col = lower_cols["sample"]
        gene_col = lower_cols["gene"]

        tmp = df[[sample_col, gene_col]].dropna()
        tmp["patient"] = tmp[sample_col].map(tcga_patient_id)
        tmp["gene"] = tmp[gene_col].astype(str).str.upper().str.strip()
        tmp = tmp.dropna(subset=["patient", "gene"])
        tmp["value"] = 1.0

        mat = tmp.drop_duplicates(["gene", "patient"]).pivot_table(
            index="gene",
            columns="patient",
            values="value",
            aggfunc="max",
            fill_value=0.0,
        )
        mat.index = mat.index.astype(str).str.upper()
        mat.columns = [tcga_patient_id(c) for c in mat.columns]
        mat = mat.loc[:, [c is not None for c in mat.columns]]
        return mat.astype(float)

    # Detect gene column
    first_col = df.columns[0]
    first_col_lower = first_col.lower()

    if first_col_lower in ["gene", "genes", "symbol", "hugo_symbol", "id", "name", "sample"]:
        df = df.set_index(first_col)
    else:
        numeric_ratio = pd.to_numeric(df.iloc[:20, 0], errors="coerce").notna().mean()
        if numeric_ratio < 0.5:
            df = df.set_index(first_col)

    # Drop common annotation columns
    drop_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if cl in ["description", "gene_id", "entrez", "chrom", "chr", "start", "end", "reference", "alt", "effect"]:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    numeric = df.apply(pd.to_numeric, errors="coerce")

    new_cols = [tcga_patient_id(c) for c in numeric.columns]
    numeric.columns = new_cols
    numeric = numeric.loc[:, [c is not None and str(c) != "" for c in numeric.columns]]

    # Merge duplicated patient columns
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
    log(f"Loaded {layer}: {path.name}, matrix shape genes x patients = {mat.shape}")
    return mat


# =============================================================================
# 5. CLINICAL PARSING
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

    df["patient"] = df[id_col].map(tcga_patient_id)
    df = df.dropna(subset=["patient"])
    df = df.drop_duplicates("patient", keep="first")
    df = df.set_index("patient", drop=True)
    return df


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
    s2 = s2.replace("STAGE", "")
    s2 = s2.replace("PATHOLOGIC", "")
    s2 = s2.replace("CLINICAL", "")
    s2 = s2.replace("AJCC", "")
    s2 = s2.replace("TUMOR", "")
    s2 = s2.replace(" ", "")
    s2 = s2.replace("_", "")

    if s2.startswith("IV") or re.search(r"\bIV\b", s2):
        return "ADVANCED"
    if s2.startswith("III") or re.search(r"\bIII\b", s2):
        return "ADVANCED"
    if s2.startswith("II") or re.search(r"\bII\b", s2):
        return "EARLY"
    if s2.startswith("I") or re.search(r"\bI\b", s2):
        return "EARLY"

    return None


def load_clinical(files: Dict[str, Optional[Path]]) -> pd.DataFrame:
    dfs = []

    for key in ["PHENO", "CLINICAL", "STAGE_GROUP"]:
        p = files.get(key)
        if p is not None and p.exists():
            try:
                d = standardize_clinical_table(read_table_auto(p))
                dfs.append(d)
                log(f"Clinical source loaded: {key} | {p.name} | shape={d.shape}")
            except Exception as e:
                log(f"Warning: failed to read clinical source {key}: {p} | {e}")

    if not dfs:
        return pd.DataFrame()

    clin = dfs[0].copy()
    for d in dfs[1:]:
        clin = clin.combine_first(d)

    out = pd.DataFrame(index=clin.index)

    time_col = find_first_matching_col(
        clin,
        [
            "OS.time", "OS_time", "OS.time.days",
            "days_to_death", "days_to_last_followup", "days_to_last_follow_up",
            "DSS.time", "PFI.time", "time"
        ]
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
        [
            "stage_group", "stage_binary", "TNM_stage_group",
            "pathologic_stage", "ajcc_pathologic_tumor_stage",
            "clinical_stage", "tumor_stage", "stage"
        ]
    )

    if stage_col is not None:
        out["stage_group"] = clin[stage_col].map(parse_stage_group)
    else:
        out["stage_group"] = None

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

    return out


# =============================================================================
# 6. BP SCORE CONSTRUCTION
# =============================================================================

def construct_layer_bp_scores(
    mat_gene_by_patient: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
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

    for bp, genes in gene_sets.items():
        matched = sorted(set(g.upper() for g in genes).intersection(measured))
        ready = len(matched) >= min_genes

        readiness_rows.append({
            "bp": bp,
            "layer": layer,
            "matched_n": len(matched),
            "ready": int(ready),
        })

        if not ready:
            continue

        sub = zmat.loc[matched]
        raw = sub.mean(axis=0, skipna=True)
        scores[bp] = zscore_series(raw)

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

    O1 = 0.0
    E1 = 0.0
    V1 = 0.0

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
    df["high"] = (df["score"] >= med).astype(int)

    _, p = logrank_test_basic(df["time"].values, df["event"].values, df["high"].values)
    D = neglog10_p(p)

    hr = np.nan
    cidx = np.nan

    if HAS_LIFELINES:
        try:
            cph = CoxPHFitter()
            cph.fit(df[["time", "event", "high"]], duration_col="time", event_col="event")
            hr = float(np.exp(cph.params_["high"]))
        except Exception:
            pass

        try:
            cidx = float(concordance_index(df["time"], df["score"], df["event"]))
        except Exception:
            pass

    return {
        "p": p,
        "D": D,
        "n": int(df.shape[0]),
        "events": int(df["event"].sum()),
        "hr": hr,
        "c_index": cidx,
    }


def stage_test_from_score(score: pd.Series, clinical: pd.DataFrame) -> Dict[str, Any]:
    common = sorted(set(score.dropna().index).intersection(clinical.index))

    if len(common) < 30 or "stage_group" not in clinical.columns:
        return {"p": np.nan, "D": np.nan, "n": len(common), "OR": np.nan, "AUC": np.nan}

    df = pd.DataFrame({
        "score": score.loc[common],
        "stage": clinical.loc[common, "stage_group"],
    }).dropna()

    df = df[df["stage"].isin(["EARLY", "ADVANCED"])]

    if df.shape[0] < 30 or df["stage"].nunique() != 2:
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan}

    med = df["score"].median()
    df["high"] = (df["score"] >= med).astype(int)
    df["advanced"] = (df["stage"] == "ADVANCED").astype(int)

    tab = pd.crosstab(df["high"], df["advanced"])

    if tab.shape != (2, 2):
        return {"p": np.nan, "D": np.nan, "n": df.shape[0], "OR": np.nan, "AUC": np.nan}

    try:
        if (tab.values < 5).any():
            OR, p = fisher_exact(tab.values)
        else:
            chi, p, _, _ = chi2_contingency(tab.values)
            a, b = tab.iloc[1, 1], tab.iloc[1, 0]
            c, d = tab.iloc[0, 1], tab.iloc[0, 0]
            OR = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    except Exception:
        p = np.nan
        OR = np.nan

    D = neglog10_p(p)

    # rank-based AUC from Mann-Whitney U
    AUC = np.nan
    try:
        pos = df.loc[df["advanced"] == 1, "score"]
        neg = df.loc[df["advanced"] == 0, "score"]
        if len(pos) > 0 and len(neg) > 0:
            U, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            AUC = float(U / (len(pos) * len(neg)))
            AUC = max(AUC, 1 - AUC)
    except Exception:
        pass

    return {
        "p": p,
        "D": D,
        "n": int(df.shape[0]),
        "OR": OR,
        "AUC": AUC,
    }


# =============================================================================
# 8. REPRESENTATION MAPS
# =============================================================================

def evaluate_representation_maps(
    cohort: str,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    all_bps = sorted(set().union(*[set(df.columns) for df in rep_scores.values()]))

    for bp in all_bps:
        for rep, score_df in rep_scores.items():
            if bp not in score_df.columns:
                continue

            score = score_df[bp]

            os_res = survival_test_from_score(score, clinical)
            rows.append({
                "cohort": cohort,
                "bp": bp,
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
            })

            st_res = stage_test_from_score(score, clinical)
            rows.append({
                "cohort": cohort,
                "bp": bp,
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
            })

    return pd.DataFrame(rows)


def select_best_representations(rep_map: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (cohort, bp, endpoint), sub in rep_map.groupby(["cohort", "bp", "endpoint"]):
        sub2 = sub.dropna(subset=["D"])
        if sub2.empty:
            continue

        d_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}
        p_by_rep = {r: np.nan for r in REPRESENTATIONS.keys()}
        for _, row in sub2.iterrows():
            d_by_rep[row["representation"]] = row["D"]
            p_by_rep[row["representation"]] = row["p"]

        best_idx = sub2["D"].idxmax()
        best = sub2.loc[best_idx]

        D_ge = d_by_rep.get("GE", np.nan)
        D_full = d_by_rep.get("GE_CN_MU", np.nan)
        D_best = best["D"]

        delta_best_ge = D_best - D_ge if pd.notna(D_best) and pd.notna(D_ge) else np.nan
        delta_full_ge = D_full - D_ge if pd.notna(D_full) and pd.notna(D_ge) else np.nan
        delta_best_full = D_best - D_full if pd.notna(D_best) and pd.notna(D_full) else np.nan

        endpoint_informative = int(pd.notna(D_best) and D_best >= D_THRESHOLD)

        if endpoint_informative == 0:
            rep_class = "endpoint_weak"
        elif best["representation"] == "GE":
            rep_class = "GE_sufficient"
        elif best["representation"] == "GE_CN":
            rep_class = "CN_informative"
        elif best["representation"] == "GE_MU":
            rep_class = "MU_informative"
        elif best["representation"] == "GE_CN_MU":
            rep_class = "multi_layer_informative"
        else:
            rep_class = "other"

        integration_fragile = int(
            pd.notna(D_full)
            and pd.notna(D_ge)
            and D_full < D_ge
            and endpoint_informative == 1
        )

        if integration_fragile:
            fragility_class = "full_integration_worse_than_GE"
        elif pd.notna(D_full) and pd.notna(D_ge) and D_full >= D_ge:
            fragility_class = "full_integration_not_worse_than_GE"
        else:
            fragility_class = "not_evaluable"

        gain_class = "not_evaluable"
        if pd.notna(delta_best_ge):
            if delta_best_ge >= 1.0:
                gain_class = "strong_gain"
            elif delta_best_ge >= 0.5:
                gain_class = "moderate_gain"
            elif delta_best_ge > 0:
                gain_class = "weak_gain"
            else:
                gain_class = "no_gain"

        rows.append({
            "cohort": cohort,
            "bp": bp,
            "endpoint": endpoint,
            "best_representation": best["representation"],
            "best_D": D_best,
            "best_p": best["p"],
            "D_GE": D_ge,
            "D_GE_CN": d_by_rep.get("GE_CN", np.nan),
            "D_GE_MU": d_by_rep.get("GE_MU", np.nan),
            "D_GE_CN_MU": D_full,
            "deltaD_best_minus_GE": delta_best_ge,
            "deltaD_full_minus_GE": delta_full_ge,
            "deltaD_best_minus_full": delta_best_full,
            "endpoint_informative": endpoint_informative,
            "representation_class": rep_class,
            "integration_fragile": integration_fragile,
            "fragility_class": fragility_class,
            "gain_class": gain_class,
            "n": best.get("n", np.nan),
            "events": best.get("events", np.nan),
            "HR": best.get("HR", np.nan),
            "C_index": best.get("C_index", np.nan),
            "OR": best.get("OR", np.nan),
            "AUC": best.get("AUC", np.nan),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 9. BP BURDEN
# =============================================================================

def infer_risk_direction(score: pd.Series, clinical: pd.DataFrame) -> int:
    """
    Return +1 for high-score-risk, -1 for low-score-risk.
    Uses OS event fraction, then mean OS time as tie-breaker.
    """
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


def abnormal_indicator(score: pd.Series, direction: int, cutoff: str, reference_score: Optional[pd.Series] = None) -> pd.Series:
    """
    Construct risk-aligned abnormality.

    cutoff:
    - SD1: score >= 1 for high-risk direction; score <= -1 for low-risk direction.
    - EXT20: upper 20% or lower 20% based on reference_score.
    """
    s = score.copy()

    if reference_score is None:
        ref = s.dropna()
    else:
        ref = reference_score.dropna()

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
    endpoint: str,
    selected_table: pd.DataFrame,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    K: int,
    cutoff: str,
    patient_subset: Optional[List[str]] = None,
    training_reference_patients: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct BP burden using top K selected endpoint-informative BPs.

    If patient_subset is provided, burden is returned for those patients.
    If training_reference_patients is provided, risk direction and EXT20 thresholds
    are inferred from training/reference patients.
    """
    sub = selected_table[
        (selected_table["cohort"] == cohort)
        & (selected_table["endpoint"] == endpoint)
        & (selected_table["endpoint_informative"] == 1)
    ].copy()

    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = sub.sort_values("best_D", ascending=False).head(K)

    selected_rows = []
    abnormal_cols = []
    common_patients = None

    for _, row in sub.iterrows():
        bp = row["bp"]
        rep = row["best_representation"]

        if rep not in rep_scores or bp not in rep_scores[rep].columns:
            continue

        score = rep_scores[rep][bp].dropna()

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

        abnormal_cols.append(abn.rename(bp))

        selected_rows.append({
            "cohort": cohort,
            "endpoint": endpoint,
            "K": K,
            "cutoff": cutoff,
            "bp": bp,
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
        "endpoint_selection": endpoint,
        "K": K,
        "cutoff": cutoff,
        "burden": burden.values,
    })

    high_cutoff = burden.quantile(0.67)
    burden_df["high_burden"] = (burden_df["burden"] >= high_cutoff).astype(int)
    burden_df["burden_cutoff_q67"] = high_cutoff

    selected_df = pd.DataFrame(selected_rows)
    return burden_df, selected_df


def evaluate_burden(burden_df: pd.DataFrame, clinical: pd.DataFrame) -> Dict[str, Any]:
    if burden_df.empty:
        return {
            "OS_p": np.nan, "OS_D": np.nan, "OS_HR": np.nan, "OS_C_index": np.nan,
            "STAGE_p": np.nan, "STAGE_D": np.nan, "STAGE_OR": np.nan, "STAGE_AUC": np.nan,
            "n_OS": np.nan, "events": np.nan, "n_STAGE": np.nan,
        }

    b = burden_df.set_index("patient")
    score = b["high_burden"]

    os_res = survival_test_from_score(score, clinical)
    st_res = stage_test_from_score(score, clinical)

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


# =============================================================================
# 10. RANDOM BASELINE
# =============================================================================

def random_bp_burden_baseline(
    cohort: str,
    endpoint: str,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    candidate_bps: List[str],
    K: int,
    cutoff: str,
    n_iter: int,
) -> pd.DataFrame:
    rows = []

    candidate_bps = sorted(set(candidate_bps))
    if len(candidate_bps) < K:
        return pd.DataFrame()

    # Use GE as random baseline representation if available.
    if "GE" not in rep_scores:
        return pd.DataFrame()

    ge_scores = rep_scores["GE"]
    candidate_bps = [bp for bp in candidate_bps if bp in ge_scores.columns]

    if len(candidate_bps) < K:
        return pd.DataFrame()

    for t in range(n_iter):
        sampled = random.sample(candidate_bps, K)

        abnormal_cols = []
        common_patients = None

        for bp in sampled:
            score = ge_scores[bp].dropna()
            patients = sorted(set(score.index).intersection(clinical.index))

            if len(patients) < 20:
                continue

            direction = infer_risk_direction(score.loc[patients], clinical.loc[patients])
            abn = abnormal_indicator(score.loc[patients], direction=direction, cutoff=cutoff)

            abnormal_cols.append(abn.rename(bp))
            common_patients = set(patients) if common_patients is None else common_patients.intersection(patients)

        if not abnormal_cols or common_patients is None or len(common_patients) < 20:
            continue

        common_patients = sorted(common_patients)
        abn_df = pd.concat(abnormal_cols, axis=1).loc[common_patients]
        burden = abn_df.sum(axis=1)

        bdf = pd.DataFrame({
            "patient": burden.index,
            "burden": burden.values,
        })
        cutoff_q = burden.quantile(0.67)
        bdf["high_burden"] = (bdf["burden"] >= cutoff_q).astype(int)

        eval_res = evaluate_burden(
            pd.DataFrame({
                "patient": bdf["patient"],
                "cohort": cohort,
                "endpoint_selection": endpoint,
                "K": K,
                "cutoff": cutoff,
                "burden": bdf["burden"],
                "high_burden": bdf["high_burden"],
            }),
            clinical,
        )

        rows.append({
            "cohort": cohort,
            "endpoint_selection": endpoint,
            "K": K,
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
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    train_patients: List[str],
) -> pd.DataFrame:
    clinical_train = clinical.loc[train_patients].copy()

    rep_scores_train = {}
    for rep, df in rep_scores.items():
        pats = sorted(set(train_patients).intersection(df.index))
        if len(pats) >= 20:
            rep_scores_train[rep] = df.loc[pats].copy()

    rep_map_train = evaluate_representation_maps(cohort, rep_scores_train, clinical_train)
    selected_train = select_best_representations(rep_map_train)
    return selected_train


def split_validation(
    cohort: str,
    rep_scores: Dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    K: int,
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
        random.shuffle(available_patients)
        n_train = int(len(available_patients) * TRAIN_FRAC)
        train = sorted(available_patients[:n_train])
        test = sorted(available_patients[n_train:])

        if len(train) < 40 or len(test) < 20:
            continue

        try:
            selected_train = train_selected_representations(cohort, rep_scores, clinical, train)
        except Exception as e:
            log(f"Split train selection failed: {cohort} iter {t + 1}: {e}")
            continue

        for endpoint in ENDPOINTS:
            for ctf in [cutoff]:
                bdf_test, selected_bps = construct_bp_burden(
                    cohort=cohort,
                    endpoint=endpoint,
                    selected_table=selected_train,
                    rep_scores=rep_scores,
                    clinical=clinical,
                    K=K,
                    cutoff=ctf,
                    patient_subset=test,
                    training_reference_patients=train,
                )

                eval_res = evaluate_burden(bdf_test, clinical)

                rows.append({
                    "cohort": cohort,
                    "iteration": t + 1,
                    "endpoint_selection": endpoint,
                    "K": K,
                    "cutoff": ctf,
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
# 12. COHORT PIPELINE
# =============================================================================

def process_one_cohort(
    cohort: str,
    folder: Path,
    gene_sets: Dict[str, List[str]],
    out_dir: Path,
) -> Dict[str, Any]:
    log("=" * 80)
    log(f"Processing cohort {cohort}: {folder}")

    cohort_out = out_dir / "cohort_outputs" / cohort
    ensure_dir(cohort_out)

    files = get_cohort_files(folder, cohort)

    file_report = {k: str(v) if v is not None else None for k, v in files.items()}
    write_json(file_report, cohort_out / f"{cohort}_input_files.json")

    if cohort in DEFAULT_EXCLUDE_CODES:
        log(f"Skipping {cohort}: in DEFAULT_EXCLUDE_CODES")
        return {"cohort": cohort, "status": "skipped_default_exclude"}

    if files["GE"] is None or files["CN"] is None or files["MU"] is None:
        log(f"Skipping {cohort}: missing GE/CN/MU file")
        return {"cohort": cohort, "status": "skipped_missing_molecular"}

    clinical = load_clinical(files)
    if clinical.empty:
        log(f"Skipping {cohort}: no clinical data")
        return {"cohort": cohort, "status": "skipped_missing_clinical"}

    write_csv(clinical.reset_index().rename(columns={"index": "patient"}), cohort_out / f"{cohort}_clinical_parsed.csv")

    layer_scores = {}
    readiness_all = []

    for layer in ["GE", "CN", "MU"]:
        try:
            mat = load_layer_matrix(files[layer], layer)
            score_df, ready_df = construct_layer_bp_scores(
                mat,
                gene_sets=gene_sets,
                layer=layer,
                min_genes=MIN_MATCHED_GENES,
            )
            layer_scores[layer] = score_df
            ready_df["cohort"] = cohort
            readiness_all.append(ready_df)

            write_csv(ready_df, cohort_out / f"{cohort}_{layer}_readiness.csv")
            write_csv(score_df.reset_index(), cohort_out / f"{cohort}_{layer}_BP_scores.csv")

            del mat
            gc.collect()

        except Exception as e:
            log(f"Failed layer {layer} for {cohort}: {e}")

    if not all(layer in layer_scores and not layer_scores[layer].empty for layer in ["GE", "CN", "MU"]):
        log(f"Skipping {cohort}: failed to construct all layer scores")
        return {"cohort": cohort, "status": "skipped_failed_layer_scores"}

    readiness = pd.concat(readiness_all, ignore_index=True) if readiness_all else pd.DataFrame()
    write_csv(readiness, cohort_out / f"{cohort}_readiness_all_layers.csv")

    rep_scores = construct_representation_scores(layer_scores, REPRESENTATIONS)

    if "GE" not in rep_scores:
        log(f"Skipping {cohort}: no GE representation")
        return {"cohort": cohort, "status": "skipped_no_GE_rep"}

    for rep, df in rep_scores.items():
        write_csv(df.reset_index(), cohort_out / f"{cohort}_{rep}_representation_scores.csv")

    rep_map = evaluate_representation_maps(cohort, rep_scores, clinical)
    selected = select_best_representations(rep_map)

    write_csv(rep_map, cohort_out / f"{cohort}_representation_map_all_reps.csv")
    write_csv(selected, cohort_out / f"{cohort}_selected_representations_deltaD.csv")

    # Primary burden and K sensitivity
    burden_rows = []
    selected_bp_rows = []

    for K in K_VALUES:
        for endpoint in ENDPOINTS:
            for cutoff in ABNORMALITY_CUTOFFS:
                bdf, sdf = construct_bp_burden(
                    cohort=cohort,
                    endpoint=endpoint,
                    selected_table=selected,
                    rep_scores=rep_scores,
                    clinical=clinical,
                    K=K,
                    cutoff=cutoff,
                )

                if not bdf.empty:
                    eval_res = evaluate_burden(bdf, clinical)
                    burden_rows.append({
                        "cohort": cohort,
                        "endpoint_selection": endpoint,
                        "K": K,
                        "cutoff": cutoff,
                        "n_patients": bdf.shape[0],
                        "mean_burden": bdf["burden"].mean(),
                        "median_burden": bdf["burden"].median(),
                        "high_burden_fraction": bdf["high_burden"].mean(),
                        **eval_res,
                    })

                    write_csv(bdf, cohort_out / f"{cohort}_burden_{endpoint}_K{K}_{cutoff}.csv")

                if not sdf.empty:
                    selected_bp_rows.append(sdf)

    burden_summary = pd.DataFrame(burden_rows)
    selected_bp_summary = pd.concat(selected_bp_rows, ignore_index=True) if selected_bp_rows else pd.DataFrame()

    write_csv(burden_summary, cohort_out / f"{cohort}_burden_K_sensitivity_summary.csv")
    if not selected_bp_summary.empty:
        write_csv(selected_bp_summary, cohort_out / f"{cohort}_selected_BP_for_burden_allK.csv")

    # Random baseline: primary K only
    random_rows = []
    observed_lookup = burden_summary[
        (burden_summary["K"] == PRIMARY_K)
    ].copy()

    candidate_bps = sorted(rep_scores["GE"].columns)

    for endpoint in ENDPOINTS:
        for cutoff in ABNORMALITY_CUTOFFS:
            log(f"{cohort}: random baseline endpoint={endpoint}, cutoff={cutoff}, K={PRIMARY_K}")
            rand_df = random_bp_burden_baseline(
                cohort=cohort,
                endpoint=endpoint,
                rep_scores=rep_scores,
                clinical=clinical,
                candidate_bps=candidate_bps,
                K=PRIMARY_K,
                cutoff=cutoff,
                n_iter=RANDOM_ITER,
            )

            if not rand_df.empty:
                write_csv(rand_df, cohort_out / f"{cohort}_random_baseline_{endpoint}_K{PRIMARY_K}_{cutoff}.csv")

                obs_sub = observed_lookup[
                    (observed_lookup["endpoint_selection"] == endpoint)
                    & (observed_lookup["cutoff"] == cutoff)
                ]

                obs_os = obs_sub["OS_D"].iloc[0] if not obs_sub.empty else np.nan
                obs_stage = obs_sub["STAGE_D"].iloc[0] if not obs_sub.empty else np.nan

                rand_os = rand_df["random_OS_D"].dropna()
                rand_stage = rand_df["random_STAGE_D"].dropna()

                random_rows.append({
                    "cohort": cohort,
                    "endpoint_selection": endpoint,
                    "K": PRIMARY_K,
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

    random_summary = pd.DataFrame(random_rows)
    write_csv(random_summary, cohort_out / f"{cohort}_random_baseline_summary.csv")

    # Split validation: primary K, EXT20 only to control runtime
    log(f"{cohort}: split validation K={PRIMARY_K}, cutoff=EXT20")
    split_df = split_validation(
        cohort=cohort,
        rep_scores=rep_scores,
        clinical=clinical,
        K=PRIMARY_K,
        cutoff="EXT20",
        n_iter=SPLIT_ITER,
    )
    write_csv(split_df, cohort_out / f"{cohort}_split_validation_K{PRIMARY_K}_EXT20.csv")

    status = {
        "cohort": cohort,
        "status": "completed",
        "n_clinical": int(clinical.shape[0]),
        "n_OS_available": int(clinical[["OS_time", "OS_event"]].dropna().shape[0]),
        "n_OS_events": int(pd.to_numeric(clinical["OS_event"], errors="coerce").fillna(0).sum()),
        "n_stage_available": int(clinical["stage_group"].isin(["EARLY", "ADVANCED"]).sum()),
        "n_rep_map_rows": int(rep_map.shape[0]),
        "n_selected_rows": int(selected.shape[0]),
        "n_burden_rows": int(burden_summary.shape[0]),
        "n_random_summary_rows": int(random_summary.shape[0]),
        "n_split_rows": int(split_df.shape[0]) if not split_df.empty else 0,
    }

    write_json(status, cohort_out / f"{cohort}_status.json")

    del layer_scores, rep_scores, rep_map, selected, burden_summary, random_summary, split_df
    gc.collect()

    return status


# =============================================================================
# 13. AGGREGATION AND FIGURES
# =============================================================================

def aggregate_outputs(out_dir: Path, cohort_status: pd.DataFrame) -> None:
    cohort_root = out_dir / "cohort_outputs"

    patterns = {
        "all_representation_maps": "*_representation_map_all_reps.csv",
        "all_selected_representations": "*_selected_representations_deltaD.csv",
        "all_burden_K_sensitivity": "*_burden_K_sensitivity_summary.csv",
        "all_random_baseline_summary": "*_random_baseline_summary.csv",
        "all_split_validation": "*_split_validation_K8_EXT20.csv",
        "all_readiness": "*_readiness_all_layers.csv",
    }

    aggregate_dir = out_dir / "aggregate_tables"
    ensure_dir(aggregate_dir)

    for name, pattern in patterns.items():
        dfs = []
        for p in cohort_root.glob(f"*/{pattern}"):
            try:
                dfs.append(pd.read_csv(p, encoding="utf-8-sig"))
            except Exception as e:
                log(f"Failed aggregate read: {p} | {e}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            write_csv(df, aggregate_dir / f"{name}.csv")
            log(f"Aggregated {name}: {df.shape}")
        else:
            log(f"No files for aggregate: {name}")

    write_csv(cohort_status, aggregate_dir / "cohort_status_summary.csv")


def make_simple_figures(out_dir: Path) -> None:
    if not HAS_MPL:
        log("matplotlib unavailable. Skipping figures.")
        return

    agg = out_dir / "aggregate_tables"
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    selected_path = agg / "all_selected_representations.csv"
    burden_path = agg / "all_burden_K_sensitivity.csv"
    random_path = agg / "all_random_baseline_summary.csv"
    split_path = agg / "all_split_validation.csv"

    # Figure: Delta D over GE
    if selected_path.exists():
        df = pd.read_csv(selected_path, encoding="utf-8-sig")
        for endpoint in ENDPOINTS:
            sub = df[(df["endpoint"] == endpoint) & df["deltaD_best_minus_GE"].notna()]
            if not sub.empty:
                plt.figure(figsize=(8, 5))
                plt.hist(sub["deltaD_best_minus_GE"], bins=40)
                plt.axvline(0, linestyle="--")
                plt.axvline(0.5, linestyle="--")
                plt.axvline(1.0, linestyle="--")
                plt.xlabel("Delta D: best representation minus GE")
                plt.ylabel("Cancer-BP observations")
                plt.title(f"{endpoint}: Delta D over GE baseline")
                plt.tight_layout()
                plt.savefig(fig_dir / f"DeltaD_best_minus_GE_{endpoint}.png", dpi=300)
                plt.close()

        # Representation class counts
        if "representation_class" in df.columns:
            counts = df.groupby(["endpoint", "representation_class"]).size().reset_index(name="n")
            write_csv(counts, agg / "summary_representation_class_counts.csv")

    # Figure: random baseline observed vs random 95pct
    if random_path.exists():
        df = pd.read_csv(random_path, encoding="utf-8-sig")
        for endpoint_metric in ["OS", "STAGE"]:
            obs_col = f"observed_{endpoint_metric}_D"
            rand_col = f"random_{endpoint_metric}_95pct"
            sub = df[df[obs_col].notna() & df[rand_col].notna()]
            if not sub.empty:
                plt.figure(figsize=(6, 6))
                plt.scatter(sub[rand_col], sub[obs_col])
                lim = max(sub[rand_col].max(), sub[obs_col].max()) + 0.5
                plt.plot([0, lim], [0, lim], linestyle="--")
                plt.xlabel(f"Random 95th percentile {endpoint_metric} D")
                plt.ylabel(f"Observed selected burden {endpoint_metric} D")
                plt.title(f"All-cohort random baseline: {endpoint_metric}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"Random_baseline_observed_vs_random95_{endpoint_metric}.png", dpi=300)
                plt.close()

    # Figure: K sensitivity
    if burden_path.exists():
        df = pd.read_csv(burden_path, encoding="utf-8-sig")
        for metric in ["OS_D", "STAGE_D"]:
            sub = df[df[metric].notna()]
            if not sub.empty:
                summary = sub.groupby(["endpoint_selection", "K", "cutoff"])[metric].median().reset_index()
                write_csv(summary, agg / f"summary_K_sensitivity_median_{metric}.csv")

    # Split validation summary
    if split_path.exists():
        df = pd.read_csv(split_path, encoding="utf-8-sig")
        if not df.empty:
            summary = df.groupby(["cohort", "endpoint_selection", "K", "cutoff"]).agg(
                median_test_OS_D=("test_OS_D", "median"),
                q05_test_OS_D=("test_OS_D", lambda x: np.nanquantile(x, 0.05)),
                q95_test_OS_D=("test_OS_D", lambda x: np.nanquantile(x, 0.95)),
                frac_test_OS_D_ge_1p301=("test_OS_D", lambda x: np.nanmean(np.array(x) >= D_THRESHOLD)),
                median_test_STAGE_D=("test_STAGE_D", "median"),
                q05_test_STAGE_D=("test_STAGE_D", lambda x: np.nanquantile(x, 0.05)),
                q95_test_STAGE_D=("test_STAGE_D", lambda x: np.nanquantile(x, 0.95)),
                frac_test_STAGE_D_ge_1p301=("test_STAGE_D", lambda x: np.nanmean(np.array(x) >= D_THRESHOLD)),
                n_splits=("iteration", "count"),
            ).reset_index()
            write_csv(summary, agg / "summary_split_validation_by_cohort.csv")


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    start = time.time()

    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "aggregate_tables")
    ensure_dir(OUT_DIR / "figures")
    ensure_dir(OUT_DIR / "logs")

    config = {
        "BASE_DIR": str(BASE_DIR),
        "GSEA_DIR": str(GSEA_DIR),
        "HALLMARK_GMT": str(HALLMARK_GMT),
        "OUT_DIR": str(OUT_DIR),
        "MIN_MATCHED_GENES": MIN_MATCHED_GENES,
        "D_THRESHOLD": D_THRESHOLD,
        "PRIMARY_K": PRIMARY_K,
        "K_VALUES": K_VALUES,
        "RANDOM_ITER": RANDOM_ITER,
        "SPLIT_ITER": SPLIT_ITER,
        "TRAIN_FRAC": TRAIN_FRAC,
        "RANDOM_SEED": RANDOM_SEED,
        "REPRESENTATIONS": REPRESENTATIONS,
        "DEFAULT_EXCLUDE_CODES": sorted(DEFAULT_EXCLUDE_CODES),
        "COHORT_WHITELIST": COHORT_WHITELIST,
        "HAS_LIFELINES": HAS_LIFELINES,
    }
    write_json(config, OUT_DIR / "run_config.json")

    log("AIDO-Multi-Omics-I-4.0 internal benchmark started")
    log(f"Output directory: {OUT_DIR}")

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")
    if not HALLMARK_GMT.exists():
        raise FileNotFoundError(f"Hallmark GMT not found: {HALLMARK_GMT}")

    gene_sets = load_gmt(HALLMARK_GMT)
    log(f"Loaded gene sets: {len(gene_sets)} from {HALLMARK_GMT.name}")

    cohorts = discover_cohorts(BASE_DIR)
    write_csv(cohorts, OUT_DIR / "discovered_cohorts.csv")

    log(f"Discovered cohorts: {cohorts.shape[0]}")
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
                gene_sets=gene_sets,
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

    log("Making simple summary figures")
    make_simple_figures(OUT_DIR)

    elapsed = time.time() - start
    log(f"Completed in {elapsed / 60:.2f} minutes")

    final_report = {
        "output_dir": str(OUT_DIR),
        "elapsed_minutes": elapsed / 60,
        "n_cohorts_discovered": int(cohorts.shape[0]),
        "n_cohorts_completed": int((cohort_status["status"] == "completed").sum()) if "status" in cohort_status else 0,
        "n_cohorts_error": int((cohort_status["status"] == "error").sum()) if "status" in cohort_status else 0,
        "note": "All CSV outputs use UTF-8-SIG encoding for Excel compatibility.",
    }
    write_json(final_report, OUT_DIR / "final_report.json")

    log("AIDO-Multi-Omics-I-4.0 internal benchmark finished")


if __name__ == "__main__":
    main()