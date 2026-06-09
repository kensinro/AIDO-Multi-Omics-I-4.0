# -*- coding: utf-8 -*-
"""
AIDO-Multi-Omics-I-4.0
Evidence packaging / manuscript-ready summary pipeline v1

Purpose
-------
This script DOES NOT rerun the full TCGA or external validation pipelines.
It consumes already generated outputs from:

1. TCGA internal benchmark V3:
   D:/AIDO-Temp/AIDO_MultiOmics_I_4_internal_benchmark_v3_NARGAB_<timestamp>/

2. External validation V5:
   D:/AIDO-Temp/AIDO_MultiOmics_I_4_external_validation_v5_CLINICAL_OVERRIDES_<timestamp>/

It creates manuscript-ready tables and figures for AIDO-Multi-Omics-I-4.0:

A. Cohort priority score
B. Random95-supported subset analysis
C. FDR-supported subset analysis
D. KIRP focused representation panel
E. BRCA / METABRIC external validation panel
F. Integration-fragility paired-bar examples
G. Compact tables for main text and supplementary files

Output
------
D:/AIDO-Temp/AIDO_MultiOmics_I_4_evidence_package_<timestamp>/

Interpretation
--------------
This script packages existing evidence. It is intended to reduce cherry-picking
risk and make the Results section easier to write.
"""

from __future__ import annotations

import math
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

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

# If None, script auto-detects latest matching folders.
TCGA_V3_OUT: Optional[Path] = None
EXTERNAL_V5_OUT: Optional[Path] = None

OUT_DIR = AIDO_TEMP / f"AIDO_MultiOmics_I_4_evidence_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

PRIMARY_K = 10
D_THRESHOLD = -math.log10(0.05)
Q10 = 0.10
Q05 = 0.05

# Cohorts of special interest for AIDO-Multi-Omics-I-4.0
HIGHLIGHT_COHORTS = ["BRCA", "KIRP", "KIRC", "BLCA", "LAML", "UCEC", "LUAD"]

# External main validation anchor
MAIN_EXTERNAL_DATASET = "METABRIC_BRCA"

# KIRP backup is reproducibility/sanity check, not independent external validation.
KIRP_BACKUP_DATASET = "KIRP_TCGA_CBIO_BACKUP"

# Example BPs to prefer for visual panels if available.
PREFERRED_FRAGILITY_EXAMPLES = [
    ("KIRP", "STAGE", "HALLMARK_UNFOLDED_PROTEIN_RESPONSE"),
    ("KIRP", "STAGE", "HALLMARK_PANCREAS_BETA_CELLS"),
    ("KIRP", "STAGE", "HALLMARK_GLYCOLYSIS"),
    ("KIRC", "OS", "HALLMARK_MYC_TARGETS_V2"),
    ("KIRC", "OS", "HALLMARK_DNA_REPAIR"),
    ("BRCA", "OS", "HALLMARK_ESTROGEN_RESPONSE_EARLY"),
    ("BRCA", "OS", "HALLMARK_ESTROGEN_RESPONSE_LATE"),
    ("BLCA", "STAGE", "HALLMARK_MYOGENESIS"),
    ("BLCA", "STAGE", "HALLMARK_ANGIOGENESIS"),
]

PREFERRED_KIRP_BPS = [
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_APICAL_JUNCTION",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_PANCREAS_BETA_CELLS",
]

PREFERRED_METABRIC_BPS = [
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_MITOTIC_SPINDLE",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_APICAL_SURFACE",
    "HALLMARK_UV_RESPONSE_UP",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
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
    s = str(x)
    for old, new in [("::", "_"), ("/", "_"), (" ", "_"), ("+", "plus"), ("-", "_")]:
        s = s.replace(old, new)
    return "".join(c if c.isalnum() or c in "._" else "_" for c in s).strip("_")


def clean_bp_name(x: str) -> str:
    s = str(x)
    if "::" in s:
        s = s.split("::", 1)[1]
    return s


def latest_dir(pattern: str) -> Path:
    candidates = [p for p in AIDO_TEMP.glob(pattern) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directory found for pattern: {pattern}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def maybe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return pd.DataFrame()


def add_bp_clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "bp" in out.columns:
        out["bp_clean"] = out["bp"].map(clean_bp_name)
    elif "bp_key" in out.columns:
        out["bp_clean"] = out["bp_key"].map(clean_bp_name)
    return out


def finite_median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.median()) if len(x) else np.nan


def count_ge_threshold(x: pd.Series, threshold: float = D_THRESHOLD) -> int:
    x = pd.to_numeric(x, errors="coerce")
    return int((x >= threshold).sum())


def frac_ge_threshold(x: pd.Series, threshold: float = D_THRESHOLD) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[x.notna()]
    if len(x) == 0:
        return np.nan
    return float((x >= threshold).mean())


# =============================================================================
# 2. LOAD INPUTS
# =============================================================================

def load_inputs() -> Dict[str, pd.DataFrame]:
    tcga_dir = TCGA_V3_OUT if TCGA_V3_OUT is not None else latest_dir("AIDO_MultiOmics_I_4_internal_benchmark_v3_NARGAB_*")
    external_dir = EXTERNAL_V5_OUT if EXTERNAL_V5_OUT is not None else latest_dir("AIDO_MultiOmics_I_4_external_validation_v5_CLINICAL_OVERRIDES_*")

    log(f"Using TCGA V3 output: {tcga_dir}")
    log(f"Using External V5 output: {external_dir}")

    tcga_agg = tcga_dir / "aggregate_tables"
    ext_agg = external_dir / "aggregate_tables"

    data = {
        "tcga_dir": tcga_dir,
        "external_dir": external_dir,
        "selected": add_bp_clean(read_csv_required(tcga_agg / "all_selected_representations.csv")),
        "rep_maps": add_bp_clean(read_csv_required(tcga_agg / "all_representation_maps.csv")),
        "random_gs": add_bp_clean(maybe_read(tcga_agg / "all_size_matched_random_geneset.csv")),
        "split": maybe_read(tcga_agg / "all_split_validation.csv"),
        "external": add_bp_clean(maybe_read(ext_agg / "external_validation_all_records.csv")),
        "external_availability": maybe_read(ext_agg / "external_validation_data_availability.csv"),
        "external_summary": maybe_read(ext_agg / "external_validation_summary_by_dataset_endpoint.csv"),
    }

    return data


# =============================================================================
# 3. FILTERS AND MERGES
# =============================================================================

def primary_selected(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "readiness_K" in out.columns:
        out = out[out["readiness_K"] == PRIMARY_K]
    if "collection" in out.columns:
        out = out[out["collection"] == "HALLMARK"]
    return out


def primary_informative(df: pd.DataFrame) -> pd.DataFrame:
    out = primary_selected(df)
    if "endpoint_informative" in out.columns:
        out = out[out["endpoint_informative"] == 1]
    return out


def add_random95_flags(selected: pd.DataFrame, random_gs: pd.DataFrame) -> pd.DataFrame:
    if selected.empty or random_gs.empty:
        out = selected.copy()
        out["random95_supported_best"] = np.nan
        return out

    key_cols = ["cohort", "collection", "readiness_K", "bp_key", "endpoint", "best_representation"]
    rg = random_gs.copy()
    rg_key = rg.rename(columns={
        "real_exceeds_random95": "random95_supported_best",
        "random_gene_set_D_95pct": "random_gene_set_D_95pct_best",
        "random_gene_set_D_median": "random_gene_set_D_median_best",
        "empirical_p_ge_observed": "random_empirical_p_best",
        "random_gene_set_iter_valid": "random_gene_set_iter_valid_best",
    })

    keep = key_cols + [
        "random95_supported_best",
        "random_gene_set_D_95pct_best",
        "random_gene_set_D_median_best",
        "random_empirical_p_best",
        "random_gene_set_iter_valid_best",
    ]

    keep = [c for c in keep if c in rg_key.columns]
    merged = selected.merge(rg_key[keep], on=[c for c in key_cols if c in selected.columns and c in rg_key.columns], how="left")
    return merged


def q_support_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q_col = "best_q_by_endpoint"
    if q_col not in out.columns:
        out["q10_supported"] = np.nan
        out["q05_supported"] = np.nan
        return out
    q = pd.to_numeric(out[q_col], errors="coerce")
    out["q10_supported"] = (q <= Q10).astype(int)
    out["q05_supported"] = (q <= Q05).astype(int)
    return out


# =============================================================================
# 4. COHORT PRIORITY SCORE
# =============================================================================

def make_cohort_priority_table(selected: pd.DataFrame, random_gs: pd.DataFrame, split: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    sel = primary_selected(selected)
    sel = add_random95_flags(sel, random_gs)
    sel = q_support_flags(sel)

    rows = []
    for cohort, sub_all in sel.groupby("cohort"):
        sub_inf = sub_all[sub_all.get("endpoint_informative", 0) == 1].copy()

        row = {
            "cohort": cohort,
            "n_total_records": int(sub_all.shape[0]),
            "n_endpoint_informative": int(sub_inf.shape[0]),
            "n_OS_informative": int((sub_inf["endpoint"] == "OS").sum()) if "endpoint" in sub_inf.columns else 0,
            "n_STAGE_informative": int((sub_inf["endpoint"] == "STAGE").sum()) if "endpoint" in sub_inf.columns else 0,
            "n_strong_gain": int((sub_inf.get("gain_class", "") == "strong_gain").sum()),
            "n_moderate_gain": int((sub_inf.get("gain_class", "") == "moderate_gain").sum()),
            "n_any_fragile": int(pd.to_numeric(sub_inf.get("any_fragile", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_any_strong_fragile": int(pd.to_numeric(sub_inf.get("any_strong_fragile", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_any_signal_lost": int(pd.to_numeric(sub_inf.get("any_signal_lost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_random95_supported": int(pd.to_numeric(sub_inf.get("random95_supported_best", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_q10_supported": int(pd.to_numeric(sub_inf.get("q10_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_q05_supported": int(pd.to_numeric(sub_inf.get("q05_supported", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "median_best_D": finite_median(sub_inf.get("best_D", pd.Series(dtype=float))),
            "median_delta_best_GE": finite_median(sub_inf.get("deltaD_best_minus_GE", pd.Series(dtype=float))),
            "median_delta_full_GE": finite_median(sub_inf.get("deltaD_GE_CN_MU_minus_GE", pd.Series(dtype=float))),
        }

        rows.append(row)

    priority = pd.DataFrame(rows)

    # Split stability.
    if not split.empty:
        sp = split.copy()
        # Per cohort: max/median stability across endpoint selections.
        sp_summary = sp.groupby("cohort").agg(
            split_n=("iteration", "count"),
            median_test_OS_D=("test_OS_D", "median"),
            frac_test_OS_D_ge_threshold=("test_OS_D", frac_ge_threshold),
            median_test_STAGE_D=("test_STAGE_D", "median"),
            frac_test_STAGE_D_ge_threshold=("test_STAGE_D", frac_ge_threshold),
        ).reset_index()
        sp_summary["split_stability_score"] = (
            sp_summary["frac_test_OS_D_ge_threshold"].fillna(0)
            + sp_summary["frac_test_STAGE_D_ge_threshold"].fillna(0)
        )
        priority = priority.merge(sp_summary, on="cohort", how="left")
    else:
        priority["split_stability_score"] = np.nan

    # External bonus.
    priority["external_validation_bonus"] = 0.0
    if not external.empty:
        ext = external.copy()
        ext_valid = ext[
            (ext["external_type"] != "TCGA_backup_not_external")
            & (ext["external_same_rep_supported_D_ge_1p301"] == 1)
        ]
        ext_count = ext_valid.groupby("source_cohort").size().reset_index(name="n_independent_external_supported")
        priority = priority.merge(ext_count, left_on="cohort", right_on="source_cohort", how="left")
        priority["n_independent_external_supported"] = priority["n_independent_external_supported"].fillna(0)
        priority["external_validation_bonus"] = np.minimum(priority["n_independent_external_supported"], 10) * 2.0
        priority = priority.drop(columns=["source_cohort"], errors="ignore")
    else:
        priority["n_independent_external_supported"] = 0

    # TCGA/cBio backup reproducibility bonus separately.
    priority["tcga_cbio_backup_bonus"] = 0.0
    if not external.empty:
        backup = external[
            (external["external_type"] == "TCGA_backup_not_external")
            & (external["external_same_rep_supported_D_ge_1p301"] == 1)
        ]
        bcount = backup.groupby("source_cohort").size().reset_index(name="n_tcga_cbio_backup_supported")
        priority = priority.merge(bcount, left_on="cohort", right_on="source_cohort", how="left")
        priority["n_tcga_cbio_backup_supported"] = priority["n_tcga_cbio_backup_supported"].fillna(0)
        priority["tcga_cbio_backup_bonus"] = np.minimum(priority["n_tcga_cbio_backup_supported"], 10) * 1.0
        priority = priority.drop(columns=["source_cohort"], errors="ignore")
    else:
        priority["n_tcga_cbio_backup_supported"] = 0

    # Priority score, deliberately simple and transparent.
    priority["priority_score"] = (
        priority["n_endpoint_informative"].fillna(0)
        + 2.0 * priority["n_strong_gain"].fillna(0)
        + 1.0 * priority["n_moderate_gain"].fillna(0)
        + 1.5 * priority["n_any_strong_fragile"].fillna(0)
        + 1.5 * priority["n_any_signal_lost"].fillna(0)
        + 2.0 * priority["n_random95_supported"].fillna(0)
        + 1.0 * priority["n_q10_supported"].fillna(0)
        + 20.0 * priority.get("split_stability_score", 0).fillna(0)
        + priority["external_validation_bonus"].fillna(0)
        + priority["tcga_cbio_backup_bonus"].fillna(0)
    )

    # Assign role notes.
    def role(cohort: str) -> str:
        if cohort == "BRCA":
            return "main external anchor; METABRIC OS/RFS validation"
        if cohort == "KIRP":
            return "strong internal highlight; cBioPortal TCGA backup reproducibility"
        if cohort == "KIRC":
            return "high-D / fragility cautionary kidney contrast"
        if cohort == "BLCA":
            return "endpoint-specific STAGE / fragility example"
        if cohort == "LAML":
            return "OS-only MU-informative example"
        if cohort in ["UCEC", "LUAD"]:
            return "external CPTAC stage-only stress-test context"
        return "pan-cancer benchmark cohort"

    priority["proposed_role"] = priority["cohort"].map(role)
    priority = priority.sort_values("priority_score", ascending=False)
    return priority


# =============================================================================
# 5. SUBSET ANALYSES: RANDOM95 AND FDR
# =============================================================================

def representation_class_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for endpoint, sub in df.groupby("endpoint"):
        total = sub.shape[0]
        for cls, n in sub["representation_class"].value_counts(dropna=False).items():
            rows.append({
                "subset": label,
                "endpoint": endpoint,
                "representation_class": cls,
                "n": int(n),
                "fraction": n / total if total else np.nan,
                "n_total_endpoint_records": int(total),
            })
    return pd.DataFrame(rows)


def gain_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for endpoint, sub in df.groupby("endpoint"):
        total = sub.shape[0]
        for cls, n in sub["gain_class"].value_counts(dropna=False).items():
            rows.append({
                "subset": label,
                "endpoint": endpoint,
                "gain_class": cls,
                "n": int(n),
                "fraction": n / total if total else np.nan,
                "n_total_endpoint_records": int(total),
            })
    return pd.DataFrame(rows)


def fragility_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "fragile_GE_CN", "fragile_GE_MU", "fragile_GE_CN_MU",
        "strong_fragile_GE_CN", "strong_fragile_GE_MU", "strong_fragile_GE_CN_MU",
        "signal_lost_GE_CN", "signal_lost_GE_MU", "signal_lost_GE_CN_MU",
        "any_fragile", "any_strong_fragile", "any_signal_lost",
    ]
    rows = []
    for endpoint, sub in df.groupby("endpoint"):
        for c in cols:
            if c not in sub.columns:
                continue
            val = int(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())
            rows.append({
                "subset": label,
                "endpoint": endpoint,
                "fragility_metric": c,
                "n": val,
                "fraction": val / sub.shape[0] if sub.shape[0] else np.nan,
                "n_total_endpoint_records": int(sub.shape[0]),
            })
    return pd.DataFrame(rows)


def make_subset_analyses(selected: pd.DataFrame, random_gs: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base = primary_informative(selected)
    base = add_random95_flags(base, random_gs)
    base = q_support_flags(base)

    subsets = {
        "p05_endpoint_informative": base,
        "random95_supported": base[pd.to_numeric(base.get("random95_supported_best", 0), errors="coerce") == 1],
        "q10_supported": base[pd.to_numeric(base.get("q10_supported", 0), errors="coerce") == 1],
        "q05_supported": base[pd.to_numeric(base.get("q05_supported", 0), errors="coerce") == 1],
        "random95_and_q10_supported": base[
            (pd.to_numeric(base.get("random95_supported_best", 0), errors="coerce") == 1)
            & (pd.to_numeric(base.get("q10_supported", 0), errors="coerce") == 1)
        ],
    }

    rep_tables = []
    gain_tables = []
    frag_tables = []
    overall_rows = []

    for label, df in subsets.items():
        rep_tables.append(representation_class_summary(df, label))
        gain_tables.append(gain_summary(df, label))
        frag_tables.append(fragility_summary(df, label))

        for endpoint, sub in df.groupby("endpoint"):
            overall_rows.append({
                "subset": label,
                "endpoint": endpoint,
                "n_records": int(sub.shape[0]),
                "median_best_D": finite_median(sub["best_D"]),
                "median_delta_best_GE": finite_median(sub["deltaD_best_minus_GE"]),
                "median_delta_full_GE": finite_median(sub["deltaD_GE_CN_MU_minus_GE"]),
                "n_GE_sufficient": int((sub["representation_class"] == "GE_sufficient").sum()),
                "n_CN_informative": int((sub["representation_class"] == "CN_informative").sum()),
                "n_MU_informative": int((sub["representation_class"] == "MU_informative").sum()),
                "n_multi_layer_informative": int((sub["representation_class"] == "multi_layer_informative").sum()),
                "n_strong_gain": int((sub["gain_class"] == "strong_gain").sum()),
                "n_any_strong_fragile": int(pd.to_numeric(sub.get("any_strong_fragile", 0), errors="coerce").fillna(0).sum()),
                "n_any_signal_lost": int(pd.to_numeric(sub.get("any_signal_lost", 0), errors="coerce").fillna(0).sum()),
            })

    return {
        "subset_overall_summary": pd.DataFrame(overall_rows),
        "subset_representation_class_summary": pd.concat([x for x in rep_tables if not x.empty], ignore_index=True) if rep_tables else pd.DataFrame(),
        "subset_gain_summary": pd.concat([x for x in gain_tables if not x.empty], ignore_index=True) if gain_tables else pd.DataFrame(),
        "subset_fragility_summary": pd.concat([x for x in frag_tables if not x.empty], ignore_index=True) if frag_tables else pd.DataFrame(),
        "base_with_support_flags": base,
    }


# =============================================================================
# 6. FOCUSED PANELS
# =============================================================================

def make_kirp_focused_tables(selected: pd.DataFrame, external: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sel = primary_informative(selected)
    kirp = sel[sel["cohort"] == "KIRP"].copy()
    if kirp.empty:
        return {"kirp_selected": pd.DataFrame(), "kirp_external_backup": pd.DataFrame(), "kirp_panel_table": pd.DataFrame()}

    kirp["preferred_order"] = kirp["bp_clean"].apply(lambda x: PREFERRED_KIRP_BPS.index(x) if x in PREFERRED_KIRP_BPS else 999)
    kirp_focus = kirp[
        (kirp["bp_clean"].isin(PREFERRED_KIRP_BPS))
        | (kirp["best_D"] >= 4)
        | (kirp["deltaD_best_minus_GE"] >= 1.5)
        | (kirp["any_strong_fragile"] == 1)
    ].copy()
    kirp_focus = kirp_focus.sort_values(["endpoint", "preferred_order", "best_D"], ascending=[True, True, False])

    backup = pd.DataFrame()
    if not external.empty:
        backup = external[
            (external["dataset_id"] == KIRP_BACKUP_DATASET)
            & (external["source_cohort"] == "KIRP")
        ].copy()
        backup = backup.sort_values(["external_endpoint", "external_same_rep_D"], ascending=[True, False])

    # Merge compact panel.
    if not backup.empty:
        ext_pivot = backup.pivot_table(
            index=["bp_clean", "tcga_endpoint", "tcga_best_representation"],
            columns="external_endpoint",
            values="external_same_rep_D",
            aggfunc="max"
        ).reset_index()
        ext_pivot.columns = [str(c) for c in ext_pivot.columns]
    else:
        ext_pivot = pd.DataFrame()

    panel_cols = [
        "cohort", "endpoint", "bp_clean", "best_representation", "best_D",
        "D_GE", "D_GE_CN", "D_GE_MU", "D_GE_CN_MU",
        "deltaD_best_minus_GE", "deltaD_GE_CN_minus_GE", "deltaD_GE_MU_minus_GE", "deltaD_GE_CN_MU_minus_GE",
        "representation_class", "gain_class", "any_strong_fragile", "any_signal_lost",
    ]
    panel_cols = [c for c in panel_cols if c in kirp_focus.columns]
    panel = kirp_focus[panel_cols].copy()

    if not ext_pivot.empty:
        ext_pivot = ext_pivot.rename(columns={
            "tcga_endpoint": "endpoint",
            "tcga_best_representation": "best_representation",
        })
        panel = panel.merge(ext_pivot, on=["bp_clean", "endpoint", "best_representation"], how="left")

    return {
        "kirp_selected": kirp_focus,
        "kirp_external_backup": backup,
        "kirp_panel_table": panel,
    }


def make_metabric_external_tables(external: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if external.empty:
        return {"metabric_records": pd.DataFrame(), "metabric_top": pd.DataFrame(), "metabric_summary": pd.DataFrame()}

    meta = external[external["dataset_id"] == MAIN_EXTERNAL_DATASET].copy()
    if meta.empty:
        return {"metabric_records": pd.DataFrame(), "metabric_top": pd.DataFrame(), "metabric_summary": pd.DataFrame()}

    # Valid METABRIC endpoints for manuscript.
    meta_valid = meta[meta["external_endpoint"].isin(["OS", "RFS", "STAGE"])].copy()

    top = meta_valid[
        (meta_valid["bp_clean"].isin(PREFERRED_METABRIC_BPS))
        | (meta_valid["external_same_rep_D"] >= D_THRESHOLD)
        | (meta_valid["external_best_D"] >= D_THRESHOLD)
    ].copy()
    top["preferred_order"] = top["bp_clean"].apply(lambda x: PREFERRED_METABRIC_BPS.index(x) if x in PREFERRED_METABRIC_BPS else 999)
    top = top.sort_values(["external_endpoint", "preferred_order", "external_same_rep_D"], ascending=[True, True, False])

    summary = meta_valid.groupby("external_endpoint").agg(
        n_records=("bp_clean", "count"),
        n_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "sum"),
        frac_same_rep_supported=("external_same_rep_supported_D_ge_1p301", "mean"),
        n_external_best_supported=("external_best_supported_D_ge_1p301", "sum"),
        frac_external_best_supported=("external_best_supported_D_ge_1p301", "mean"),
        median_tcga_best_D=("tcga_best_D", "median"),
        median_external_same_rep_D=("external_same_rep_D", "median"),
        median_external_best_D=("external_best_D", "median"),
    ).reset_index()

    return {
        "metabric_records": meta_valid,
        "metabric_top": top,
        "metabric_summary": summary,
    }


def make_fragility_example_table(selected: pd.DataFrame) -> pd.DataFrame:
    sel = primary_selected(selected)
    rows = []

    for cohort, endpoint, bp in PREFERRED_FRAGILITY_EXAMPLES:
        sub = sel[
            (sel["cohort"] == cohort)
            & (sel["endpoint"] == endpoint)
            & (sel["bp_clean"] == bp)
        ].copy()
        if not sub.empty:
            # If both informative and weak rows exist, prioritize informative/best D.
            sub = sub.sort_values(["endpoint_informative", "best_D"], ascending=[False, False])
            rows.append(sub.iloc[0])

    if rows:
        examples = pd.DataFrame(rows)
    else:
        examples = pd.DataFrame()

    # Add top fragility examples not already included.
    frag = sel[sel["deltaD_GE_CN_MU_minus_GE"].notna()].copy()
    frag = frag.sort_values("deltaD_GE_CN_MU_minus_GE", ascending=True)
    if not examples.empty:
        used = set(zip(examples["cohort"], examples["endpoint"], examples["bp_clean"]))
        frag = frag[~frag.apply(lambda r: (r["cohort"], r["endpoint"], r["bp_clean"]) in used, axis=1)]

    extra = frag.head(max(0, 12 - examples.shape[0]))
    out = pd.concat([examples, extra], ignore_index=True) if not examples.empty else extra.copy()

    keep = [
        "cohort", "endpoint", "bp_clean", "best_representation", "best_D",
        "D_GE", "D_GE_CN", "D_GE_MU", "D_GE_CN_MU",
        "deltaD_GE_CN_minus_GE", "deltaD_GE_MU_minus_GE", "deltaD_GE_CN_MU_minus_GE",
        "representation_class", "gain_class",
        "strong_fragile_GE_CN", "strong_fragile_GE_MU", "strong_fragile_GE_CN_MU",
        "signal_lost_GE_CN", "signal_lost_GE_MU", "signal_lost_GE_CN_MU",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep]


# =============================================================================
# 7. FIGURES
# =============================================================================

def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def fig_cohort_priority(priority: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL or priority.empty:
        return
    top = priority.head(15).copy().sort_values("priority_score", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top["cohort"], top["priority_score"])
    plt.xlabel("Cohort priority score")
    plt.ylabel("TCGA cohort")
    plt.title("AIDO-Multi-Omics-I-4.0 cohort evidence priority")
    save_fig(fig_dir / "Figure_priority_score_top15.png")


def fig_subset_summary(subset_overall: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL or subset_overall.empty:
        return
    for endpoint in sorted(subset_overall["endpoint"].dropna().unique()):
        sub = subset_overall[subset_overall["endpoint"] == endpoint].copy()
        sub = sub.sort_values("subset")
        plt.figure(figsize=(9, 5))
        plt.bar(sub["subset"], sub["n_records"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Number of records")
        plt.title(f"{endpoint}: retained records under robustness subsets")
        save_fig(fig_dir / f"Figure_subset_record_counts_{safe_name(endpoint)}.png")


def fig_metabric_external(meta_records: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL or meta_records.empty:
        return

    valid = meta_records[meta_records["external_endpoint"].isin(["OS", "RFS", "STAGE"])].copy()
    if valid.empty:
        return

    # Panel 1: endpoint support fractions.
    summary = valid.groupby("external_endpoint").agg(
        same=("external_same_rep_supported_D_ge_1p301", "mean"),
        best=("external_best_supported_D_ge_1p301", "mean"),
    ).reset_index()

    endpoints = ["OS", "RFS", "STAGE"]
    summary["external_endpoint"] = pd.Categorical(summary["external_endpoint"], categories=endpoints, ordered=True)
    summary = summary.sort_values("external_endpoint")
    x = np.arange(summary.shape[0])
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, summary["same"], width, label="Same representation")
    plt.bar(x + width/2, summary["best"], width, label="External best")
    plt.xticks(x, summary["external_endpoint"])
    plt.ylim(0, 1.05)
    plt.ylabel("Fraction supported (D ≥ 1.301)")
    plt.title("METABRIC external transport support")
    plt.legend()
    save_fig(fig_dir / "Figure_METABRIC_external_support_fraction.png")

    # Panel 2: top BP D values for OS/RFS.
    top = valid[
        (valid["external_endpoint"].isin(["OS", "RFS"]))
        & (valid["bp_clean"].isin(PREFERRED_METABRIC_BPS))
    ].copy()
    if not top.empty:
        top["label"] = top["bp_clean"].str.replace("HALLMARK_", "", regex=False) + " / " + top["external_endpoint"]
        top = top.sort_values("external_same_rep_D", ascending=True)
        plt.figure(figsize=(9, max(4, top.shape[0] * 0.35)))
        plt.barh(top["label"], top["external_same_rep_D"])
        plt.axvline(D_THRESHOLD, linestyle="--")
        plt.xlabel("METABRIC same-representation D")
        plt.title("Top transported BRCA BP representations in METABRIC")
        save_fig(fig_dir / "Figure_METABRIC_top_transported_BP.png")

    # Panel 3: TCGA D vs external same-rep D.
    for endpoint in ["OS", "RFS", "STAGE"]:
        sub = valid[valid["external_endpoint"] == endpoint].dropna(subset=["tcga_best_D", "external_same_rep_D"])
        if sub.shape[0] < 3:
            continue
        plt.figure(figsize=(5, 5))
        plt.scatter(sub["tcga_best_D"], sub["external_same_rep_D"])
        plt.axhline(D_THRESHOLD, linestyle="--")
        plt.axvline(D_THRESHOLD, linestyle="--")
        plt.xlabel("TCGA-BRCA selected D")
        plt.ylabel("METABRIC same-representation D")
        plt.title(f"METABRIC transfer: {endpoint}")
        save_fig(fig_dir / f"Figure_METABRIC_TCGA_vs_external_{endpoint}.png")


def fig_fragility_examples(examples: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL or examples.empty:
        return

    plot_df = examples.copy()
    needed = ["D_GE", "D_GE_CN", "D_GE_MU", "D_GE_CN_MU"]
    for c in needed:
        if c not in plot_df.columns:
            return

    plot_df = plot_df.head(10)
    reps = ["D_GE", "D_GE_CN", "D_GE_MU", "D_GE_CN_MU"]

    # Single grouped bar chart with concise labels.
    labels = (
        plot_df["cohort"].astype(str)
        + " "
        + plot_df["endpoint"].astype(str)
        + " / "
        + plot_df["bp_clean"].astype(str).str.replace("HALLMARK_", "", regex=False)
    )

    x = np.arange(plot_df.shape[0])
    width = 0.20

    plt.figure(figsize=(12, 6))
    for i, rep in enumerate(reps):
        vals = pd.to_numeric(plot_df[rep], errors="coerce").fillna(0)
        plt.bar(x + (i - 1.5) * width, vals, width, label=rep.replace("D_", ""))
    plt.axhline(D_THRESHOLD, linestyle="--")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("D = -log10(p)")
    plt.title("Representative integration-fragility examples")
    plt.legend()
    save_fig(fig_dir / "Figure_integration_fragility_paired_bars.png")


def fig_kirp_panel(kirp_panel: pd.DataFrame, fig_dir: Path) -> None:
    if not HAS_MPL or kirp_panel.empty:
        return

    # KIRP representation D bars for selected BPs.
    plot_df = kirp_panel.copy()
    plot_df = plot_df.drop_duplicates(["endpoint", "bp_clean", "best_representation"])
    plot_df = plot_df.head(14)

    reps = ["D_GE", "D_GE_CN", "D_GE_MU", "D_GE_CN_MU"]
    if not all(c in plot_df.columns for c in reps):
        return

    labels = plot_df["endpoint"].astype(str) + " / " + plot_df["bp_clean"].astype(str).str.replace("HALLMARK_", "", regex=False)
    x = np.arange(plot_df.shape[0])
    width = 0.20

    plt.figure(figsize=(12, 6))
    for i, rep in enumerate(reps):
        vals = pd.to_numeric(plot_df[rep], errors="coerce").fillna(0)
        plt.bar(x + (i - 1.5) * width, vals, width, label=rep.replace("D_", ""))
    plt.axhline(D_THRESHOLD, linestyle="--")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("D = -log10(p)")
    plt.title("KIRP focused representation panel")
    plt.legend()
    save_fig(fig_dir / "Figure_KIRP_focused_representation_panel.png")

    # KIRP backup endpoint heatmap-like table as image using imshow.
    endpoint_cols = [c for c in ["OS", "RFS", "PFS", "DSS", "STAGE"] if c in plot_df.columns]
    if endpoint_cols:
        mat = plot_df[endpoint_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        plt.figure(figsize=(8, max(4, plot_df.shape[0] * 0.35)))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="cBioPortal backup same-rep D")
        plt.yticks(np.arange(plot_df.shape[0]), labels)
        plt.xticks(np.arange(len(endpoint_cols)), endpoint_cols)
        plt.title("KIRP TCGA-cBioPortal backup support")
        save_fig(fig_dir / "Figure_KIRP_backup_endpoint_support_heatmap.png")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUT_DIR)
    table_dir = OUT_DIR / "tables"
    fig_dir = OUT_DIR / "figures"
    ensure_dir(table_dir)
    ensure_dir(fig_dir)

    data = load_inputs()

    config = {
        "TCGA_V3_OUT": str(data["tcga_dir"]),
        "EXTERNAL_V5_OUT": str(data["external_dir"]),
        "OUT_DIR": str(OUT_DIR),
        "PRIMARY_K": PRIMARY_K,
        "D_THRESHOLD": D_THRESHOLD,
        "Q10": Q10,
        "Q05": Q05,
        "note": "Evidence packaging only; does not rerun V3 or external validation.",
    }
    write_json(config, OUT_DIR / "evidence_package_config.json")

    selected = data["selected"]
    random_gs = data["random_gs"]
    split = data["split"]
    external = data["external"]

    # 1. Cohort priority.
    log("Creating cohort priority table")
    priority = make_cohort_priority_table(selected, random_gs, split, external)
    write_csv(priority, table_dir / "Table_cohort_priority_score.csv")
    write_csv(priority[priority["cohort"].isin(HIGHLIGHT_COHORTS)], table_dir / "Table_highlight_cohort_priority_score.csv")

    # 2. Robustness subset analyses.
    log("Creating random95/FDR-supported subset analyses")
    subset = make_subset_analyses(selected, random_gs)
    for name, df in subset.items():
        write_csv(df, table_dir / f"{name}.csv")

    # 3. KIRP focused panel.
    log("Creating KIRP focused tables")
    kirp_tables = make_kirp_focused_tables(selected, external)
    for name, df in kirp_tables.items():
        write_csv(df, table_dir / f"{name}.csv")

    # 4. METABRIC external validation.
    log("Creating METABRIC external validation tables")
    meta_tables = make_metabric_external_tables(external)
    for name, df in meta_tables.items():
        write_csv(df, table_dir / f"{name}.csv")

    # 5. Fragility examples.
    log("Creating integration fragility example table")
    frag_examples = make_fragility_example_table(selected)
    write_csv(frag_examples, table_dir / "Table_integration_fragility_examples.csv")

    # 6. Main-text compact tables.
    log("Creating manuscript compact tables")

    # Main Table candidate: representation class by endpoint, all vs robust subsets.
    rep_summary = subset["subset_representation_class_summary"]
    if not rep_summary.empty:
        pivot_rep = rep_summary.pivot_table(
            index=["subset", "endpoint"],
            columns="representation_class",
            values="n",
            aggfunc="sum",
            fill_value=0
        ).reset_index()
        write_csv(pivot_rep, table_dir / "MainTable_representation_classes_by_subset.csv")

    # Main Table candidate: METABRIC summary.
    if not meta_tables["metabric_summary"].empty:
        write_csv(meta_tables["metabric_summary"], table_dir / "MainTable_METABRIC_external_summary.csv")

    # Main Table candidate: top gain + fragility examples.
    base = subset["base_with_support_flags"]
    top_gain = base.sort_values("deltaD_best_minus_GE", ascending=False).head(50)
    write_csv(top_gain, table_dir / "Supplementary_top50_integration_gain_with_support_flags.csv")

    top_frag = base.sort_values("deltaD_GE_CN_MU_minus_GE", ascending=True).head(50)
    write_csv(top_frag, table_dir / "Supplementary_top50_full_integration_fragility_with_support_flags.csv")

    # 7. Figures.
    log("Creating figures")
    if HAS_MPL:
        fig_cohort_priority(priority, fig_dir)
        fig_subset_summary(subset["subset_overall_summary"], fig_dir)
        fig_metabric_external(meta_tables["metabric_records"], fig_dir)
        fig_fragility_examples(frag_examples, fig_dir)
        fig_kirp_panel(kirp_tables["kirp_panel_table"], fig_dir)
    else:
        log("matplotlib not available; skipped figures")

    # 8. Plain-language result bullets for manuscript drafting.
    log("Writing manuscript evidence bullets")
    bullets = []

    # External.
    meta_sum = meta_tables["metabric_summary"]
    if not meta_sum.empty:
        for _, r in meta_sum.iterrows():
            bullets.append(
                f"METABRIC {r['external_endpoint']}: "
                f"{int(r['n_same_rep_supported'])}/{int(r['n_records'])} same-representation supported; "
                f"{int(r['n_external_best_supported'])}/{int(r['n_records'])} external-best supported; "
                f"median same-rep D={r['median_external_same_rep_D']:.2f}."
            )

    # Subset robustness.
    overall = subset["subset_overall_summary"]
    if not overall.empty:
        for label in ["random95_supported", "q10_supported", "q05_supported", "random95_and_q10_supported"]:
            sub = overall[overall["subset"] == label]
            for _, r in sub.iterrows():
                bullets.append(
                    f"{label} / {r['endpoint']}: n={int(r['n_records'])}; "
                    f"GE={int(r['n_GE_sufficient'])}, CN={int(r['n_CN_informative'])}, "
                    f"MU={int(r['n_MU_informative'])}, multi-layer={int(r['n_multi_layer_informative'])}; "
                    f"strong fragility={int(r['n_any_strong_fragile'])}."
                )

    # Priority.
    if not priority.empty:
        top5 = ", ".join(priority.head(5)["cohort"].tolist())
        bullets.append(f"Top five priority cohorts by evidence score: {top5}.")

    with open(OUT_DIR / "manuscript_evidence_bullets.txt", "w", encoding="utf-8") as f:
        for b in bullets:
            f.write("- " + b + "\n")

    final_report = {
        "output_dir": str(OUT_DIR),
        "n_selected_records": int(selected.shape[0]),
        "n_external_records": int(external.shape[0]) if not external.empty else 0,
        "n_priority_cohorts": int(priority.shape[0]),
        "n_fragility_examples": int(frag_examples.shape[0]),
        "has_matplotlib": HAS_MPL,
        "key_outputs": [
            "tables/Table_cohort_priority_score.csv",
            "tables/subset_overall_summary.csv",
            "tables/MainTable_representation_classes_by_subset.csv",
            "tables/MainTable_METABRIC_external_summary.csv",
            "tables/kirp_panel_table.csv",
            "tables/Table_integration_fragility_examples.csv",
            "figures/Figure_METABRIC_external_support_fraction.png",
            "figures/Figure_integration_fragility_paired_bars.png",
            "figures/Figure_KIRP_focused_representation_panel.png",
        ],
    }
    write_json(final_report, OUT_DIR / "final_report.json")

    log(f"Evidence package completed: {OUT_DIR}")


if __name__ == "__main__":
    main()
