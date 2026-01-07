# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


# ------------------------------------------------------------
# Canonical dictionaries (Scope-Entscheidung)
# ------------------------------------------------------------

IN_SCOPE_TASKS = {"AD", "KWS", "VWW", "IC"}

# OUT_OF_SCOPE: Streaming Wakeword (1D DS-CNN) – wird bewusst nicht analysiert
OUT_OF_SCOPE_MODEL_CANON = {"1D_DS_CNN"}


@dataclass(frozen=True)
class CanonicalMaps:
    model_map: Dict[str, str]
    task_by_model: Dict[str, str]
    mode_by_task: Dict[str, str]


CANON = CanonicalMaps(
    model_map={
        # KWS
        "DSCNN": "DS_CNN",
        "DS-CNN": "DS_CNN",
        "DS CNN": "DS_CNN",
        "DS_CNN": "DS_CNN",
        "DS CN N": "DS_CNN",
        "DSCNN ": "DS_CNN",
        "DS-CNN ": "DS_CNN",
        # AD
        "FC AUTOENCODER": "FC_AE",
        "FC AUTO ENCODER": "FC_AE",
        "AUTOENCODER": "FC_AE",
        "FC_AE": "FC_AE",
        "DENSE": "FC_AE",  # MLPerf-Tiny AD wird häufig als Dense/FC angegeben
        # VWW
        "MOBILENETV1 (0.25X)": "MOBILENETV1",
        "MOBILENETV1(0.25X)": "MOBILENETV1",
        "MOBILENETV1 (0.25)": "MOBILENETV1",
        "MOBILENETV1": "MOBILENETV1",
        "MOBILENET": "MOBILENETV1",
        # IC
        "RESNET-V1": "RESNET_V1",
        "RESNET V1": "RESNET_V1",
        "RESNET": "RESNET_V1",
        "RESNET_V1": "RESNET_V1",
        # NULL / Missing Model Column
        "NULL": "NULL",
        "NONE": "NULL",
        "NAN": "NULL",
        "<NA>": "NULL",
        # OUT OF SCOPE
        "1D DS-CNN": "1D_DS_CNN",
        "1D DS CNN": "1D_DS_CNN",
        "1D_DS_CNN": "1D_DS_CNN",
    },
    task_by_model={
        "DS_CNN": "KWS",
        "FC_AE": "AD",
        "MOBILENETV1": "VWW",
        "RESNET_V1": "IC",
        # NULL => nicht zuordenbar
        "NULL": "UNKNOWN",
        # Out-of-scope bekommt ein separates Task-Label, damit es nicht in KWS “reinrutscht”
        "1D_DS_CNN": "OUT_OF_SCOPE",
    },
    mode_by_task={
        "AD": "single_stream",
        "IC": "single_stream",
        "VWW": "single_stream",
        "KWS": "offline",
        "UNKNOWN": "unknown",
        "OUT_OF_SCOPE": "out_of_scope",
    },
)


def _norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _canon_model(model_mlc: object) -> str:
    raw = _norm_text(model_mlc)
    if raw == "":
        return "NULL"
    key = raw.upper()
    return CANON.model_map.get(key, "UNKNOWN")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds canonical columns used throughout the pipeline.

    Output columns (new):
      - model_mlc_canon: DS_CNN / FC_AE / MOBILENETV1 / RESNET_V1 / NULL / 1D_DS_CNN / UNKNOWN
      - task_canon: AD / KWS / VWW / IC / UNKNOWN / OUT_OF_SCOPE
      - mode_canon: single_stream / offline / unknown / out_of_scope
      - scope_status: IN_SCOPE / OUT_OF_SCOPE / UNKNOWN

    Scope-Entscheidung:
      - 1D_DS_CNN (Streaming Wakeword) wird als OUT_OF_SCOPE markiert und nicht analysiert.
    """
    d = df.copy()

    # model canonical
    if "model_mlc" not in d.columns:
        d["model_mlc"] = pd.NA
    d["model_mlc_canon"] = d["model_mlc"].map(_canon_model)

    # task canonical
    d["task_canon"] = d["model_mlc_canon"].map(CANON.task_by_model).fillna("UNKNOWN")

    # mode canonical
    d["mode_canon"] = d["task_canon"].map(CANON.mode_by_task).fillna("unknown")

    # scope_status
    def _scope(row) -> str:
        t = row["task_canon"]
        m = row["model_mlc_canon"]
        if t == "OUT_OF_SCOPE" or m in OUT_OF_SCOPE_MODEL_CANON:
            return "OUT_OF_SCOPE"
        if t == "UNKNOWN" or m == "UNKNOWN":
            return "UNKNOWN"
        if t in IN_SCOPE_TASKS:
            return "IN_SCOPE"
        return "UNKNOWN"

    d["scope_status"] = d.apply(_scope, axis=1)

    return d


# ------------------------------------------------------------
# Helpers used by EDA / checks
# ------------------------------------------------------------

def _parse_round_value(r: object) -> float:
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return 1e9
    s = str(r).strip()
    if s.startswith("v"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return 1e9


def sort_rounds(df: pd.DataFrame, round_col: str = "round") -> pd.DataFrame:
    if round_col not in df.columns:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].map(_parse_round_value)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


def get_analysis_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Standardisierte Subsets:
      - DS_ALL: alles (inkl. UNKNOWN & OUT_OF_SCOPE)
      - DS_NON_OOS: alles außer OUT_OF_SCOPE (UNKNOWN bleibt drin)
      - DS_IN_SCOPE: nur IN_SCOPE (AD/KWS/VWW/IC)
      - DS_LATENCY / DS_ENERGY / DS_POWER / DS_ACCURACY / DS_AUC: jeweils auf IN_SCOPE gefiltert
    """
    if "scope_status" not in df.columns or "task_canon" not in df.columns:
        raise ValueError("add_features(df) muss vor get_analysis_subsets(df) ausgeführt werden.")

    ds_all = df.copy()
    ds_non_oos = ds_all[ds_all["scope_status"].ne("OUT_OF_SCOPE")].copy()
    ds_in_scope = ds_all[ds_all["scope_status"].eq("IN_SCOPE")].copy()

    def _metric_subset(metric: str) -> pd.DataFrame:
        if metric not in ds_in_scope.columns:
            return ds_in_scope.iloc[0:0].copy()
        return ds_in_scope[ds_in_scope[metric].notna()].copy()

    return {
        "DS_ALL": ds_all,
        "DS_NON_OOS": ds_non_oos,
        "DS_IN_SCOPE": ds_in_scope,
        "DS_LATENCY": _metric_subset("latency_us"),
        "DS_ENERGY": _metric_subset("energy_uj"),
        "DS_POWER": _metric_subset("power_mw"),
        "DS_ACCURACY": _metric_subset("accuracy"),
        "DS_AUC": _metric_subset("auc"),
    }


def scope_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts per round: IN_SCOPE / OUT_OF_SCOPE / UNKNOWN.
    """
    if "round" not in df.columns or "scope_status" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'scope_status' fehlen.")

    out = (
        df.groupby(["round", "scope_status"])
        .size()
        .rename("rows")
        .reset_index()
    )
    out = sort_rounds(out, "round")
    return out.sort_values(["round", "scope_status"])


def round_task_counts(df: pd.DataFrame, *, include_unknown: bool) -> pd.DataFrame:
    """
    Coverage Round×Task (Anzahl Zeilen), standardmäßig auf DS_NON_OOS gedacht.
    """
    if "round" not in df.columns or "task_canon" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'task_canon' fehlen.")

    d = df.copy()
    if not include_unknown:
        d = d[d["task_canon"].ne("UNKNOWN")]

    out = (
        d.groupby(["round", "task_canon"])
        .size()
        .rename("rows")
        .reset_index()
        .rename(columns={"task_canon": "task"})
    )
    out = sort_rounds(out, "round")
    return out.sort_values(["round", "task"])


def unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNKNOWN summary pro Round:
      - rows_total (auf dem übergebenen df)
      - rows_unknown (scope_status == UNKNOWN)
      - share_unknown
    Empfehlung: df = DS_NON_OOS verwenden, damit OUT_OF_SCOPE separat bleibt.
    """
    if "round" not in df.columns or "scope_status" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'scope_status' fehlen.")

    tmp = df.copy()
    tmp["is_unknown"] = tmp["scope_status"].eq("UNKNOWN").astype(int)

    out = (
        tmp.groupby("round", as_index=False)
        .agg(rows_total=("scope_status", "size"), rows_unknown=("is_unknown", "sum"))
    )
    out["rows_total"] = out["rows_total"].astype(int)
    out["rows_unknown"] = out["rows_unknown"].astype(int)
    out["share_unknown"] = out["rows_unknown"] / out["rows_total"]

    return sort_rounds(out, "round")


def metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Coverage pro Round×Task für eine Metrik:
      - rows
      - rows_with_metric
      - share_metric
    """
    if "round" not in df.columns or "task_canon" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'task_canon' fehlen.")
    if metric not in df.columns:
        return pd.DataFrame(columns=["round", "task", "rows", "rows_with_metric", "share_metric"])

    base = df.groupby(["round", "task_canon"]).size().rename("rows")
    with_metric = df[df[metric].notna()].groupby(["round", "task_canon"]).size().rename("rows_with_metric")

    out = pd.concat([base, with_metric], axis=1).fillna(0).reset_index()
    out["rows"] = out["rows"].astype(int)
    out["rows_with_metric"] = out["rows_with_metric"].astype(int)
    out["share_metric"] = out["rows_with_metric"] / out["rows"].replace(0, pd.NA)

    out = out.rename(columns={"task_canon": "task"})
    out = sort_rounds(out, "round")
    return out.sort_values(["round", "task"])
