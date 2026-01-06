# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


METRIC_COLS = ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"]


@dataclass(frozen=True)
class SubsetSpec:
    """Formalisiert ein Analyse-Subset für reproduzierbare Auswertungen."""
    name: str
    description: str


SUBSETS = {
    "DS_LATENCY": SubsetSpec(
        name="DS_LATENCY",
        description="Task-known + latency_us vorhanden (Standardbasis für Latenz-EDA/Trends).",
    ),
    "DS_ENERGY": SubsetSpec(
        name="DS_ENERGY",
        description="Task-known + latency_us & energy_uj vorhanden (Energie-Trends/Trade-offs).",
    ),
    "DS_POWER": SubsetSpec(
        name="DS_POWER",
        description="Task-known + power_mw vorhanden (kleines Subset, eher illustrativ).",
    ),
    "DS_QUALITY": SubsetSpec(
        name="DS_QUALITY",
        description="Task-known + (accuracy oder auc) vorhanden (sehr kleines Subset).",
    ),
    "DS_UNKNOWN": SubsetSpec(
        name="DS_UNKNOWN",
        description="Task unbekannt bzw. model_mlc fehlt (Pivot 'Null'-Spalte), nur Reporting.",
    ),
}


def ensure_task_column(df: pd.DataFrame) -> pd.DataFrame:
    """Stellt sicher, dass df eine task-Spalte hat und UNKNOWN konsistent gesetzt ist."""
    df = df.copy()
    if "task" not in df.columns:
        df["task"] = "UNKNOWN"
    else:
        df["task"] = df["task"].fillna("UNKNOWN")
        df.loc[df["task"].astype("string").str.strip().eq(""), "task"] = "UNKNOWN"
    return df


def add_availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt Standard-Flags zur Subset-Bildung und Validierung.

    Flags:
      - has_latency, has_energy, has_power, has_quality
      - is_task_known (task != UNKNOWN)
      - is_model_known (model_mlc notna, falls vorhanden)
    """
    df = ensure_task_column(df)

    df["has_latency"] = df.get("latency_us", pd.Series(index=df.index, dtype="float64")).notna()
    df["has_energy"] = df.get("energy_uj", pd.Series(index=df.index, dtype="float64")).notna()
    df["has_power"] = df.get("power_mw", pd.Series(index=df.index, dtype="float64")).notna()

    acc = df.get("accuracy", pd.Series(index=df.index, dtype="float64")).notna()
    auc = df.get("auc", pd.Series(index=df.index, dtype="float64")).notna()
    df["has_quality"] = acc | auc

    df["is_task_known"] = df["task"].ne("UNKNOWN")

    if "model_mlc" in df.columns:
        df["is_model_known"] = df["model_mlc"].notna()
    else:
        df["is_model_known"] = True

    return df


def get_analysis_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Liefert konsistente Analyse-Subsets, um EDA/Trends/Pareto reproduzierbar
    und methodisch sauber zu fahren.
    """
    df = add_availability_flags(df)

    subsets = {
        "DS_LATENCY": df[df["is_task_known"] & df["has_latency"]].copy(),
        "DS_ENERGY": df[df["is_task_known"] & df["has_latency"] & df["has_energy"]].copy(),
        "DS_POWER": df[df["is_task_known"] & df["has_power"]].copy(),
        "DS_QUALITY": df[df["is_task_known"] & df["has_quality"]].copy(),
        "DS_UNKNOWN": df[~df["is_task_known"]].copy(),
    }
    return subsets


def round_task_counts(df: pd.DataFrame, *, include_unknown: bool = True) -> pd.DataFrame:
    """
    Coverage-Tabelle (Round×Task) als DataFrame.
    """
    df = ensure_task_column(df)
    if not include_unknown:
        df = df[df["task"].ne("UNKNOWN")].copy()

    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt im DataFrame.")

    cov = df.groupby(["round", "task"]).size().rename("rows").reset_index()
    return cov


def metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Coverage je Round×Task für eine Metrik: rows_total, rows_with_metric, share_metric.
    """
    if metric not in df.columns:
        raise ValueError(f"Metrikspalte '{metric}' fehlt im DataFrame.")

    df = ensure_task_column(df)
    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt im DataFrame.")

    base = df.groupby(["round", "task"]).size().rename("rows_total")
    with_m = df[df[metric].notna()].groupby(["round", "task"]).size().rename("rows_with_metric")

    out = pd.concat([base, with_m], axis=1).fillna(0).astype(int)
    out["share_metric"] = (out["rows_with_metric"] / out["rows_total"]).round(4)
    return out.reset_index().sort_values(["round", "task"])


def unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary pro Round: rows_total, rows_unknown, share_unknown.
    UNKNOWN wird über task=='UNKNOWN' bestimmt.
    """
    df = ensure_task_column(df)
    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt im DataFrame.")

    total = df.groupby("round").size().rename("rows_total")
    unk = df[df["task"].eq("UNKNOWN")].groupby("round").size().rename("rows_unknown")

    out = pd.concat([total, unk], axis=1).fillna(0).astype(int)
    out["share_unknown"] = (out["rows_unknown"] / out["rows_total"]).round(4)
    return out.reset_index().sort_values("round")
