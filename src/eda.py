from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.features import add_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
REPORTS_DIR_DEFAULT = PROJECT_ROOT / "reports"
TABLES_DIR_DEFAULT = REPORTS_DIR_DEFAULT / "tables"


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def _ensure_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(parquet_path: Path = PARQUET_DEFAULT) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _parse_round_value(r: object) -> float:
    """
    Rounds typischerweise: 'v0.5', 'v1.3' etc.
    Fallback: sehr groß, damit Unbekanntes ans Ende sortiert.
    """
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return 1e9
    s = str(r).strip()
    if s.startswith("v"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return 1e9


def _sort_rounds(df: pd.DataFrame, round_col: str = "round") -> pd.DataFrame:
    if round_col not in df.columns:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].map(_parse_round_value)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


# ------------------------------------------------------------
# Subsets / validity utilities
# ------------------------------------------------------------

def get_analysis_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Liefert definierte Analyse-Subsets basierend auf canonical Features.

    Annahmen:
      - task_canon == 'UNKNOWN' bedeutet: Task/Model nicht sauber zuordenbar
      - latency_us / energy_uj / power_mw optional vorhanden
    """
    if "task_canon" not in df.columns:
        raise ValueError("Spalte 'task_canon' fehlt. Bitte erst add_features(df) anwenden.")

    ds_all = df.copy()
    ds_task_known = ds_all[ds_all["task_canon"].ne("UNKNOWN")].copy()

    # metrics subsets (auf task-known Basis)
    if "latency_us" in ds_task_known.columns:
        ds_latency = ds_task_known[ds_task_known["latency_us"].notna()].copy()
    else:
        ds_latency = ds_task_known.iloc[0:0].copy()

    if "energy_uj" in ds_task_known.columns:
        ds_energy = ds_task_known[ds_task_known["energy_uj"].notna()].copy()
    else:
        ds_energy = ds_task_known.iloc[0:0].copy()

    if "power_mw" in ds_task_known.columns:
        ds_power = ds_task_known[ds_task_known["power_mw"].notna()].copy()
    else:
        ds_power = ds_task_known.iloc[0:0].copy()

    # quality subsets (accuracy/auc separat; zusätzlich quality_value falls vorhanden)
    if "accuracy" in ds_task_known.columns:
        ds_acc = ds_task_known[ds_task_known["accuracy"].notna()].copy()
    else:
        ds_acc = ds_task_known.iloc[0:0].copy()

    if "auc" in ds_task_known.columns:
        ds_auc = ds_task_known[ds_task_known["auc"].notna()].copy()
    else:
        ds_auc = ds_task_known.iloc[0:0].copy()

    if "quality_value" in ds_task_known.columns:
        ds_quality_value = ds_task_known[ds_task_known["quality_value"].notna()].copy()
    else:
        ds_quality_value = ds_task_known.iloc[0:0].copy()

    return {
        "DS_ALL": ds_all,
        "DS_TASK_KNOWN": ds_task_known,
        "DS_LATENCY": ds_latency,
        "DS_ENERGY": ds_energy,
        "DS_POWER": ds_power,
        "DS_ACCURACY": ds_acc,
        "DS_AUC": ds_auc,
        "DS_QUALITY_VALUE": ds_quality_value,
    }


def round_task_counts(df: pd.DataFrame, *, include_unknown: bool) -> pd.DataFrame:
    """
    Coverage Round×Task (Anzahl Zeilen).
    Verwendet task_canon und schreibt Output-Spalte als 'task' (Kompatibilität).
    """
    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt.")
    if "task_canon" not in df.columns:
        raise ValueError("Spalte 'task_canon' fehlt.")

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
    return _sort_rounds(out, "round").sort_values(["round", "task"])


def unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNKNOWN summary pro Round:
      - rows_total
      - rows_unknown (task_canon == UNKNOWN)
      - share_unknown
    """
    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt.")
    if "task_canon" not in df.columns:
        raise ValueError("Spalte 'task_canon' fehlt.")

    tmp = df.copy()
    tmp["is_unknown"] = tmp["task_canon"].eq("UNKNOWN").astype(int)

    out = (
        tmp.groupby("round", as_index=False)
        .agg(rows_total=("task_canon", "size"), rows_unknown=("is_unknown", "sum"))
    )
    out["rows_total"] = out["rows_total"].astype(int)
    out["rows_unknown"] = out["rows_unknown"].astype(int)
    out["share_unknown"] = out["rows_unknown"] / out["rows_total"]

    return _sort_rounds(out, "round")


def metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Coverage pro Round×Task für eine Metrik:
      - rows (gesamt)
      - rows_with_metric (metric notna)
      - share_metric
    """
    if "round" not in df.columns:
        raise ValueError("Spalte 'round' fehlt.")
    if "task_canon" not in df.columns:
        raise ValueError("Spalte 'task_canon' fehlt.")
    if metric not in df.columns:
        return pd.DataFrame(columns=["round", "task", "rows", "rows_with_metric", "share_metric"])

    base = df.groupby(["round", "task_canon"]).size().rename("rows")
    with_metric = df[df[metric].notna()].groupby(["round", "task_canon"]).size().rename("rows_with_metric")

    out = pd.concat([base, with_metric], axis=1).fillna(0).reset_index()
    out["rows"] = out["rows"].astype(int)
    out["rows_with_metric"] = out["rows_with_metric"].astype(int)
    out["share_metric"] = out["rows_with_metric"] / out["rows"].replace(0, pd.NA)

    out = out.rename(columns={"task_canon": "task"})
    out = _sort_rounds(out, "round")
    return out.sort_values(["round", "task"])


def _quantile_table(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    """
    Robuste Deskriptiv-Tabelle je Gruppe: n, median, q25, q75, min, max.
    """
    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' fehlt im DataFrame.")

    d = df[df[metric].notna()].copy()
    if d.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                f"{prefix}_n",
                f"{prefix}_median",
                f"{prefix}_q25",
                f"{prefix}_q75",
                f"{prefix}_min",
                f"{prefix}_max",
            ]
        )

    # Guard: Gruppierspalten müssen eindeutig 1D sein (keine Duplicate-Column-Namen)
    for c in group_cols:
        if c not in d.columns:
            raise ValueError(f"Gruppierspalte '{c}' fehlt.")
        if isinstance(d[c], pd.DataFrame):
            raise ValueError(f"Gruppierspalte '{c}' ist nicht 1-dimensional (duplizierter Spaltenname).")

    g = d.groupby(group_cols)[metric]
    out = pd.DataFrame(
        {
            f"{prefix}_n": g.size(),
            f"{prefix}_median": g.median(),
            f"{prefix}_q25": g.quantile(0.25),
            f"{prefix}_q75": g.quantile(0.75),
            f"{prefix}_min": g.min(),
            f"{prefix}_max": g.max(),
        }
    ).reset_index()

    out = _sort_rounds(out, "round")
    return out.sort_values(group_cols)


# ------------------------------------------------------------
# Core tables
# ------------------------------------------------------------

def build_core_tables(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Baut zentrale Tabellen für die EDA/Trendbasis (canon-basiert):
      - Round×Task Coverage (inkl UNKNOWN)
      - UNKNOWN Summary by Round
      - Metric Coverage (Energy/Power/Accuracy/AUC/QualityValue) by Round×Task (task-known base)
      - Trendtable latency (Round×Task, task-known + latency)
      - Trendtable energy (Round×Task, task-known + energy)
    """
    df = add_features(df_raw)
    subsets = get_analysis_subsets(df)

    cov_all = round_task_counts(subsets["DS_ALL"], include_unknown=True)
    unk_sum = unknown_summary_by_round(subsets["DS_ALL"])

    df_task_known = subsets["DS_TASK_KNOWN"]

    cov_energy = metric_coverage_by_round_task(df_task_known, "energy_uj")
    cov_power = metric_coverage_by_round_task(df_task_known, "power_mw")
    cov_acc = metric_coverage_by_round_task(df_task_known, "accuracy")
    cov_auc = metric_coverage_by_round_task(df_task_known, "auc")
    cov_quality_value = metric_coverage_by_round_task(df_task_known, "quality_value")

    # Trendtables: explizit "task" aus task_canon erzeugen (ohne raw task zu überschreiben!)
    latency_trend = pd.DataFrame()
    if not subsets["DS_LATENCY"].empty and "latency_us" in subsets["DS_LATENCY"].columns:
        d_lat = subsets["DS_LATENCY"][["round", "task_canon", "latency_us"]].copy()
        d_lat = d_lat.rename(columns={"task_canon": "task"})
        latency_trend = _quantile_table(
            d_lat,
            metric="latency_us",
            group_cols=["round", "task"],
            prefix="latency_us",
        )

    energy_trend = pd.DataFrame()
    if not subsets["DS_ENERGY"].empty and "energy_uj" in subsets["DS_ENERGY"].columns:
        d_en = subsets["DS_ENERGY"][["round", "task_canon", "energy_uj"]].copy()
        d_en = d_en.rename(columns={"task_canon": "task"})
        energy_trend = _quantile_table(
            d_en,
            metric="energy_uj",
            group_cols=["round", "task"],
            prefix="energy_uj",
        )

    return {
        "coverage_round_task_all": cov_all,
        "unknown_summary_by_round": unk_sum,
        "coverage_energy_by_round_task": cov_energy,
        "coverage_power_by_round_task": cov_power,
        "coverage_quality_by_round_task_accuracy": cov_acc,
        "coverage_quality_by_round_task_auc": cov_auc,
        "coverage_quality_by_round_task_quality_value": cov_quality_value,
        "trend_latency_us_round_task": latency_trend,
        "trend_energy_uj_round_task": energy_trend,
    }


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, t in tables.items():
        out_path = out_dir / f"{name}.csv"
        if t is None or t.empty:
            pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[table] {name}: EMPTY -> {out_path}")
            continue

        t.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[table] {name}: {len(t)} rows -> {out_path}")


def run_eda(parquet_path: Path = PARQUET_DEFAULT, out_dir: Path = TABLES_DIR_DEFAULT) -> None:
    _ensure_dirs(out_dir)
    df = load_dataset(parquet_path)
    tables = build_core_tables(df)
    save_tables(tables, out_dir)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build EDA core tables from MLPerf Tiny parquet.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=PARQUET_DEFAULT,
        help="Path to data/interim/mlperf_tiny_raw.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TABLES_DIR_DEFAULT,
        help="Output directory for CSV tables (default: reports/tables)",
    )
    args = parser.parse_args(argv)
    run_eda(args.parquet, args.out_dir)


if __name__ == "__main__":
    # Wichtig: bevorzugt als Modul ausführen: python -m src.eda
    main()
