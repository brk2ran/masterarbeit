# src/eda.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.features import (
    get_analysis_subsets,
    metric_coverage_by_round_task,
    round_task_counts,
    unknown_summary_by_round,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
REPORTS_DIR_DEFAULT = PROJECT_ROOT / "reports"
TABLES_DIR_DEFAULT = REPORTS_DIR_DEFAULT / "tables"


def _ensure_dirs() -> None:
    TABLES_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)


def load_dataset(parquet_path: Path = PARQUET_DEFAULT) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _quantile_table(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    """
    Erzeugt robuste Deskriptiv-Tabelle je Gruppe: n, median, q25, q75, min, max.
    """
    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' fehlt im DataFrame.")

    # nur nicht-null Werte
    d = df[df[metric].notna()].copy()

    if d.empty:
        return pd.DataFrame(columns=group_cols + [
            f"{prefix}_n", f"{prefix}_median", f"{prefix}_q25", f"{prefix}_q75",
            f"{prefix}_min", f"{prefix}_max"
        ])

    g = d.groupby(group_cols)[metric]
    out = pd.DataFrame({
        f"{prefix}_n": g.size(),
        f"{prefix}_median": g.median(),
        f"{prefix}_q25": g.quantile(0.25),
        f"{prefix}_q75": g.quantile(0.75),
        f"{prefix}_min": g.min(),
        f"{prefix}_max": g.max(),
    }).reset_index()

    return out.sort_values(group_cols)


def build_core_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Baut zentrale Tabellen für die EDA/Trendbasis:
      - Round×Task Coverage
      - UNKNOWN Summary by Round
      - Metric Coverage (Energy/Power/Quality) by Round×Task
      - Trendtable latency (Round×Task)
      - Trendtable energy (Round×Task, Energy subset)
    """
    subsets = get_analysis_subsets(df)

    # Coverage: alle (inkl UNKNOWN)
    cov_all = round_task_counts(df, include_unknown=True)

    # UNKNOWN summary
    unk_sum = unknown_summary_by_round(df)

    # Coverage: Energy/Power/Quality (auf task-known Basis sinnvoll)
    df_task_known = subsets["DS_LATENCY"]  # task-known + latency vorhanden (praktisch "task-known base")
    cov_energy = metric_coverage_by_round_task(df_task_known, "energy_uj") if "energy_uj" in df.columns else pd.DataFrame()
    cov_power = metric_coverage_by_round_task(df_task_known, "power_mw") if "power_mw" in df.columns else pd.DataFrame()
    cov_quality = metric_coverage_by_round_task(df_task_known, "accuracy") if "accuracy" in df.columns else pd.DataFrame()

    # Trendtables:
    # Latenz auf DS_LATENCY (task-known)
    latency_trend = _quantile_table(
        subsets["DS_LATENCY"],
        metric="latency_us",
        group_cols=["round", "task"],
        prefix="latency_us",
    )

    # Energie auf DS_ENERGY (task-known + energy)
    energy_trend = _quantile_table(
        subsets["DS_ENERGY"],
        metric="energy_uj",
        group_cols=["round", "task"],
        prefix="energy_uj",
    )

    return {
        "coverage_round_task_all": cov_all,
        "unknown_summary_by_round": unk_sum,
        "coverage_energy_by_round_task": cov_energy,
        "coverage_power_by_round_task": cov_power,
        "coverage_quality_by_round_task_accuracy": cov_quality,
        "trend_latency_us_round_task": latency_trend,
        "trend_energy_uj_round_task": energy_trend,
    }


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path = TABLES_DIR_DEFAULT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, t in tables.items():
        if t is None or t.empty:
            # Leere Tabellen trotzdem speichern? -> ja, aber klar markieren
            out_path = out_dir / f"{name}.csv"
            pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[table] {name}: EMPTY -> {out_path}")
            continue

        out_path = out_dir / f"{name}.csv"
        t.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[table] {name}: {len(t)} rows -> {out_path}")


def run_eda(parquet_path: Path = PARQUET_DEFAULT) -> None:
    _ensure_dirs()
    df = load_dataset(parquet_path)

    tables = build_core_tables(df)
    save_tables(tables)


if __name__ == "__main__":
    run_eda()
