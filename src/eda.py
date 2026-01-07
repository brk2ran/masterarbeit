# src/eda.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.features import (
    add_features,
    get_analysis_subsets,
    metric_coverage_by_round_task,
    round_task_counts,
    scope_summary_by_round,
    unknown_summary_by_round,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"


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
    Robuste Deskriptiv-Tabelle je Gruppe: n, median, q25, q75, min, max.
    """
    if metric not in df.columns:
        return pd.DataFrame(columns=group_cols + [
            f"{prefix}_n", f"{prefix}_median", f"{prefix}_q25", f"{prefix}_q75",
            f"{prefix}_min", f"{prefix}_max",
        ])

    d = df[df[metric].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=group_cols + [
            f"{prefix}_n", f"{prefix}_median", f"{prefix}_q25", f"{prefix}_q75",
            f"{prefix}_min", f"{prefix}_max",
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


def build_core_tables(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Baut zentrale Tabellen für EDA & Datenqualität.

    Scope-Logik:
      - OUT_OF_SCOPE (1D_DS_CNN / Streaming Wakeword) wird NICHT analysiert.
      - UNKNOWN bleibt als Datenqualitätsindikator sichtbar.
    """
    df = add_features(df_raw)
    subsets = get_analysis_subsets(df)

    ds_non_oos = subsets["DS_NON_OOS"]      # IN_SCOPE + UNKNOWN
    ds_in_scope = subsets["DS_IN_SCOPE"]    # nur AD/KWS/VWW/IC

    # A) Scope summary
    scope_sum = scope_summary_by_round(df)

    # B) Coverage (Round×Task) – auf NON_OOS
    cov_all = round_task_counts(ds_non_oos, include_unknown=True)

    # C) UNKNOWN summary – auf NON_OOS (damit OUT_OF_SCOPE separat bleibt)
    unk_sum = unknown_summary_by_round(ds_non_oos)

    # D) Metric coverage – auf IN_SCOPE
    cov_energy = metric_coverage_by_round_task(ds_in_scope, "energy_uj")
    cov_power = metric_coverage_by_round_task(ds_in_scope, "power_mw")
    cov_acc = metric_coverage_by_round_task(ds_in_scope, "accuracy")
    cov_auc = metric_coverage_by_round_task(ds_in_scope, "auc")

    # E) Trendtables – auf IN_SCOPE und jeweils metric-notna
    latency_trend = _quantile_table(
        subsets["DS_LATENCY"],
        metric="latency_us",
        group_cols=["round", "task_canon"],
        prefix="latency_us",
    ).rename(columns={"task_canon": "task"})

    energy_trend = _quantile_table(
        subsets["DS_ENERGY"],
        metric="energy_uj",
        group_cols=["round", "task_canon"],
        prefix="energy_uj",
    ).rename(columns={"task_canon": "task"})

    return {
        "scope_summary_by_round": scope_sum,
        "coverage_round_task_all": cov_all,
        "unknown_summary_by_round": unk_sum,
        "coverage_energy_by_round_task": cov_energy,
        "coverage_power_by_round_task": cov_power,
        "coverage_quality_by_round_task_accuracy": cov_acc,
        "coverage_quality_by_round_task_auc": cov_auc,
        "trend_latency_us_round_task": latency_trend,
        "trend_energy_uj_round_task": energy_trend,
    }


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path = TABLES_DIR_DEFAULT) -> None:
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
    df = load_dataset(parquet_path)
    tables = build_core_tables(df)
    save_tables(tables, out_dir)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="EDA tables for MLPerf-Tiny (scope-aware).")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT, help="Path to interim parquet")
    parser.add_argument("--out-dir", type=Path, default=TABLES_DIR_DEFAULT, help="Output dir for CSV tables")
    args = parser.parse_args(argv)

    run_eda(args.parquet, args.out_dir)


if __name__ == "__main__":
    main()
