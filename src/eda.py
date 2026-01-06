from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.features import (
    add_features,
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
    df = pd.read_parquet(parquet_path)

    # Robustheit: falls jemand ein altes Parquet ohne canon-Spalten hat
    if "task_canon" not in df.columns or "model_mlc_canon" not in df.columns:
        df = add_features(df)

    return df


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
        return pd.DataFrame(columns=group_cols + [
            f"{prefix}_n", f"{prefix}_median", f"{prefix}_q25", f"{prefix}_q75",
            f"{prefix}_min", f"{prefix}_max"
        ])

    g = d.groupby(group_cols, dropna=False)[metric]
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
    Zentrale Tabellen für die EDA/Trendbasis (PRIMARY_SCOPE only; OUT_OF_SCOPE ausgeschlossen):
      - Coverage Round×Task (inkl. UNKNOWN, ohne OUT_OF_SCOPE)
      - UNKNOWN Summary by Round (OUT_OF_SCOPE excluded)
      - Metric Coverage (Energy/Power/Quality) by Round×Task
      - Trendtable latency (Round×Task)
      - Trendtable energy (Round×Task)
    """
    subsets = get_analysis_subsets(df)

    # Coverage: inkl UNKNOWN, aber OUT_OF_SCOPE ausgeschlossen
    cov_all = round_task_counts(df, include_unknown=True, include_out_of_scope=False)

    # UNKNOWN summary (OUT_OF_SCOPE excluded in function)
    unk_sum = unknown_summary_by_round(df)

    # Coverage: Energy/Power/Quality auf task-known/in-scope Basis
    df_task_known = subsets["DS_LATENCY"]  # in_scope + latency vorhanden (gute Basis für round×task)
    cov_energy = metric_coverage_by_round_task(df_task_known, "energy_uj")
    cov_power = metric_coverage_by_round_task(df_task_known, "power_mw")
    cov_quality = metric_coverage_by_round_task(df_task_known, "accuracy")

    # Trendtables (nutze task_canon als "task" für Konsistenz)
    d_lat = subsets["DS_LATENCY"].copy()
    d_lat["task"] = d_lat["task_canon"]

    latency_trend = _quantile_table(
        d_lat,
        metric="latency_us",
        group_cols=["round", "task"],
        prefix="latency_us",
    )

    d_en = subsets["DS_ENERGY"].copy()
    if not d_en.empty:
        d_en["task"] = d_en["task_canon"]

    energy_trend = _quantile_table(
        d_en,
        metric="energy_uj",
        group_cols=["round", "task"],
        prefix="energy_uj",
    ) if not d_en.empty else pd.DataFrame(columns=["round", "task", "energy_uj_n", "energy_uj_median", "energy_uj_q25", "energy_uj_q75", "energy_uj_min", "energy_uj_max"])

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
        out_path = out_dir / f"{name}.csv"
        if t is None or t.empty:
            pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[table] {name}: EMPTY -> {out_path}")
            continue

        t.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[table] {name}: {len(t)} rows -> {out_path}")


def run_eda(parquet_path: Path = PARQUET_DEFAULT) -> None:
    _ensure_dirs()
    df = load_dataset(parquet_path)
    tables = build_core_tables(df)
    save_tables(tables)


if __name__ == "__main__":
    run_eda()
