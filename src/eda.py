from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

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

MIN_N_DEFAULT = 5  # Low-n: n < MIN_N_DEFAULT


# ---------------------------
# Round sorting (v0.5, v1.0, ...)
# ---------------------------
def _round_key(r: object) -> float:
    if not isinstance(r, str):
        return 1e9
    s = r.strip()
    if s.startswith("v"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return 1e9


def _sort_rounds(df: pd.DataFrame, round_col: str = "round") -> pd.DataFrame:
    if df.empty or round_col not in df.columns:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].astype(str).map(_round_key)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


# ---------------------------
# Load dataset
# ---------------------------
def load_dataset(parquet_path: Path = PARQUET_DEFAULT) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Robustheit: falls altes Parquet ohne canon-Spalten vorliegt
    if "task_canon" not in df.columns or "model_mlc_canon" not in df.columns or "scope_status" not in df.columns:
        df = add_features(df)

    return df


# ---------------------------
# Trend tables
# ---------------------------
def _quantile_table(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    *,
    prefix: str,
    min_n: int,
) -> pd.DataFrame:
    """
    Liefert je Gruppe:
      n, median, q25, q75, q10, q90, min, max, iqr, low_n
    """
    cols = group_cols + [
        f"{prefix}_n",
        f"{prefix}_median",
        f"{prefix}_q25",
        f"{prefix}_q75",
        f"{prefix}_q10",
        f"{prefix}_q90",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_iqr",
        f"{prefix}_low_n",
    ]

    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' fehlt im DataFrame.")

    d = df[df[metric].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(group_cols, dropna=False)[metric]
    out = pd.DataFrame(
        {
            f"{prefix}_n": g.size(),
            f"{prefix}_median": g.median(),
            f"{prefix}_q25": g.quantile(0.25),
            f"{prefix}_q75": g.quantile(0.75),
            f"{prefix}_q10": g.quantile(0.10),
            f"{prefix}_q90": g.quantile(0.90),
            f"{prefix}_min": g.min(),
            f"{prefix}_max": g.max(),
        }
    ).reset_index()

    out[f"{prefix}_iqr"] = out[f"{prefix}_q75"] - out[f"{prefix}_q25"]
    out[f"{prefix}_low_n"] = out[f"{prefix}_n"].astype(int) < int(min_n)

    return _sort_rounds(out[group_cols + [c for c in out.columns if c not in group_cols]], round_col="round")


def _index_trend_table(
    trend_df: pd.DataFrame,
    *,
    metric_prefix: str,
    task_col: str = "task",
    round_col: str = "round",
) -> pd.DataFrame:
    """
    Indexierung je task:
      index = value / baseline_median
    baseline = früheste Round mit median > 0
    """
    if trend_df.empty:
        return trend_df.copy()

    required = {round_col, task_col, f"{metric_prefix}_median", f"{metric_prefix}_n"}
    missing = required - set(trend_df.columns)
    if missing:
        raise ValueError(f"Indexierung nicht möglich, Spalten fehlen: {sorted(missing)}")

    df = _sort_rounds(trend_df, round_col=round_col).copy()

    # Baseline pro Task: erste Round mit median > 0
    baselines = []
    for task, d in df.groupby(task_col, dropna=False):
        med = pd.to_numeric(d[f"{metric_prefix}_median"], errors="coerce")
        valid = med.notna() & (med > 0)
        if not valid.any():
            baselines.append({task_col: task, "baseline_round": pd.NA, "baseline_n": pd.NA, "baseline_median": pd.NA})
            continue
        first = d.loc[valid].iloc[0]
        baselines.append(
            {
                task_col: task,
                "baseline_round": first[round_col],
                "baseline_n": int(first[f"{metric_prefix}_n"]),
                "baseline_median": float(first[f"{metric_prefix}_median"]),
            }
        )

    out = df.merge(pd.DataFrame(baselines), on=task_col, how="left")
    denom = pd.to_numeric(out["baseline_median"], errors="coerce")

    def _div(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce") / denom

    for base in ["median", "q25", "q75", "q10", "q90"]:
        col = f"{metric_prefix}_{base}"
        if col in out.columns:
            out[f"{metric_prefix}_{base}_index"] = _div(col)

    return out


# ---------------------------
# Core tables
# ---------------------------
def build_core_tables(df: pd.DataFrame, *, min_n: int = MIN_N_DEFAULT) -> Dict[str, pd.DataFrame]:
    """
    Zentrale Tabellen:
      - coverage_round_task_all (UNKNOWN inkl., OUT_OF_SCOPE exkl.)
      - unknown_summary_by_round
      - coverage_{metric}_by_round_task (energy_uj, power_mw, accuracy)
      - trend_latency_us_round_task (+ indexed)
      - trend_energy_uj_round_task (+ indexed)
    """
    subsets = get_analysis_subsets(df)

    cov_all = round_task_counts(df, include_unknown=True, include_out_of_scope=False)
    unk_sum = unknown_summary_by_round(df)

    # Base für Metric Coverage: IN_SCOPE + latency vorhanden
    base = subsets.get("DS_LATENCY", pd.DataFrame()).copy()

    coverage_tables = {}
    for metric in ["energy_uj", "power_mw", "accuracy"]:
        coverage_tables[f"coverage_{metric}_by_round_task"] = metric_coverage_by_round_task(base, metric)

    # Trendtables: task_canon -> task
    d_lat = base.copy()
    d_lat["task"] = d_lat["task_canon"]
    latency_trend = _quantile_table(d_lat, "latency_us", ["round", "task"], prefix="latency_us", min_n=min_n)

    d_en = subsets.get("DS_ENERGY", pd.DataFrame()).copy()
    d_en["task"] = d_en["task_canon"] if not d_en.empty else pd.Series(dtype="object")
    energy_trend = _quantile_table(d_en, "energy_uj", ["round", "task"], prefix="energy_uj", min_n=min_n)

    return {
        "coverage_round_task_all": cov_all,
        "unknown_summary_by_round": unk_sum,
        **coverage_tables,
        "coverage_quality_by_round_task_accuracy": coverage_tables["coverage_accuracy_by_round_task"],
        "trend_latency_us_round_task": latency_trend,
        "trend_energy_uj_round_task": energy_trend,
        "trend_latency_us_round_task_indexed": _index_trend_table(latency_trend, metric_prefix="latency_us"),
        "trend_energy_uj_round_task_indexed": _index_trend_table(energy_trend, metric_prefix="energy_uj"),
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


def run_eda(parquet_path: Path = PARQUET_DEFAULT, *, min_n: int = MIN_N_DEFAULT) -> None:
    TABLES_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    df = load_dataset(parquet_path)
    tables = build_core_tables(df, min_n=min_n)
    save_tables(tables)


if __name__ == "__main__":
    run_eda()
