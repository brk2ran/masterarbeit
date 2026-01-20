from __future__ import annotations

from pathlib import Path
from typing import Dict

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

MIN_N_DEFAULT = 5  # Standard-Gate für Low-n Markierung (konsistent zu plots.py)


def _ensure_dirs() -> None:
    TABLES_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)


def _parse_round_value(r: str) -> float:
    """
    Sortierhilfe für Rounds wie 'v0.5', 'v1.0', ...
    """
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
    if round_col not in df.columns or df.empty:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].astype(str).map(_parse_round_value)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


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
    min_n: int = MIN_N_DEFAULT,
) -> pd.DataFrame:
    """
    Robuste Deskriptiv-Tabelle je Gruppe:
      - n, median, q25, q75, min, max (bestehend)
      - q10, q90, iqr, low_n (neu)
    """
    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' fehlt im DataFrame.")

    d = df[df[metric].notna()].copy()
    if d.empty:
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

    return _sort_rounds(out.sort_values(group_cols), round_col="round")


def _index_trend_table(
    trend_df: pd.DataFrame,
    *,
    metric_prefix: str,
    task_col: str = "task",
    round_col: str = "round",
) -> pd.DataFrame:
    """
    Erzeugt eine indexierte Version der Trendtabelle:
      index = value / baseline_median (baseline je task = früheste Round mit validem median > 0)

    Fügt hinzu:
      - baseline_round, baseline_n, baseline_median
      - *_median_index, *_q25_index, *_q75_index, *_q10_index, *_q90_index
    """
    if trend_df.empty:
        return trend_df.copy()

    required = {round_col, task_col, f"{metric_prefix}_median", f"{metric_prefix}_n"}
    missing = required - set(trend_df.columns)
    if missing:
        raise ValueError(f"Indexierung nicht möglich, Spalten fehlen: {sorted(missing)}")

    df = _sort_rounds(trend_df, round_col=round_col).copy()

    # Baseline je task: früheste Round mit median > 0
    baselines = []
    for task, d in df.groupby(task_col, dropna=False):
        d2 = d.copy()
        med = pd.to_numeric(d2[f"{metric_prefix}_median"], errors="coerce")
        valid = med.notna() & (med > 0)
        if not valid.any():
            baselines.append(
                {
                    task_col: task,
                    "baseline_round": pd.NA,
                    "baseline_n": pd.NA,
                    "baseline_median": pd.NA,
                }
            )
            continue

        first = d2.loc[valid].iloc[0]
        baselines.append(
            {
                task_col: task,
                "baseline_round": first[round_col],
                "baseline_n": int(first[f"{metric_prefix}_n"]),
                "baseline_median": float(first[f"{metric_prefix}_median"]),
            }
        )

    bdf = pd.DataFrame(baselines)
    out = df.merge(bdf, on=task_col, how="left")

    denom = pd.to_numeric(out["baseline_median"], errors="coerce")

    def _safe_div(s: pd.Series) -> pd.Series:
        num = pd.to_numeric(s, errors="coerce")
        return num / denom

    # Index-Spalten
    for base_name in ["median", "q25", "q75", "q10", "q90"]:
        col = f"{metric_prefix}_{base_name}"
        if col in out.columns:
            out[f"{metric_prefix}_{base_name}_index"] = _safe_div(out[col])

    return out


def build_core_tables(df: pd.DataFrame, *, min_n: int = MIN_N_DEFAULT) -> Dict[str, pd.DataFrame]:
    """
    Zentrale Tabellen für die EDA/Trendbasis (PRIMARY_SCOPE only; OUT_OF_SCOPE ausgeschlossen):
      - Coverage Round×Task (inkl. UNKNOWN, ohne OUT_OF_SCOPE)
      - UNKNOWN Summary by Round (OUT_OF_SCOPE excluded)
      - Metric Coverage (Energy/Power/Quality) by Round×Task
      - Trendtable latency (Round×Task) (erweitert: q10/q90/iqr/low_n)
      - Trendtable energy (Round×Task) (erweitert: q10/q90/iqr/low_n)
      - Indexierte Trendtables (neu, zusätzlich)
    """
    subsets = get_analysis_subsets(df)

    # Coverage: inkl UNKNOWN, aber OUT_OF_SCOPE ausgeschlossen
    cov_all = round_task_counts(df, include_unknown=True, include_out_of_scope=False)

    # UNKNOWN summary (OUT_OF_SCOPE excluded in function)
    unk_sum = unknown_summary_by_round(df)

    # Coverage: Energy/Power/Quality auf task-known/in-scope Basis
    df_task_known = subsets["DS_LATENCY"]  # in_scope + latency vorhanden (Basis für round×task)
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
        min_n=min_n,
    )

    d_en = subsets["DS_ENERGY"].copy()
    if not d_en.empty:
        d_en["task"] = d_en["task_canon"]

    if not d_en.empty:
        energy_trend = _quantile_table(
            d_en,
            metric="energy_uj",
            group_cols=["round", "task"],
            prefix="energy_uj",
            min_n=min_n,
        )
    else:
        energy_trend = pd.DataFrame(
            columns=[
                "round",
                "task",
                "energy_uj_n",
                "energy_uj_median",
                "energy_uj_q25",
                "energy_uj_q75",
                "energy_uj_q10",
                "energy_uj_q90",
                "energy_uj_min",
                "energy_uj_max",
                "energy_uj_iqr",
                "energy_uj_low_n",
            ]
        )

    # Indexierte Trends (zusätzliche Tabellen; brechen Plots nicht)
    latency_trend_indexed = _index_trend_table(latency_trend, metric_prefix="latency_us")
    energy_trend_indexed = _index_trend_table(energy_trend, metric_prefix="energy_uj")

    return {
        "coverage_round_task_all": cov_all,
        "unknown_summary_by_round": unk_sum,
        "coverage_energy_by_round_task": cov_energy,
        "coverage_power_by_round_task": cov_power,
        "coverage_quality_by_round_task_accuracy": cov_quality,
        "trend_latency_us_round_task": latency_trend,
        "trend_energy_uj_round_task": energy_trend,
        "trend_latency_us_round_task_indexed": latency_trend_indexed,
        "trend_energy_uj_round_task_indexed": energy_trend_indexed,
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
    _ensure_dirs()
    df = load_dataset(parquet_path)
    tables = build_core_tables(df, min_n=min_n)
    save_tables(tables)


if __name__ == "__main__":
    run_eda()
