from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]
CORE_TASKS = ["AD", "IC", "KWS", "VWW"]

# Option A: Baseline-Kandidaten erst ab dieser Round zulassen
MIN_BASELINE_ROUND = "v0.7"


def _round_rank(r: str) -> int:
    try:
        return ROUND_ORDER.index(str(r))
    except ValueError:
        return 10_000


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _task_label(row: pd.Series) -> str:
    """
    Label-Regel für Coverage:
    - OUT_OF_SCOPE bleibt OUT_OF_SCOPE (als eigene Kategorie)
    - fehlende / unbekannte Tasks -> UNKNOWN
    - sonst task_canon (AD/IC/KWS/VWW)
    """
    scope_status = row.get("scope_status", None)
    task_canon = row.get("task_canon", None)

    if pd.notna(scope_status) and str(scope_status) == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"

    if pd.isna(task_canon) or str(task_canon).strip() == "" or str(task_canon) == "UNKNOWN":
        return "UNKNOWN"

    return str(task_canon)


def _write_table(df: pd.DataFrame, out_path: Path, tag: str) -> None:
    df.to_csv(out_path, index=False)
    print(f"[table] {tag}: {len(df)} rows -> {out_path}")


def _coverage_round_task_all(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["task_label"] = tmp.apply(_task_label, axis=1)

    g = tmp.groupby(["round", "task_label"], dropna=False).size().reset_index(name="rows")

    pivot = (
        g.pivot_table(index="round", columns="task_label", values="rows", fill_value=0, aggfunc="sum")
        .reset_index()
    )

    for c in CORE_TASKS + ["UNKNOWN", "OUT_OF_SCOPE"]:
        if c not in pivot.columns:
            pivot[c] = 0

    pivot = pivot[["round"] + CORE_TASKS + ["UNKNOWN", "OUT_OF_SCOPE"]]
    pivot["round_rank"] = pivot["round"].map(_round_rank)
    pivot = pivot.sort_values("round_rank").drop(columns=["round_rank"]).reset_index(drop=True)
    return pivot


def _unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["is_unknown"] = tmp["scope_status"].eq("UNKNOWN") if "scope_status" in tmp.columns else False

    g = tmp.groupby("round").agg(
        rows_total=("round", "size"),
        rows_unknown=("is_unknown", "sum"),
    ).reset_index()

    g["share_unknown"] = np.where(g["rows_total"] > 0, g["rows_unknown"] / g["rows_total"], np.nan)
    g["round_rank"] = g["round"].map(_round_rank)
    g = g.sort_values("round_rank").drop(columns=["round_rank"]).reset_index(drop=True)
    return g


def _metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Coverage je Round×Task_label (inkl. UNKNOWN/OUT_OF_SCOPE als Kategorien).
    """
    tmp = df.copy()
    tmp["task_label"] = tmp.apply(_task_label, axis=1)

    if metric not in tmp.columns:
        tmp["_has_metric"] = False
    else:
        tmp["_has_metric"] = tmp[metric].notna()

    g = tmp.groupby(["round", "task_label"]).agg(
        rows=("task_label", "size"),
        rows_with_metric=("_has_metric", "sum"),
    ).reset_index()

    g["share_metric"] = np.where(g["rows"] > 0, g["rows_with_metric"] / g["rows"], np.nan)
    g["round_rank"] = g["round"].map(_round_rank)
    g = g.sort_values(["round_rank", "task_label"]).drop(columns=["round_rank"]).reset_index(drop=True)
    return g


def _trend_round_task(df: pd.DataFrame, metric: str, min_n: int) -> pd.DataFrame:
    """
    Trendtabellen nur für IN_SCOPE + CORE_TASKS.
    """
    tmp = df.copy()

    tmp = tmp[
        (tmp["scope_status"] == "IN_SCOPE")
        & (tmp["task_canon"].isin(CORE_TASKS))
        & (tmp[metric].notna())
    ].copy()

    def q(x: pd.Series, p: float) -> float:
        return float(np.quantile(x.to_numpy(), p))

    g = tmp.groupby(["round", "task_canon"])[metric].agg(
        n="count",
        median="median",
        q10=lambda x: q(x, 0.10),
        q25=lambda x: q(x, 0.25),
        q75=lambda x: q(x, 0.75),
        q90=lambda x: q(x, 0.90),
        min="min",
        max="max",
    ).reset_index()

    g["iqr"] = g["q75"] - g["q25"]
    g["low_n"] = g["n"] < int(min_n)

    g["round_rank"] = g["round"].map(_round_rank)
    g["task_rank"] = g["task_canon"].map({t: i for i, t in enumerate(CORE_TASKS)})
    g = g.sort_values(["round_rank", "task_rank"]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)

    g = g.rename(
        columns={
            "task_canon": "task",
            "n": f"{metric}_n",
            "median": f"{metric}_median",
            "q10": f"{metric}_q10",
            "q25": f"{metric}_q25",
            "q75": f"{metric}_q75",
            "q90": f"{metric}_q90",
            "min": f"{metric}_min",
            "max": f"{metric}_max",
            "iqr": f"{metric}_iqr",
            "low_n": f"{metric}_low_n",
        }
    )
    return g


def _trend_round_task_factor(df: pd.DataFrame, metric: str, min_n: int, factor_col: str) -> pd.DataFrame:
    """
    FF1b/FF1c: Trendtabellen für IN_SCOPE + CORE_TASKS zusätzlich nach Faktor.
    (Minimal: deskriptiv, keine Modellierung)
    """
    tmp = df.copy()

    if factor_col not in tmp.columns:
        tmp[factor_col] = "UNKNOWN"

    # NaNs in Faktor explizit labeln, damit grouping stabil ist
    tmp[factor_col] = tmp[factor_col].astype("object").where(tmp[factor_col].notna(), "UNKNOWN")

    tmp = tmp[
        (tmp["scope_status"] == "IN_SCOPE")
        & (tmp["task_canon"].isin(CORE_TASKS))
        & (tmp[metric].notna())
    ].copy()

    def q(x: pd.Series, p: float) -> float:
        return float(np.quantile(x.to_numpy(), p))

    g = tmp.groupby(["round", "task_canon", factor_col])[metric].agg(
        n="count",
        median="median",
        q10=lambda x: q(x, 0.10),
        q25=lambda x: q(x, 0.25),
        q75=lambda x: q(x, 0.75),
        q90=lambda x: q(x, 0.90),
        min="min",
        max="max",
    ).reset_index()

    g["iqr"] = g["q75"] - g["q25"]
    g["low_n"] = g["n"] < int(min_n)

    g["round_rank"] = g["round"].map(_round_rank)
    g["task_rank"] = g["task_canon"].map({t: i for i, t in enumerate(CORE_TASKS)})
    g = g.sort_values(["round_rank", "task_rank", factor_col]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)

    g = g.rename(
        columns={
            "task_canon": "task",
            "n": f"{metric}_n",
            "median": f"{metric}_median",
            "q10": f"{metric}_q10",
            "q25": f"{metric}_q25",
            "q75": f"{metric}_q75",
            "q90": f"{metric}_q90",
            "min": f"{metric}_min",
            "max": f"{metric}_max",
            "iqr": f"{metric}_iqr",
            "low_n": f"{metric}_low_n",
            factor_col: factor_col,
        }
    )
    return g


def _index_trend_table(trend: pd.DataFrame, metric: str, min_n: int) -> pd.DataFrame:
    """
    Option A: Baseline-Kandidaten erst ab MIN_BASELINE_ROUND zulassen.

    Baseline pro Metrik×Task:
      - erste Round (nach ROUND_ORDER) mit n>=min_n und median vorhanden
      - ABER nur für Rounds mit round_rank >= round_rank(MIN_BASELINE_ROUND)

    Pre-Baseline wird in Index-Spalten ausgeblendet (NaN), damit Plots/Checks nicht irritieren.
    """
    n_col = f"{metric}_n"
    med_col = f"{metric}_median"

    min_base_rank = _round_rank(MIN_BASELINE_ROUND)

    baseline_map: Dict[str, Tuple[Optional[str], Optional[int], Optional[float]]] = {}

    for task, g in trend.groupby("task", sort=False):
        g2 = g.copy()
        g2["round_rank"] = g2["round"].map(_round_rank)
        g2 = g2.sort_values("round_rank")

        # Option A: Baseline-Kandidaten ab MIN_BASELINE_ROUND
        base_row = g2[
            (g2["round_rank"] >= min_base_rank)
            & (g2[n_col] >= int(min_n))
            & (g2[med_col].notna())
        ].head(1)

        if len(base_row) == 0:
            baseline_map[task] = (None, None, None)
        else:
            r0 = str(base_row["round"].iloc[0])
            n0 = int(base_row[n_col].iloc[0])
            m0 = float(base_row[med_col].iloc[0])
            baseline_map[task] = (r0, n0, m0)

    out = trend.copy()
    out["baseline_round"] = out["task"].map(lambda t: baseline_map.get(t, (None, None, None))[0])
    out["baseline_n"] = out["task"].map(lambda t: baseline_map.get(t, (None, None, None))[1])
    out["baseline_median"] = out["task"].map(lambda t: baseline_map.get(t, (None, None, None))[2])

    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return np.where((b.notna()) & (b != 0), a / b, np.nan)

    out[f"{metric}_median_index"] = safe_div(out[med_col], out["baseline_median"])
    out[f"{metric}_q25_index"] = safe_div(out[f"{metric}_q25"], out["baseline_median"])
    out[f"{metric}_q75_index"] = safe_div(out[f"{metric}_q75"], out["baseline_median"])
    out[f"{metric}_q10_index"] = safe_div(out[f"{metric}_q10"], out["baseline_median"])
    out[f"{metric}_q90_index"] = safe_div(out[f"{metric}_q90"], out["baseline_median"])

    # Pre-Baseline ausblenden: round < baseline_round => Index-Spalten = NaN
    out["round_rank"] = out["round"].map(_round_rank)
    base_rank = out["baseline_round"].map(lambda r: _round_rank(r) if isinstance(r, str) else np.nan)

    out["pre_baseline"] = np.where(base_rank.notna() & (out["round_rank"] < base_rank), True, False)

    idx_cols = [
        f"{metric}_median_index",
        f"{metric}_q25_index",
        f"{metric}_q75_index",
        f"{metric}_q10_index",
        f"{metric}_q90_index",
    ]
    out.loc[out["pre_baseline"] == True, idx_cols] = np.nan

    out = out.drop(columns=["round_rank"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/mlperf_tiny_raw.parquet", help="Input parquet")
    ap.add_argument("--tables-dir", default="reports/tables", help="Output directory for tables")
    ap.add_argument("--min-n", type=int, default=5, help="Minimum n for 'stable' statistics / baseline selection")
    args = ap.parse_args()

    data_path = Path(args.data)
    tables_dir = Path(args.tables_dir)
    _ensure_dirs(tables_dir)

    df = pd.read_parquet(data_path)

    # Optional: Features ergänzen, falls Parquet die neuen Spalten noch nicht enthält
    need_cols = {"accelerator_present", "processor_family", "software_stack"}
    if not need_cols.issubset(set(df.columns)):
        try:
            from src.features import add_features
            df = add_features(df)
        except Exception:
            # bewusst kein harter Fail: EDA kann weiterlaufen, Faktor-Tabellen werden dann ggf. "UNKNOWN"
            pass

    # A) Coverage & Data quality
    cov_all = _coverage_round_task_all(df)
    _write_table(cov_all, tables_dir / "coverage_round_task_all.csv", "coverage_round_task_all")

    unk = _unknown_summary_by_round(df)
    _write_table(unk, tables_dir / "unknown_summary_by_round.csv", "unknown_summary_by_round")

    # B) Metric coverage (inkl. UNKNOWN/OUT_OF_SCOPE)
    #    (Accuracy nur EINMAL: coverage_accuracy_by_round_task.csv)
    metrics_general = ["energy_uj", "power_mw", "auc"]
    for metric in metrics_general:
        cov = _metric_coverage_by_round_task(df, metric=metric)
        _write_table(cov, tables_dir / f"coverage_{metric}_by_round_task.csv", f"coverage_{metric}_by_round_task")

    cov_acc = _metric_coverage_by_round_task(df, metric="accuracy")
    _write_table(cov_acc, tables_dir / "coverage_accuracy_by_round_task.csv", "coverage_accuracy_by_round_task")

    # C) Trendtabellen + Index
    min_n = int(args.min_n)

    for metric in ["latency_us", "energy_uj"]:
        if metric not in df.columns:
            continue

        trend = _trend_round_task(df, metric=metric, min_n=min_n)
        _write_table(trend, tables_dir / f"trend_{metric}_round_task.csv", f"trend_{metric}_round_task")

        trend_idx = _index_trend_table(trend, metric=metric, min_n=min_n)
        _write_table(trend_idx, tables_dir / f"trend_{metric}_round_task_indexed.csv", f"trend_{metric}_round_task_indexed")

    # D) FF1b/FF1c Faktor-Tabellen (minimal, kein Overengineering)
    #    Nur wenn Basis-Metrik vorhanden ist.
    for metric in ["latency_us", "energy_uj"]:
        if metric not in df.columns:
            continue

        t_acc = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="accelerator_present")
        _write_table(t_acc, tables_dir / f"trend_{metric}_round_task_accelerator_present.csv", f"trend_{metric}_round_task_accelerator_present")

        t_cpu = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="processor_family")
        _write_table(t_cpu, tables_dir / f"trend_{metric}_round_task_processor_family.csv", f"trend_{metric}_round_task_processor_family")

        t_sw = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="software_stack")
        _write_table(t_sw, tables_dir / f"trend_{metric}_round_task_software_stack.csv", f"trend_{metric}_round_task_software_stack")


if __name__ == "__main__":
    main()
