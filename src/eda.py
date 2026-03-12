from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]
CORE_TASKS = ["AD", "IC", "KWS", "VWW"]
MIN_BASELINE_ROUND = "v0.7"


def _round_rank(r: str) -> int:
    try:
        return ROUND_ORDER.index(str(r))
    except ValueError:
        return 10_000


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _write_table(df: pd.DataFrame, out_path: Path, tag: str) -> None:
    df.to_csv(out_path, index=False)
    print(f"[table] {tag}: {len(df)} rows -> {out_path}")


def _task_label(row: pd.Series) -> str:
    scope_status = row.get("scope_status", None)
    task_canon = row.get("task_canon", None)

    if pd.notna(scope_status) and str(scope_status) == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"

    if pd.isna(task_canon) or str(task_canon).strip() == "" or str(task_canon) == "UNKNOWN":
        return "UNKNOWN"

    return str(task_canon)


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


def _coverage_latency_energy_by_round_task(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    tmp = tmp[
        (tmp.get("scope_status") == "IN_SCOPE")
        & (tmp.get("task_canon").isin(CORE_TASKS))
        & (tmp.get("latency_us").notna())
    ].copy()

    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "round", "task", "rows_total",
                "latency_us_n", "latency_us_share",
                "energy_uj_n", "energy_uj_share",
            ]
        )

    g = tmp.groupby(["round", "task_canon"], dropna=False).agg(
        rows_total=("latency_us", "size"),
        latency_us_n=("latency_us", lambda s: int(s.notna().sum())),
        energy_uj_n=("energy_uj", lambda s: int(s.notna().sum())) if "energy_uj" in tmp.columns else ("latency_us", lambda s: 0),
    ).reset_index()

    g["latency_us_share"] = np.where(g["rows_total"] > 0, g["latency_us_n"] / g["rows_total"], np.nan)
    g["energy_uj_share"] = np.where(g["rows_total"] > 0, g["energy_uj_n"] / g["rows_total"], np.nan)

    g = g.rename(columns={"task_canon": "task"})
    g["round_rank"] = g["round"].map(_round_rank)
    g["task_rank"] = g["task"].map({t: i for i, t in enumerate(CORE_TASKS)})
    g = g.sort_values(["round_rank", "task_rank"]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)
    return g


def _trend_round_task(df: pd.DataFrame, metric: str, min_n: int) -> pd.DataFrame:
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
    tmp = df.copy()

    if factor_col not in tmp.columns:
        tmp[factor_col] = "UNKNOWN"

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
    g = (
        g.sort_values(["round_rank", "task_rank", factor_col])
        .drop(columns=["round_rank", "task_rank"])
        .reset_index(drop=True)
    )

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
    n_col = f"{metric}_n"
    med_col = f"{metric}_median"

    min_base_rank = _round_rank(MIN_BASELINE_ROUND)
    baseline_map: Dict[str, Tuple[Optional[str], Optional[int], Optional[float]]] = {}

    for task, g in trend.groupby("task", sort=False):
        g2 = g.copy()
        g2["round_rank"] = g2["round"].map(_round_rank)
        g2 = g2.sort_values("round_rank")

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


def _accel_effect_table(trend_factor: pd.DataFrame, metric: str) -> pd.DataFrame:
    med = f"{metric}_median"
    ncol = f"{metric}_n"

    df = trend_factor.copy()
    if "accelerator_present" not in df.columns:
        return df.iloc[0:0].copy()

    df["accel"] = df["accelerator_present"].astype(str)
    piv = df.pivot_table(
        index=["round", "task"],
        columns="accel",
        values=[med, ncol],
        aggfunc="first",
    )

    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    m_t = piv.get(f"{med}__True")
    m_f = piv.get(f"{med}__False")
    n_t = piv.get(f"{ncol}__True")
    n_f = piv.get(f"{ncol}__False")

    out = piv[["round", "task"]].copy()
    out[f"{metric}_median_true"] = m_t
    out[f"{metric}_median_false"] = m_f
    out[f"{metric}_n_true"] = n_t
    out[f"{metric}_n_false"] = n_f

    out[f"{metric}_delta_true_minus_false"] = pd.to_numeric(m_t, errors="coerce") - pd.to_numeric(m_f, errors="coerce")
    out[f"{metric}_ratio_true_over_false"] = np.where(
        pd.to_numeric(m_f, errors="coerce").notna() & (pd.to_numeric(m_f, errors="coerce") != 0),
        pd.to_numeric(m_t, errors="coerce") / pd.to_numeric(m_f, errors="coerce"),
        np.nan,
    )

    out["round_rank"] = out["round"].map(_round_rank)
    out["task_rank"] = out["task"].map({t: i for i, t in enumerate(CORE_TASKS)})
    out = out.sort_values(["round_rank", "task_rank"]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)
    return out


def _span_effect_table(trend_factor: pd.DataFrame, metric: str, factor_col: str) -> pd.DataFrame:
    med = f"{metric}_median"
    ncol = f"{metric}_n"

    df = trend_factor.copy()
    if factor_col not in df.columns:
        return df.iloc[0:0].copy()

    df[factor_col] = df[factor_col].astype("object").where(df[factor_col].notna(), "UNKNOWN")
    df[med] = pd.to_numeric(df[med], errors="coerce")
    df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    idx_min = df.groupby(["round", "task"])[med].idxmin()
    idx_max = df.groupby(["round", "task"])[med].idxmax()

    min_rows = df.loc[idx_min, ["round", "task", factor_col, med, ncol]].rename(
        columns={
            factor_col: "best_factor",
            med: f"{metric}_best_median",
            ncol: f"{metric}_best_n",
        }
    )
    max_rows = df.loc[idx_max, ["round", "task", factor_col, med, ncol]].rename(
        columns={
            factor_col: "worst_factor",
            med: f"{metric}_worst_median",
            ncol: f"{metric}_worst_n",
        }
    )

    out = pd.merge(min_rows, max_rows, on=["round", "task"], how="outer")

    out[f"{metric}_delta_worst_minus_best"] = out[f"{metric}_worst_median"] - out[f"{metric}_best_median"]
    out[f"{metric}_ratio_worst_over_best"] = np.where(
        out[f"{metric}_best_median"].notna() & (out[f"{metric}_best_median"] != 0),
        out[f"{metric}_worst_median"] / out[f"{metric}_best_median"],
        np.nan,
    )

    out["round_rank"] = out["round"].map(_round_rank)
    out["task_rank"] = out["task"].map({t: i for i, t in enumerate(CORE_TASKS)})
    out = out.sort_values(["round_rank", "task_rank"]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)
    return out


_SOFTWARE_TOKEN_SPLIT_RE = re.compile(r"(;|,|\s*\+\s*|\s*/\s*)")

_SOFTWARE_FAMILY_RULES = [
    (re.compile(r"cmsis[-\s]?nn", re.IGNORECASE), "CMSIS-NN"),
    (re.compile(r"(tensorflow\s*lite\s*for\s*microcontrollers|\btflm\b|tensorflowlite)", re.IGNORECASE), "TFLM"),
    (re.compile(r"(\btvm\b|microtvm)", re.IGNORECASE), "TVM"),
    (re.compile(r"x[-\s]?cube[-\s]?ai", re.IGNORECASE), "X-CUBE-AI"),
    (re.compile(r"syntiant.*(sdk|tdk)|\bsyntiant\s+tdk\b", re.IGNORECASE), "Syntiant SDK/TDK"),
    (re.compile(r"qualcomm\s+ai\s+stack", re.IGNORECASE), "Qualcomm AI Stack"),
    (re.compile(r"plumerai", re.IGNORECASE), "Plumerai"),
    (re.compile(r"\bfinn\b", re.IGNORECASE), "FINN"),
    (re.compile(r"\bleip\b", re.IGNORECASE), "LEIP"),
    (re.compile(r"self[-\s]?developed", re.IGNORECASE), "self-developed"),
]


def _split_software_values(val: object) -> list[str]:
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() == "null":
        return []
    parts = [p.strip() for p in _SOFTWARE_TOKEN_SPLIT_RE.split(s) if p and p.strip() and p.strip() not in {",", ";"}]
    return parts


def _software_family(token: object) -> str:
    if token is None:
        return "UNKNOWN"
    try:
        if pd.isna(token):
            return "UNKNOWN"
    except Exception:
        pass
    t = str(token).strip()
    if not t:
        return "UNKNOWN"
    for rx, label in _SOFTWARE_FAMILY_RULES:
        if rx.search(t):
            return label
    return "OTHER"


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA table generation for FF1 analyses.")
    ap.add_argument("--data", default="data/interim/mlperf_tiny_raw.parquet", help="Input parquet path")
    ap.add_argument("--tables-dir", default="reports/tables", help="Output directory for tables")
    ap.add_argument("--min-n", type=int, default=5, help="Low-n threshold / baseline threshold")
    args = ap.parse_args()

    data_path = Path(args.data)
    tables_dir = Path(args.tables_dir)
    _ensure_dirs(tables_dir)

    df = pd.read_parquet(data_path)

    need_cols = {
        "scope_status", "task_canon", "model_mlc_canon",
        "accelerator_present", "processor_family",
        "software_stack", "sw_cmsis_nn", "sw_tflm", "sw_tvm",
    }
    if not need_cols.issubset(set(df.columns)):
        try:
            from src.features import add_features
            df = add_features(df)
        except Exception:
            pass

    min_n = int(args.min_n)

    cov_all = _coverage_round_task_all(df)
    _write_table(cov_all, tables_dir / "coverage_round_task_all.csv", "coverage_round_task_all")

    unk = _unknown_summary_by_round(df)
    _write_table(unk, tables_dir / "unknown_summary_by_round.csv", "unknown_summary_by_round")

    for metric in ["energy_uj", "power_mw", "auc"]:
        cov = _metric_coverage_by_round_task(df, metric=metric)
        _write_table(cov, tables_dir / f"coverage_{metric}_by_round_task.csv", f"coverage_{metric}_by_round_task")

    cov_acc = _metric_coverage_by_round_task(df, metric="accuracy")
    _write_table(cov_acc, tables_dir / "coverage_accuracy_by_round_task.csv", "coverage_accuracy_by_round_task")

    cov_le = _coverage_latency_energy_by_round_task(df)
    _write_table(cov_le, tables_dir / "coverage_latency_energy_by_round_task.csv", "coverage_latency_energy_by_round_task")

    for metric in ["latency_us", "energy_uj"]:
        if metric not in df.columns:
            continue

        trend = _trend_round_task(df, metric=metric, min_n=min_n)
        _write_table(trend, tables_dir / f"trend_{metric}_round_task.csv", f"trend_{metric}_round_task")

        trend_idx = _index_trend_table(trend, metric=metric, min_n=min_n)
        _write_table(trend_idx, tables_dir / f"trend_{metric}_round_task_indexed.csv", f"trend_{metric}_round_task_indexed")

    for metric in ["latency_us", "energy_uj"]:
        if metric not in df.columns:
            continue

        t_acc = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="accelerator_present")
        _write_table(
            t_acc,
            tables_dir / f"trend_{metric}_round_task_accelerator_present.csv",
            f"trend_{metric}_round_task_accelerator_present",
        )

        t_cpu = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="processor_family")
        _write_table(
            t_cpu,
            tables_dir / f"trend_{metric}_round_task_processor_family.csv",
            f"trend_{metric}_round_task_processor_family",
        )

        if "cpu_freq_bucket" in df.columns:
            t_freq = _trend_round_task_factor(df, metric=metric, min_n=min_n, factor_col="cpu_freq_bucket")
            _write_table(
                t_freq,
                tables_dir / f"trend_{metric}_round_task_cpu_freq_bucket.csv",
                f"trend_{metric}_round_task_cpu_freq_bucket",
            )

        eff_acc = _accel_effect_table(t_acc, metric=metric)
        _write_table(
            eff_acc,
            tables_dir / f"ff1b_effect_{metric}_accelerator_present_round_task.csv",
            f"ff1b_effect_{metric}_accelerator_present_round_task",
        )

        eff_cpu_span = _span_effect_table(t_cpu, metric=metric, factor_col="processor_family")
        _write_table(
            eff_cpu_span,
            tables_dir / f"ff1b_span_{metric}_processor_family_round_task.csv",
            f"ff1b_span_{metric}_processor_family_round_task",
        )

        sw_src_col = "software" if "software" in df.columns else ("software_stack" if "software_stack" in df.columns else None)
        if sw_src_col is None:
            print("WARN: Software column missing; FF1c tables skipped")
        else:
            df_sw = df.copy()

            if sw_src_col == "software":
                df_sw["_software_token"] = df_sw[sw_src_col].apply(_split_software_values)
            else:
                df_sw["_software_token"] = df_sw[sw_src_col].apply(lambda x: [] if pd.isna(x) else [str(x)])

            df_sw = df_sw.explode("_software_token")
            df_sw["_software_family"] = df_sw["_software_token"].apply(_software_family)

            cov_sw_rt = (
                df_sw.groupby(["round", "task_canon", "_software_family"])["public_id"]
                .nunique()
                .reset_index(name="rows")
                .sort_values(["round", "task_canon", "rows"], ascending=[True, True, False])
            )
            _write_table(
                cov_sw_rt,
                tables_dir / "ff1c_coverage_software_family_by_round_task.csv",
                "ff1c_coverage_software_family_by_round_task",
            )

            cov_sw_r = (
                df_sw.groupby(["round", "_software_family"])["public_id"]
                .nunique()
                .reset_index(name="rows")
                .sort_values(["round", "rows"], ascending=[True, False])
            )
            _write_table(
                cov_sw_r,
                tables_dir / "ff1c_coverage_software_family_by_round.csv",
                "ff1c_coverage_software_family_by_round",
            )

            t_sw = _trend_round_task_factor(df_sw, metric=metric, min_n=min_n, factor_col="_software_family")
            _write_table(
                t_sw,
                tables_dir / f"ff1c_trend_{metric}_round_task_software_family.csv",
                f"ff1c_trend_{metric}_round_task_software_family",
            )


if __name__ == "__main__":
    main()