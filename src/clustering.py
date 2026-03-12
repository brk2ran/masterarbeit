from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]
CORE_TASKS = ["AD", "IC", "KWS", "VWW"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_table(df: pd.DataFrame, path: Path, tag: str) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[table] {tag}: {len(df)} rows -> {path}")


def _round_rank(r: object) -> int:
    try:
        return ROUND_ORDER.index(str(r))
    except ValueError:
        return 10_000


def _sorted_round(df: pd.DataFrame, col: str = "round") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df["_round_rank"] = df[col].map(_round_rank)
        df = df.sort_values(["_round_rank"] + [c for c in df.columns if c != "_round_rank"]).drop(
            columns=["_round_rank"]
        )
    return df


def _col_or_default(df: pd.DataFrame, col: str, default: object) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _pick_system_id(df: pd.DataFrame) -> pd.Series:
    if "public_id" in df.columns:
        return df["public_id"].astype(str)

    parts = []
    for c in ["round", "system_name", "platform", "vendor", "processor_family", "host_processor_frequency"]:
        if c in df.columns:
            parts.append(df[c].astype(str).fillna("NA"))

    if not parts:
        return df.index.astype(str)

    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    return key


def _mode(series: pd.Series) -> object:
    s = series.dropna()
    if s.empty:
        return np.nan
    vc = s.astype(str).value_counts()
    return vc.index[0] if not vc.empty else np.nan


def _build_cluster_class_map(profiles: pd.DataFrame) -> pd.DataFrame:
    p = profiles.copy()
    p["_lat_sort"] = pd.to_numeric(p["latency_us_median"], errors="coerce").fillna(np.inf)
    p = p.sort_values(["_lat_sort", "cluster_id"], ascending=[True, True]).reset_index(drop=True)
    p["cluster_rank"] = np.arange(1, len(p) + 1, dtype=int)
    p["cluster_class"] = p["cluster_rank"].map(lambda i: f"C{i}")
    return p[["cluster_id", "cluster_rank", "cluster_class"]]


def _build_system_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["system_id"] = _pick_system_id(d)

    d["accelerator_present"] = _col_or_default(d, "accelerator_present", "UNKNOWN")
    d["accelerator_present"] = d["accelerator_present"].map({True: "True", False: "False"}).fillna(
        d["accelerator_present"].astype(str).replace({"nan": "UNKNOWN"})
    )

    d["processor_family"] = _col_or_default(d, "processor_family", "UNKNOWN").astype(str).replace({"nan": "UNKNOWN"})
    d["cpu_freq_bucket"] = _col_or_default(d, "cpu_freq_bucket", "UNKNOWN").astype(str).replace({"nan": "UNKNOWN"})

    agg = (
        d.groupby("system_id", dropna=False)
        .agg(
            round=("round", _mode),
            accelerator_present=("accelerator_present", _mode),
            processor_family=("processor_family", _mode),
            cpu_freq_bucket=("cpu_freq_bucket", _mode),
            systems_rows=("system_id", "size"),
        )
        .reset_index()
    )
    return agg


def _cluster_systems_kmeans(system_df: pd.DataFrame, k: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import silhouette_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as e:
        raise RuntimeError(
            "scikit-learn required for FF2 clustering."
            f" Original error: {e}"
        )

    cat_cols = ["accelerator_present", "processor_family", "cpu_freq_bucket"]
    X = system_df[cat_cols].fillna("UNKNOWN")

    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
    )
    model = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto")

    pipe = Pipeline([("pre", pre), ("kmeans", model)])
    labels = pipe.fit_predict(X)

    meta: Dict[str, float] = {}
    try:
        Xt = pre.fit_transform(X)
        if Xt.shape[0] > k and k >= 2:
            meta["silhouette"] = float(silhouette_score(Xt, labels))
    except Exception:
        pass

    out = system_df.copy()
    out["cluster_id"] = labels.astype(int)
    return out, meta


def _choose_k(system_df: pd.DataFrame, seed: int) -> Tuple[int, Dict[int, float]]:
    try:
        from sklearn.metrics import silhouette_score  # noqa: F401
    except Exception:
        return 6, {}

    n = len(system_df)
    if n < 6:
        return max(2, min(3, n)), {}

    ks = [k for k in range(3, 9) if k < n]
    scores: Dict[int, float] = {}
    best_k = ks[0]
    best_s = -1.0

    for k in ks:
        _, meta = _cluster_systems_kmeans(system_df, k=k, seed=seed)
        s = meta.get("silhouette", np.nan)
        if pd.notna(s):
            scores[k] = float(s)
            if float(s) > best_s:
                best_s = float(s)
                best_k = k

    if best_s < 0 and ks:
        best_k = 6 if 6 in ks else (4 if 4 in ks else ks[0])

    return best_k, scores


def _pareto_front_2d(lat: np.ndarray, en: np.ndarray) -> np.ndarray:
    idx = np.argsort(lat, kind="mergesort")
    en_s = en[idx]

    best = np.inf
    keep_s = np.zeros_like(en_s, dtype=bool)
    for i in range(len(en_s)):
        if en_s[i] < best:
            keep_s[i] = True
            best = en_s[i]

    keep = np.zeros_like(keep_s)
    keep[idx] = keep_s
    return keep


def _pareto_by_round_task(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["task"] = _col_or_default(d, "task_canon", _col_or_default(d, "task", "UNKNOWN")).astype(str)
    d["round"] = _col_or_default(d, "round", "UNKNOWN").astype(str)

    d["_has_both"] = d["latency_us"].notna() & d["energy_uj"].notna()
    d["pareto_flag"] = 0

    for (r, t), sub in d[d["_has_both"]].groupby(["round", "task"], dropna=False):
        lat = sub["latency_us"].to_numpy(dtype=float)
        en = sub["energy_uj"].to_numpy(dtype=float)
        keep = _pareto_front_2d(lat, en)
        d.loc[sub.index, "pareto_flag"] = keep.astype(int)

    d = d.drop(columns=["_has_both"])
    return d


def build_ff2(
    parquet_path: Path,
    reports_dir: Path,
    min_n: int = 5,
    seed: int = 7,
    k: Optional[int] = 6,
) -> None:
    df = pd.read_parquet(parquet_path)

    if "scope_status" in df.columns:
        df = df[df["scope_status"] == "IN_SCOPE"].copy()

    df["task"] = _col_or_default(df, "task_canon", _col_or_default(df, "task", "UNKNOWN")).astype(str)
    df = df[df["task"].isin(CORE_TASKS)].copy()

    if "latency_us" not in df.columns:
        raise RuntimeError("latency_us missing in consolidated dataset.")
    if "energy_uj" not in df.columns:
        df["energy_uj"] = np.nan

    tables_dir = reports_dir / "tables"
    _ensure_dir(tables_dir)

    sys_df = _build_system_table(df)

    chosen_k = k
    k_scores: Dict[int, float] = {}
    if chosen_k is None:
        chosen_k, k_scores = _choose_k(sys_df, seed=seed)

    sys_clustered, meta = _cluster_systems_kmeans(sys_df, k=int(chosen_k), seed=seed)
    if k_scores:
        print("INFO: silhouette_scores", {k: round(v, 4) for k, v in k_scores.items()})
    if "silhouette" in meta:
        print(f"INFO: chosen_k={chosen_k} silhouette={meta['silhouette']:.4f}")
    else:
        print(f"INFO: chosen_k={chosen_k}")

    df = df.copy()
    df["system_id"] = _pick_system_id(df)
    df = df.merge(sys_clustered[["system_id", "cluster_id"]], on="system_id", how="left")

    prof = (
        sys_clustered.groupby("cluster_id", dropna=False)
        .agg(
            systems=("system_id", "size"),
            accelerator_present=("accelerator_present", _mode),
            processor_family=("processor_family", _mode),
            cpu_freq_bucket=("cpu_freq_bucket", _mode),
        )
        .reset_index()
    )

    metric_prof = (
        df.groupby("cluster_id", dropna=False)
        .agg(
            latency_us_median=("latency_us", "median"),
            energy_uj_median=("energy_uj", "median"),
            latency_us_n=("latency_us", lambda s: int(s.notna().sum())),
            energy_uj_n=("energy_uj", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )

    prof = prof.merge(metric_prof, on="cluster_id", how="left")

    class_map = _build_cluster_class_map(prof)
    prof = (
        prof.merge(class_map, on="cluster_id", how="left")
        .sort_values(["cluster_rank", "cluster_id"])
        .reset_index(drop=True)
    )
    _write_table(prof, tables_dir / "ff2_cluster_profiles.csv", "ff2_cluster_profiles")

    df = df.merge(class_map, on="cluster_id", how="left")
    assign_cols = [
        c
        for c in ["system_id", "round", "task", "cluster_id", "cluster_class", "cluster_rank", "latency_us", "energy_uj"]
        if c in df.columns
    ]
    _write_table(df[assign_cols].copy(), tables_dir / "ff2_cluster_assignments.csv", "ff2_cluster_assignments")

    share = (
        df.groupby(["round", "task", "cluster_id", "cluster_class", "cluster_rank"], dropna=False)
        .agg(rows=("system_id", "size"))
        .reset_index()
    )
    totals = share.groupby(["round", "task"], dropna=False)["rows"].sum().rename("rows_total").reset_index()
    share = share.merge(totals, on=["round", "task"], how="left")
    share["share"] = np.where(share["rows_total"] > 0, share["rows"] / share["rows_total"], np.nan)
    share = (
        _sorted_round(share, "round")
        .sort_values(["round", "task", "cluster_rank", "cluster_id"])
        .reset_index(drop=True)
    )
    _write_table(share, tables_dir / "ff2_cluster_share_by_round_task.csv", "ff2_cluster_share_by_round_task")

    summ = (
        df.groupby(["round", "task", "cluster_id", "cluster_class", "cluster_rank"], dropna=False)
        .agg(
            rows=("system_id", "size"),
            latency_us_median=("latency_us", "median"),
            energy_uj_median=("energy_uj", "median"),
            latency_us_n=("latency_us", lambda s: int(s.notna().sum())),
            energy_uj_n=("energy_uj", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    summ["low_n_energy"] = summ["energy_uj_n"] < int(min_n)
    summ = (
        _sorted_round(summ, "round")
        .sort_values(["round", "task", "cluster_rank", "cluster_id"])
        .reset_index(drop=True)
    )
    _write_table(summ, tables_dir / "ff2_cluster_metric_summary_by_round_task.csv", "ff2_cluster_metric_summary_by_round_task")

    pareto_points = _pareto_by_round_task(df)

    _write_table(
        pareto_points[
            ["round", "task", "system_id", "cluster_id", "cluster_class", "cluster_rank", "latency_us", "energy_uj", "pareto_flag"]
        ],
        tables_dir / "ff2_pareto_points_round_task.csv",
        "ff2_pareto_points_round_task",
    )

    tmp = pareto_points.copy()
    tmp["has_both"] = tmp["latency_us"].notna() & tmp["energy_uj"].notna()

    summary = (
        tmp.groupby(["round", "task"], dropna=False)
        .agg(
            rows_total=("system_id", "size"),
            rows_with_both=("has_both", "sum"),
            pareto_points=("pareto_flag", "sum"),
        )
        .reset_index()
    )
    summary["share_pareto_among_both"] = np.where(
        summary["rows_with_both"] > 0,
        summary["pareto_points"] / summary["rows_with_both"],
        np.nan,
    )
    summary["low_n_energy"] = summary["rows_with_both"] < int(min_n)
    summary = _sorted_round(summary, "round")
    _write_table(summary, tables_dir / "ff2_pareto_summary_round_task.csv", "ff2_pareto_summary_round_task")

    byc = (
        tmp[tmp["has_both"]]
        .groupby(["round", "task", "cluster_id", "cluster_class", "cluster_rank"], dropna=False)
        .agg(
            rows_with_both=("has_both", "sum"),
            pareto_points=("pareto_flag", "sum"),
        )
        .reset_index()
    )
    byc["share_pareto"] = np.where(byc["rows_with_both"] > 0, byc["pareto_points"] / byc["rows_with_both"], np.nan)
    byc["low_n_energy"] = byc["rows_with_both"] < int(min_n)
    byc = (
        _sorted_round(byc, "round")
        .sort_values(["round", "task", "cluster_rank", "cluster_id"])
        .reset_index(drop=True)
    )
    _write_table(byc, tables_dir / "ff2_pareto_by_cluster_round_task.csv", "ff2_pareto_by_cluster_round_task")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FF2 clustering and Pareto table generation.")
    p.add_argument("--parquet", type=str, default="data/interim/mlperf_tiny_raw.parquet", help="Input parquet path")
    p.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    p.add_argument("--min-n", type=int, default=5, help="Low-n threshold for energy-based summaries")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--k", type=int, default=6, help="Number of clusters")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_ff2(
        parquet_path=Path(a.parquet),
        reports_dir=Path(a.reports_dir),
        min_n=int(a.min_n),
        seed=int(a.seed),
        k=a.k,
    )


if __name__ == "__main__":
    main()