from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]
CORE_TASKS = ["AD", "IC", "KWS", "VWW"]
CAT_COLS = ["accelerator_present", "processor_family", "cpu_freq_bucket"]
SEED = 7


def _hamming_distance_matrix(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    k = centroids.shape[0]
    D = np.zeros((n, k), dtype=float)
    for j in range(k):
        D[:, j] = np.sum(X != centroids[j], axis=1)
    return D


def _update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    n_cols = X.shape[1]
    centroids = np.empty((k, n_cols), dtype=object)
    for j in range(k):
        mask = labels == j
        if not mask.any():
            centroids[j] = X[np.random.randint(0, len(X))]
            continue
        for c in range(n_cols):
            vals, counts = np.unique(X[mask, c], return_counts=True)
            centroids[j, c] = vals[np.argmax(counts)]
    return centroids


def kmodes_fit(
    X_cat: np.ndarray,
    k: int,
    seed: int = SEED,
    max_iter: int = 100,
    n_init: int = 10,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_cost = np.inf

    for _ in range(n_init):
        idx = rng.choice(len(X_cat), size=k, replace=False)
        centroids = X_cat[idx].copy()

        labels = np.zeros(len(X_cat), dtype=int)
        for _ in range(max_iter):
            D = _hamming_distance_matrix(X_cat, centroids)
            new_labels = np.argmin(D, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centroids = _update_centroids(X_cat, labels, k)

        cost = float(np.sum(np.min(_hamming_distance_matrix(X_cat, centroids), axis=1)))
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()

    return best_labels, best_cost


def _agreement_rate(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    n = len(labels_a)
    if n < 2:
        return 1.0

    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]

    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    agree = np.sum((same_a == same_b) & triu)
    total = np.sum(triu)
    return float(agree / total) if total > 0 else 1.0


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_a, labels_b))
    except Exception:
        return _agreement_rate(labels_a, labels_b)


def _mode(series: pd.Series) -> object:
    s = series.dropna()
    if s.empty:
        return "UNKNOWN"
    vc = s.astype(str).value_counts()
    return vc.index[0] if not vc.empty else "UNKNOWN"


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


def _build_system_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["system_id"] = _pick_system_id(d)

    d["accelerator_present"] = d.get("accelerator_present", pd.Series("UNKNOWN", index=d.index))
    d["accelerator_present"] = (
        d["accelerator_present"]
        .map({True: "True", False: "False"})
        .fillna(d["accelerator_present"].astype(str).replace({"nan": "UNKNOWN"}))
    )
    d["processor_family"] = (
        d.get("processor_family", pd.Series("UNKNOWN", index=d.index))
        .astype(str)
        .replace({"nan": "UNKNOWN"})
    )
    d["cpu_freq_bucket"] = (
        d.get("cpu_freq_bucket", pd.Series("UNKNOWN", index=d.index))
        .astype(str)
        .replace({"nan": "UNKNOWN"})
    )

    agg = (
        d.groupby("system_id", dropna=False)
        .agg(
            accelerator_present=("accelerator_present", _mode),
            processor_family=("processor_family", _mode),
            cpu_freq_bucket=("cpu_freq_bucket", _mode),
            systems_rows=("system_id", "size"),
        )
        .reset_index()
    )
    return agg


def _load_parquet_systems(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    if "scope_status" in df.columns:
        df = df[df["scope_status"] == "IN_SCOPE"].copy()

    if "task_canon" in df.columns:
        df = df[df["task_canon"].isin(CORE_TASKS)].copy()
    elif "task" in df.columns:
        df = df[df["task"].isin(CORE_TASKS)].copy()

    return _build_system_table(df)


def run_robustness_check(
    parquet_path: Path,
    assignments_path: Optional[Path],
    out_dir: Path,
    k: Optional[int] = None,
    seed: int = SEED,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_df = _load_parquet_systems(parquet_path)
    print(f"Systems total: {len(sys_df)}")

    kmeans_labels: Optional[np.ndarray] = None
    if assignments_path and assignments_path.exists():
        asgn = pd.read_csv(assignments_path)
        if "system_id" in asgn.columns and "cluster_id" in asgn.columns:
            asgn_sys = asgn.drop_duplicates("system_id")[["system_id", "cluster_id"]]
            sys_df = sys_df.merge(
                asgn_sys.rename(columns={"cluster_id": "cluster_kmeans"}),
                on="system_id",
                how="left",
            )
            kmeans_labels = sys_df["cluster_kmeans"].fillna(-1).to_numpy(dtype=int)
            print(
                f"K-Means labels loaded: {len(asgn_sys)} systems, "
                f"{sys_df['cluster_kmeans'].nunique()} clusters"
            )

    X_cat = sys_df[CAT_COLS].fillna("UNKNOWN").to_numpy(dtype=object)

    if k is None:
        if kmeans_labels is not None:
            k = int(pd.Series(kmeans_labels[kmeans_labels >= 0]).nunique())
            print(f"k inferred from K-Means labels: {k}")
        else:
            k = 6
            print(f"k fallback: {k}")

    print(f"Running K-Modes with k={k}, seed={seed} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmodes_labels, cost = kmodes_fit(X_cat, k=k, seed=seed)

    sys_df["cluster_kmodes"] = kmodes_labels
    print(f"K-Modes completed. Total cost: {cost:.1f}")

    if kmeans_labels is not None:
        valid = kmeans_labels >= 0
        ari = _adjusted_rand_index(kmeans_labels[valid], kmodes_labels[valid])
        rand = _agreement_rate(kmeans_labels[valid], kmodes_labels[valid])
        n_valid = int(valid.sum())

        print("\nAgreement analysis")
        print(f"Systems with K-Means label: {n_valid}")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Pairwise Rand Index: {rand:.4f}")

        comp = sys_df[
            [
                "system_id",
                "accelerator_present",
                "processor_family",
                "cpu_freq_bucket",
                "cluster_kmeans",
                "cluster_kmodes",
            ]
        ].copy()
        comp["labels_agree"] = comp["cluster_kmeans"] == comp["cluster_kmodes"]
        comp_path = out_dir / "ff2_kmodes_comparison.csv"
        comp.to_csv(comp_path, index=False)
        print(f"[table] ff2_kmodes_comparison: {len(comp)} rows -> {comp_path}")

        summary = pd.DataFrame(
            [
                {
                    "k": k,
                    "n_systems": n_valid,
                    "adjusted_rand_index": round(ari, 4),
                    "pairwise_rand_index": round(rand, 4),
                    "kmodes_total_cost": round(cost, 1),
                }
            ]
        )
        summ_path = out_dir / "ff2_kmodes_summary.csv"
        summary.to_csv(summ_path, index=False)
        print(f"[table] ff2_kmodes_summary: 1 row -> {summ_path}")

    else:
        print("No K-Means labels available. Writing K-Modes output only.")
        sys_df.to_csv(out_dir / "ff2_kmodes_only.csv", index=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Modes robustness check for FF2 clustering")
    p.add_argument(
        "--parquet",
        type=str,
        default="data/interim/mlperf_tiny_raw.parquet",
        help="Path to consolidated parquet file",
    )
    p.add_argument(
        "--assignments",
        type=str,
        default="reports/tables/ff2_cluster_assignments.csv",
        help="Path to ff2_cluster_assignments.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="reports/tables",
        help="Output directory for comparison tables",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters; inferred from K-Means assignments if omitted",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    run_robustness_check(
        parquet_path=Path(a.parquet),
        assignments_path=Path(a.assignments) if a.assignments else None,
        out_dir=Path(a.out_dir),
        k=a.k,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()