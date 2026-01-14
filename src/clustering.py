# src/clustering.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features import add_features

# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
OUT_CSV_DEFAULT = PROJECT_ROOT / "reports" / "tables" / "hardware_clusters.csv"

REQUIRED_COLS = ["latency_us", "energy_uj"]

# Stabiler System-Key: vermeidet Zusammenwerfen gleich benannter Systeme über Runden
SYSTEM_KEY = ["round", "public_id", "organization", "system_name"]


def _log(title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def run_clustering(
    parquet_path: Path = PARQUET_PATH_DEFAULT,
    out_csv: Path = OUT_CSV_DEFAULT,
    *,
    n_clusters: int = 3,
    min_systems: int = 10,
    random_state: int = 42,
) -> None:
    _log("Clustering: Systemklassen (IN_SCOPE, Latency+Energy, System-Aggregation)")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")

    print(f"[load] {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Features + Scope-Labels ergänzen
    df = add_features(df)

    # IN_SCOPE + beide Metriken vorhanden
    if "scope_status" not in df.columns:
        raise ValueError("Spalte 'scope_status' fehlt. Prüfe src/features.py:add_features().")

    mask = (df["scope_status"] == "IN_SCOPE") & df[REQUIRED_COLS].notna().all(axis=1)
    d = df.loc[mask].copy()

    print(f"Rows total: {len(df)}")
    print(f"Rows IN_SCOPE + (latency_us & energy_uj): {len(d)}")

    if d.empty:
        print("WARNUNG: Keine Datenpunkte für Clustering (IN_SCOPE + latency_us + energy_uj).")
        return

    # Coverage pro Round (Transparenz)
    cov = d.groupby("round").size().rename("rows").reset_index()
    print("\nCoverage pro Round (rows mit Latency+Energy):")
    print(cov.to_string(index=False))

    # Prüfe System-Key-Spalten
    missing_keys = [c for c in SYSTEM_KEY if c not in d.columns]
    if missing_keys:
        raise ValueError(f"System-Key Spalten fehlen im Datensatz: {missing_keys}")

    _log("Aggregation: System-Level (Median)")

    # Median je System
    sys_features = (
        d.groupby(SYSTEM_KEY, dropna=False)[REQUIRED_COLS]
        .median()
        .reset_index()
    )

    # Anzahl Roh-Zeilen pro System (n_rows)
    sys_n = d.groupby(SYSTEM_KEY, dropna=False).size().rename("n_rows").reset_index()
    sys_features = sys_features.merge(sys_n, on=SYSTEM_KEY, how="left")

    n_systems = len(sys_features)
    print(f"Anzahl Systeme (Key={SYSTEM_KEY}): {n_systems}")

    if n_systems < max(min_systems, n_clusters * 2):
        print(
            "WARNUNG: Zu wenige Systeme für robustes Clustering "
            f"(systeme={n_systems}, k={n_clusters})."
        )
        return

    # Log-Transform (typisch bei Größenordnungen) + Skalierung
    X = np.log1p(sys_features[REQUIRED_COLS].astype(float))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    sys_features["cluster_id"] = kmeans.fit_predict(X_scaled)

    # performance_class: 0 = schnellste (niedrigste median latency_us)
    cluster_order = (
        sys_features.groupby("cluster_id")["latency_us"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    mapping = {cid: i for i, cid in enumerate(cluster_order)}
    sys_features["performance_class"] = sys_features["cluster_id"].map(mapping)

    # Aufräumen
    sys_features = (
        sys_features.drop(columns=["cluster_id"])
        .sort_values(["performance_class", "round", "organization", "system_name"])
        .reset_index(drop=True)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sys_features.to_csv(out_csv, index=False, encoding="utf-8-sig")

    _log("Ergebnis")
    print(f"Gespeichert: {out_csv}")
    print("\nCluster-Statistiken (Median je performance_class):")
    print(sys_features.groupby("performance_class")[REQUIRED_COLS].median().to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KMeans Clustering (Systemklassen) für MLPerf-Tiny IN_SCOPE.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH_DEFAULT, help="Pfad zum Parquet-Datensatz")
    parser.add_argument("--out", type=Path, default=OUT_CSV_DEFAULT, help="Output CSV (reports/tables)")
    parser.add_argument("--k", type=int, default=3, help="Anzahl Cluster")
    parser.add_argument("--min-systems", type=int, default=10, help="Minimale Systemzahl, sonst Abbruch")
    parser.add_argument("--seed", type=int, default=42, help="Seed für Reproduzierbarkeit")
    args = parser.parse_args(argv)

    run_clustering(
        parquet_path=args.parquet,
        out_csv=args.out,
        n_clusters=args.k,
        min_systems=args.min_systems,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
