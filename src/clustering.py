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

# System-Identität: nicht nur system_name verwenden (kann über Rounds/Re-Submissions kollidieren)
SYSTEM_KEY = ["round", "public_id", "organization", "system_name"]


def _log_header(title: str) -> None:
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
    _log_header("Clustering: Hardware-Klassen (Latency + Energy, IN_SCOPE)")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {parquet_path}")

    print(f"Lade Daten von: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Features anreichern (scope/task_canon/model_mlc_canon etc.)
    df = add_features(df)

    # Nur IN_SCOPE + Zeilen mit Latenz UND Energie
    mask = (df["scope"] == "IN_SCOPE") & df[REQUIRED_COLS].notna().all(axis=1)
    d = df.loc[mask].copy()

    print(f"Rows total: {len(df)}")
    print(f"Rows IN_SCOPE + (Latency & Energy): {len(d)}")

    if d.empty:
        print("WARNUNG: Keine Datenpunkte mit Latency & Energy im IN_SCOPE gefunden.")
        return

    # Transparenz: Coverage pro Round
    cov = d.groupby("round").size().rename("rows").reset_index()
    print("\nCoverage (rows mit Latency & Energy) pro Round:")
    print(cov.to_string(index=False))

    # Aggregation auf System-Ebene:
    # Ein System hat mehrere Tasks/Einträge -> Median als robustes Aggregat
    _log_header("Aggregation: System-Level Features (Median)")
    sys_features = (
        d.groupby(SYSTEM_KEY, dropna=False)[REQUIRED_COLS]
        .median()
        .reset_index()
    )

    # Optional hilfreiche Meta-Infos (wie viele Messpunkte pro System)
    sys_n = d.groupby(SYSTEM_KEY, dropna=False).size().rename("n_rows").reset_index()
    sys_features = sys_features.merge(sys_n, on=SYSTEM_KEY, how="left")

    n_systems = len(sys_features)
    print(f"Anzahl Systeme (System-Key={SYSTEM_KEY}): {n_systems}")

    if n_systems < max(min_systems, n_clusters * 2):
        print(
            "WARNUNG: Zu wenige Systeme für robustes Clustering "
            f"(systeme={n_systems}, n_clusters={n_clusters})."
        )
        return

    # Feature Engineering: Log-Transformation
    # Latenz/Energie streuen typischerweise über Größenordnungen
    X = np.log1p(sys_features[REQUIRED_COLS].astype(float))

    # Skalierung (Standardisierung)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_id = kmeans.fit_predict(X_scaled)
    sys_features["cluster_id"] = cluster_id

    # Cluster logisch benennen:
    # performance_class 0 = schnellste (kleinste median latency), N = langsamste
    cluster_stats = (
        sys_features.groupby("cluster_id")["latency_us"]
        .median()
        .sort_values()
    )
    mapping = {old_id: new_class for new_class, old_id in enumerate(cluster_stats.index)}
    sys_features["performance_class"] = sys_features["cluster_id"].map(mapping)

    # Aufräumen & sortieren
    sys_features = (
        sys_features.drop(columns=["cluster_id"])
        .sort_values(["performance_class", "round", "organization", "system_name"])
        .reset_index(drop=True)
    )

    # Speichern
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sys_features.to_csv(out_csv, index=False, encoding="utf-8-sig")

    _log_header("Ergebnis")
    print(f"Clustering abgeschlossen: {n_clusters} Klassen")
    print(f"Gespeichert: {out_csv}")

    print("\nCluster-Statistiken (Median pro performance_class):")
    stats = sys_features.groupby("performance_class")[REQUIRED_COLS].median()
    print(stats.to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KMeans-Clustering (Hardware-Klassen) auf MLPerf-Tiny IN_SCOPE.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH_DEFAULT, help="Path zum Parquet-Datensatz")
    parser.add_argument("--out", type=Path, default=OUT_CSV_DEFAULT, help="Output CSV in reports/tables/")
    parser.add_argument("--k", type=int, default=3, help="Anzahl Cluster")
    parser.add_argument("--min-systems", type=int, default=10, help="Mindestanzahl Systeme, sonst Abbruch mit Warnung")
    parser.add_argument("--seed", type=int, default=42, help="Random seed für Reproduzierbarkeit")

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
