# src/checks.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.features import add_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BASELINE_DIR = PROJECT_ROOT / "docs" / "baseline"
BASELINE_TABLES_DIR = PROJECT_ROOT / "docs" / "baseline_tables"

# Baseline meta (je nach früherer Version kann es baseline_meta.json oder meta.json geben)
BASELINE_META_CANDIDATES = [
    BASELINE_DIR / "baseline_meta.json",
    BASELINE_DIR / "meta.json",
]


# -----------------------------
# Helpers: printing
# -----------------------------
def _hline(ch: str = "=", n: int = 70) -> str:
    return ch * n


def _section(title: str) -> None:
    print()
    print(_hline("="))
    print(title)
    print(_hline("="))


def _subsection(title: str) -> None:
    print()
    print(_hline("-", 70))
    print(title)
    print(_hline("-", 70))


# -----------------------------
# Helpers: filesystem / hashing
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(dir_path: Path, *, exts: Optional[Iterable[str]] = None) -> List[Path]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.rglob("*") if p.is_file()]
    if exts:
        exts_set = {e.lower() for e in exts}
        files = [p for p in files if p.suffix.lower() in exts_set]
    return sorted(files)


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Robust gegen 0-byte/leer gespeicherte CSVs.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_df_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisiert für stabile Hashes (Sortierung, Index reset).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # Sortiere Spalten alphabetisch für Stabilität
    out = out.reindex(sorted(out.columns), axis=1)

    # Sortiere Zeilen deterministisch, wenn möglich (alle Spalten)
    try:
        out = out.sort_values(by=list(out.columns), kind="mergesort")
    except Exception:
        # falls nicht sortierbar (mixed types), fallback: original order
        pass

    return out.reset_index(drop=True)


def sha256_df_csv_normalized(df: pd.DataFrame) -> str:
    """
    Hash auf Basis einer normalisierten CSV-Repräsentation (ohne Index),
    um kleine Reihenfolge-Unterschiede abzufangen.
    """
    out = normalize_df_for_hash(df)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# -----------------------------
# Baseline manifests
# -----------------------------
def expected_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def build_manifest_for_dir(
    dir_path: Path,
    *,
    exts: Iterable[str],
) -> Dict[str, str]:
    """
    returns {filename: sha256(file)}
    """
    manifest: Dict[str, str] = {}
    for p in list_files(dir_path, exts=exts):
        manifest[p.name] = sha256_file(p)
    return manifest


def load_manifest(path: Path) -> Optional[Dict[str, str]]:
    data = load_json_if_exists(path)
    if isinstance(data, dict):
        # allow stored under {"files": {...}}
        if "files" in data and isinstance(data["files"], dict):
            return {k: str(v) for k, v in data["files"].items()}
        return {k: str(v) for k, v in data.items()}
    return None


def baseline_meta() -> Optional[dict]:
    for p in BASELINE_META_CANDIDATES:
        data = load_json_if_exists(p)
        if data is not None:
            return data
    return None


@dataclass
class DiffResult:
    same: List[str]
    changed: Dict[str, Tuple[str, str]]  # file -> (baseline_hash, current_hash)
    missing_baseline: List[str]
    missing_current: List[str]


def diff_manifests(baseline: Dict[str, str], current: Dict[str, str]) -> DiffResult:
    base_files = set(baseline.keys())
    cur_files = set(current.keys())

    missing_baseline = sorted(cur_files - base_files)
    missing_current = sorted(base_files - cur_files)

    same: List[str] = []
    changed: Dict[str, Tuple[str, str]] = {}

    for fn in sorted(base_files & cur_files):
        b = baseline[fn]
        c = current[fn]
        if b == c:
            same.append(fn)
        else:
            changed[fn] = (b, c)

    return DiffResult(
        same=same,
        changed=changed,
        missing_baseline=missing_baseline,
        missing_current=missing_current,
    )


# -----------------------------
# Checks: dataset-level
# -----------------------------
def load_dataset(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # ensure canon/scope columns
    df = add_features(df)
    return df


def scope_checks(df: pd.DataFrame) -> None:
    _section("A) Scope-Checks (OUT_OF_SCOPE / IN_SCOPE / UNKNOWN)")

    total = len(df)
    in_scope = int((df["scope_status"] == "IN_SCOPE").sum())
    out_scope = int((df["scope_status"] == "OUT_OF_SCOPE").sum())
    unknown = int((df["scope_status"] == "UNKNOWN").sum())

    def pct(x: int) -> float:
        return (x / total) if total else 0.0

    print(f"Rows total: {total}")
    print(f"Rows in_scope: {in_scope} ({pct(in_scope):.3f})")
    print(f"Rows out_of_scope: {out_scope} ({pct(out_scope):.3f})")
    print(f"Rows UNKNOWN (ohne out_of_scope): {unknown} ({pct(unknown):.3f})")

    # Erwartung: OUT_OF_SCOPE nur durch 1D_DS_CNN (v1.3)
    out_rows = df.loc[df["scope_status"] == "OUT_OF_SCOPE", ["round", "task", "task_canon", "model_mlc", "model_mlc_canon"]]
    if out_scope == 0:
        print("HINWEIS: Keine OUT_OF_SCOPE Zeilen vorhanden.")
        return

    # einfache Plausibilitäts-Aussage (nicht hart failen)
    only_v13 = out_rows["round"].astype(str).eq("v1.3").all()
    only_1d = out_rows["model_mlc_canon"].astype(str).eq("1D_DS_CNN").all()

    if only_v13 and only_1d:
        print("OK: OUT_OF_SCOPE entspricht der Erwartung (nur 1D_DS_CNN in v1.3).")
    else:
        print("WARN: OUT_OF_SCOPE weicht von der Erwartung ab.")
        print(out_rows.value_counts().head(20).to_string())


def coverage_checks(df: pd.DataFrame) -> None:
    _section("B) Coverage-Checks (Round × Task)")

    # OUT_OF_SCOPE nicht in Coverage aufnehmen
    d = df[df["scope_status"] != "OUT_OF_SCOPE"].copy()
    d["_task_for_cov"] = d["task_canon"].astype(str)

    cov = (
        d.groupby(["round", "_task_for_cov"])
        .size()
        .reset_index(name="rows")
        .sort_values(["round", "_task_for_cov"])
    )

    # 1) Pivot (komfortabel)
    pivot = cov.pivot_table(index="round", columns="_task_for_cov", values="rows", fill_value=0, aggfunc="sum")
    # sort rounds in a simple numeric way if they follow v0.5 pattern
    pivot = pivot.reset_index()
    try:
        pivot["_r"] = pivot["round"].astype(str).str.replace("v", "", regex=False).astype(float)
        pivot = pivot.sort_values("_r").drop(columns="_r")
    except Exception:
        pass
    pivot = pivot.set_index("round")

    print(pivot.to_string())

    # 2) Long head (falls du es aus Logs kopieren willst)
    print()
    print(cov.head(30).to_string(index=False))


def duplicate_checks(df: pd.DataFrame) -> None:
    _section("D) Duplikat-Checks")

    key = ["public_id", "round", "system_name", "host_processor_frequency", "model_mlc"]
    missing = [k for k in key if k not in df.columns]
    if missing:
        print(f"HINWEIS: Duplikat-KEY nicht vollständig im DataFrame. Missing: {missing}")
        return

    dups = int(df.duplicated(key).sum())
    print(f"Duplicate auf KEY {key}: {dups}")


def missingness_plausibility_checks(df: pd.DataFrame) -> None:
    _section("C) Missingness & Plausibility (in_scope)")

    d = df[df["scope_status"] == "IN_SCOPE"].copy()

    metrics = [c for c in ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"] if c in d.columns]
    if not metrics:
        print("HINWEIS: Keine Metrikspalten gefunden.")
        return

    miss = d[metrics].isna().mean().sort_values(ascending=False)
    print("Missingness (Anteil NA):")
    print(miss.to_string())

    print()
    print("Nicht-NA Counts:")
    print(d[metrics].notna().sum().to_string())

    # einfache Plausibility-Spans für latency/energy
    if "latency_us" in d.columns and d["latency_us"].notna().any():
        s = d["latency_us"].dropna().astype(float)
        print()
        print(f"latency_us: min={s.min():.2f}, median={s.median():.3g}, max={s.max():.3g}")

    if "energy_uj" in d.columns and d["energy_uj"].notna().any():
        s = d["energy_uj"].dropna().astype(float)
        print(f"energy_uj: min={s.min():.3g}, median={s.median():.3g}, max={s.max():.3g}")


def clustering_sanity_check() -> None:
    _section("F) Clustering-Sanity-Check (hardware_clusters.csv)")

    path = REPORTS_TABLES_DIR / "hardware_clusters.csv"
    if not path.exists():
        print("HINWEIS: reports/tables/hardware_clusters.csv nicht gefunden – Clustering wurde ggf. noch nicht ausgeführt.")
        return

    df = safe_read_csv(path)
    if df.empty:
        print("WARN: hardware_clusters.csv ist leer.")
        return

    required = {"round", "performance_class"}
    missing = required - set(df.columns)
    if missing:
        print(f"WARN: Spalten fehlen in hardware_clusters.csv: {sorted(missing)}")
        return

    pivot = (
        df.groupby(["round", "performance_class"])
        .size()
        .unstack(fill_value=0)
    )

    # round sorting
    pivot = pivot.reset_index()
    try:
        pivot["_r"] = pivot["round"].astype(str).str.replace("v", "", regex=False).astype(float)
        pivot = pivot.sort_values("_r").drop(columns="_r")
    except Exception:
        pass
    pivot = pivot.set_index("round")

    print(pivot.to_string())

    # duplicate check on key used in clustering aggregation
    key = [c for c in ["round", "public_id", "organization", "system_name"] if c in df.columns]
    if key:
        dups = int(df.duplicated(key).sum())
        print()
        print(f"dups: {dups}")
    else:
        print()
        print("HINWEIS: Kein Key für Duplikatprüfung in hardware_clusters.csv gefunden.")


# -----------------------------
# Checks: baseline comparison
# -----------------------------
def baseline_evidence() -> None:
    meta = baseline_meta()
    print("Baseline-Nachweis:")
    print(f"- baseline_dir: {BASELINE_DIR}")
    print(f"- baseline_tables_dir: {BASELINE_TABLES_DIR}")
    if meta:
        created = meta.get("created_at") or meta.get("created") or meta.get("timestamp")
        if created:
            print(f"- created_at: {created}")
        # optional: store dirs in meta
        td = meta.get("tables_dir")
        fd = meta.get("figures_dir")
        if td:
            print(f"- tables_dir: {td}")
        if fd:
            print(f"- figures_dir: {fd}")
    else:
        print("- created_at: (nicht gefunden – meta.json/baseline_meta.json fehlt oder nicht lesbar)")


def diff_reports_tables_against_baseline() -> None:
    _section("E) Reports/Tables Diff (gegen Baseline)")

    baseline_evidence()

    baseline_manifest_path = BASELINE_DIR / "tables_manifest.json"
    current_manifest_path = REPORTS_TABLES_DIR / "tables_manifest.json"  # optional (falls du mal erzeugst)

    baseline_manifest = load_manifest(baseline_manifest_path)
    if baseline_manifest is None:
        # fallback: build from baseline_tables_dir
        baseline_manifest = build_manifest_for_dir(BASELINE_TABLES_DIR, exts=[".csv"])

    # current manifest: always build from reports/tables (damit es den echten Stand prüft)
    current_manifest = build_manifest_for_dir(REPORTS_TABLES_DIR, exts=[".csv"])

    diff = diff_manifests(baseline_manifest, current_manifest)

    print()
    print(f"same: {len(diff.same)} | changed: {len(diff.changed)} | missing_baseline: {len(diff.missing_baseline)} | missing_current: {len(diff.missing_current)}")

    if diff.changed:
        print()
        print("CHANGED (Hash differs):")
        for fn, (b, c) in diff.changed.items():
            print(f"- {fn} (baseline={b[:12]}..., current={c[:12]}...)")

    if diff.missing_baseline:
        print()
        print("MISSING in Baseline (neu):")
        for fn in diff.missing_baseline:
            print(f"- {fn}")

    if diff.missing_current:
        print()
        print("MISSING in Current (fehlt aktuell):")
        for fn in diff.missing_current:
            print(f"- {fn}")


def figures_presence_check() -> None:
    _section("G) Reports/Figures Präsenzcheck")

    pngs = list_files(REPORTS_FIGURES_DIR, exts=[".png"])
    svgs = list_files(REPORTS_FIGURES_DIR, exts=[".svg"])

    print(f"PNG: {len(pngs)} | SVG: {len(svgs)}")

    top_pngs = [p.name for p in pngs[:10]]
    if top_pngs:
        print()
        print("Top PNGs:")
        for n in top_pngs:
            print(f"- {n}")

    # optional expected lists (aus Baseline)
    expected_figs = expected_list(BASELINE_DIR / "expected_figures.txt")
    if expected_figs:
        cur = {p.name for p in pngs} | {p.name for p in svgs}
        exp = set(expected_figs)

        missing = sorted(exp - cur)
        extra = sorted(cur - exp)

        if missing:
            print()
            print("MISSING (expected, aber nicht vorhanden):")
            for n in missing:
                print(f"- {n}")

        if extra:
            print()
            print("EXTRA (vorhanden, aber nicht in expected_figures):")
            for n in extra[:30]:
                print(f"- {n}")
            if len(extra) > 30:
                print(f"... (+{len(extra)-30} weitere)")


# -----------------------------
# Main
# -----------------------------
def run_all(parquet_path: Path) -> None:
    df = load_dataset(parquet_path)

    scope_checks(df)
    coverage_checks(df)
    duplicate_checks(df)
    missingness_plausibility_checks(df)

    clustering_sanity_check()
    diff_reports_tables_against_baseline()
    figures_presence_check()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Project checks: scope, coverage, plausibility, baseline diffs.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT, help="Path to interim parquet dataset")
    args = parser.parse_args(argv)

    run_all(args.parquet)


if __name__ == "__main__":
    main()
