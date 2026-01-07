# src/checks.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.features import add_features, get_analysis_subsets, round_task_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "figures"

BASELINE_DIR_DEFAULT = PROJECT_ROOT / "docs" / "baseline"
BASELINE_TABLES_MANIFEST = BASELINE_DIR_DEFAULT / "tables_manifest.json"
BASELINE_FIGURES_MANIFEST = BASELINE_DIR_DEFAULT / "figures_manifest.json"
BASELINE_META = BASELINE_DIR_DEFAULT / "meta.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_for_dir(dir_path: Path, patterns: Tuple[str, ...]) -> Dict[str, str]:
    if not dir_path.exists():
        return {}
    files = []
    for pat in patterns:
        files.extend(sorted(dir_path.glob(pat)))
    out: Dict[str, str] = {}
    for p in sorted(set(files)):
        if p.is_file():
            out[p.name] = _sha256(p)
    return out


def set_baseline(tables_dir: Path, figures_dir: Path, baseline_dir: Path) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)

    tables_manifest = _manifest_for_dir(tables_dir, ("*.csv",))
    figures_manifest = _manifest_for_dir(figures_dir, ("*.png", "*.svg"))

    BASELINE_TABLES_MANIFEST.write_text(json.dumps(tables_manifest, indent=2), encoding="utf-8")
    BASELINE_FIGURES_MANIFEST.write_text(json.dumps(figures_manifest, indent=2), encoding="utf-8")

    meta = {
        "note": "Baseline manifest for reproducibility checks (hash-based).",
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "num_tables": len(tables_manifest),
        "num_figures": len(figures_manifest),
    }
    BASELINE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Baseline gesetzt: {baseline_dir}")
    print(f"- Tabellen: {len(tables_manifest)}")
    print(f"- Figures:  {len(figures_manifest)}")


def diff_against_baseline(current: Dict[str, str], baseline: Dict[str, str]) -> Dict[str, list[str]]:
    same, changed = [], []
    for k, v in current.items():
        if k in baseline and baseline[k] == v:
            same.append(k)
        elif k in baseline and baseline[k] != v:
            changed.append(k)
    missing_baseline = [k for k in current.keys() if k not in baseline]
    missing_current = [k for k in baseline.keys() if k not in current]
    return {
        "same": sorted(same),
        "changed": sorted(changed),
        "missing_baseline": sorted(missing_baseline),
        "missing_current": sorted(missing_current),
    }


def run_checks(parquet_path: Path, tables_dir: Path, figures_dir: Path, baseline_dir: Path) -> None:
    print("=" * 55)
    print("A) Scope-Checks (OUT_OF_SCOPE / IN_SCOPE / UNKNOWN)")
    print("=" * 55)

    df_raw = pd.read_parquet(parquet_path)
    df = add_features(df_raw)
    subsets = get_analysis_subsets(df)

    total = len(df)
    in_scope = len(subsets["DS_IN_SCOPE"])
    out_of_scope = len(df[df["scope_status"].eq("OUT_OF_SCOPE")])
    unknown = len(df[df["scope_status"].eq("UNKNOWN")])

    print(f"Rows total: {total}")
    print(f"Rows in_scope: {in_scope} ({in_scope/total:.3f})")
    print(f"Rows out_of_scope: {out_of_scope} ({out_of_scope/total:.3f})")
    print(f"Rows UNKNOWN: {unknown} ({unknown/total:.3f})")

    print("\n" + "=" * 55)
    print("B) Coverage-Checks (Round × Task) – ohne OUT_OF_SCOPE")
    print("=" * 55)
    cov = round_task_counts(subsets["DS_NON_OOS"], include_unknown=True)
    piv = cov.pivot_table(index="round", columns="task", values="rows", aggfunc="sum").fillna(0).astype(int)
    print(piv.to_string())

    print("\n" + "=" * 55)
    print("C) Missingness & Plausibility (IN_SCOPE)")
    print("=" * 55)
    d = subsets["DS_IN_SCOPE"].copy()
    metrics = [c for c in ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"] if c in d.columns]
    if metrics:
        miss = d[metrics].isna().mean().sort_values(ascending=False)
        print("Missingness (Anteil NA):")
        print(miss.to_string())
        if "latency_us" in d.columns and d["latency_us"].notna().any():
            print(f"\nlatency_us: min={d['latency_us'].min():.2f}, median={d['latency_us'].median():.2f}, max={d['latency_us'].max():.2f}")
        if "energy_uj" in d.columns and d["energy_uj"].notna().any():
            print(f"energy_uj: min={d['energy_uj'].min():.2f}, median={d['energy_uj'].median():.2f}, max={d['energy_uj'].max():.2f}")
    else:
        print("Keine Metrikspalten vorhanden.")

    print("\n" + "=" * 55)
    print("D) Duplikat-Checks")
    print("=" * 55)
    # key wie bisher: soweit verfügbar
    key_candidates = ["public_id", "round", "system_name", "host_processor_frequency", "model_mlc"]
    key = [c for c in key_candidates if c in df_raw.columns]
    if key:
        dups = df_raw.duplicated(key).sum()
        print(f"Duplicate auf KEY {key}: {int(dups)}")
    else:
        print("Kein passender Key gefunden (Spalten fehlen).")

    print("\n" + "=" * 55)
    print("E) Reports/Tables Diff (gegen Baseline)")
    print("=" * 55)
    current_tables = _manifest_for_dir(tables_dir, ("*.csv",))
    baseline_tables = json.loads(BASELINE_TABLES_MANIFEST.read_text(encoding="utf-8")) if BASELINE_TABLES_MANIFEST.exists() else {}
    diff_t = diff_against_baseline(current_tables, baseline_tables)
    print(f"same: {len(diff_t['same'])} | changed: {len(diff_t['changed'])} | missing_baseline: {len(diff_t['missing_baseline'])} | missing_current: {len(diff_t['missing_current'])}")
    if diff_t["changed"]:
        print("CHANGED:", ", ".join(diff_t["changed"]))
    if diff_t["missing_baseline"]:
        print("MISSING in Baseline (neu):", ", ".join(diff_t["missing_baseline"]))
    if diff_t["missing_current"]:
        print("MISSING in Current (weg):", ", ".join(diff_t["missing_current"]))

    print("\n" + "=" * 55)
    print("F) Reports/Figures Präsenzcheck (PNG/SVG)")
    print("=" * 55)
    figs = _manifest_for_dir(figures_dir, ("*.png", "*.svg"))
    n_png = len([k for k in figs.keys() if k.lower().endswith(".png")])
    n_svg = len([k for k in figs.keys() if k.lower().endswith(".svg")])
    print(f"PNG: {n_png} | SVG: {n_svg}")
    top_png = [k for k in sorted(figs.keys()) if k.lower().endswith(".png")][:10]
    if top_png:
        print("Top PNGs:")
        for k in top_png:
            print(f"- {k}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scope-aware checks + baseline diff.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR_DEFAULT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR_DEFAULT)
    parser.add_argument("--set-baseline", action="store_true", help="Write baseline manifests from current reports/")
    args = parser.parse_args(argv)

    if args.set_baseline:
        set_baseline(args.tables_dir, args.figures_dir, args.baseline_dir)
        return

    run_checks(args.parquet, args.tables_dir, args.figures_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
