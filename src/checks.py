from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import add_features, get_analysis_subsets, round_task_counts

# ---------------------------------------------------------------------
# Paths / Defaults
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(".")
PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

BASELINE_DIR_DEFAULT = PROJECT_ROOT / "docs" / "baseline"
BASELINE_TABLES_DIR_DEFAULT = PROJECT_ROOT / "docs" / "baseline_tables"
BASELINE_META = BASELINE_DIR_DEFAULT / "meta.json"
BASELINE_TABLES_MANIFEST = BASELINE_DIR_DEFAULT / "tables_manifest.json"
BASELINE_FIGURES_MANIFEST = BASELINE_DIR_DEFAULT / "figures_manifest.json"


def _baseline_paths(baseline_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Returns (meta.json, tables_manifest.json, figures_manifest.json, baseline_tables_dir)."""
    baseline_dir = Path(baseline_dir)
    meta_p = baseline_dir / "meta.json"
    tab_p = baseline_dir / "tables_manifest.json"
    fig_p = baseline_dir / "figures_manifest.json"
    # keep the convention: docs/baseline + docs/baseline_tables
    baseline_tables_dir = baseline_dir.parent / "baseline_tables"
    return meta_p, tab_p, fig_p, baseline_tables_dir


def _normalize_scope_status(values: pd.Series) -> pd.Series:
    """Normalisiere scope_status auf {IN_SCOPE, OUT_OF_SCOPE, UNKNOWN}.

    Unterstützt Gross-/Kleinschreibung und gängig
    variierende Schreibweisen.
    """
    s = values.astype(str).str.strip().str.upper()
    s = s.replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
    s = s.replace({"IN SCOPE": "IN_SCOPE", "OUT OF SCOPE": "OUT_OF_SCOPE"})
    s = s.where(s.isin(["IN_SCOPE", "OUT_OF_SCOPE", "UNKNOWN"]), "UNKNOWN")
    return s


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    return files


# ---------------------------------------------------------------------
# Baseline freeze (manifests + meta)
# ---------------------------------------------------------------------


def freeze_baseline(*, tables_dir: Path, figures_dir: Path, baseline_dir: Path) -> None:
    """Schreibt Baseline-Manifeste + Meta basierend auf aktuellen reports/."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)

    table_files = _collect_files(tables_dir, (".csv",))
    figure_files = _collect_files(figures_dir, (".png", ".svg"))

    tables_manifest = {p.name: _sha256_of_file(p) for p in table_files}
    figures_manifest = {p.name: _sha256_of_file(p) for p in figure_files}

    meta = {
        "tables_dir": str(tables_dir.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "created_at": pd.Timestamp.now(tz="Europe/Berlin").isoformat(),
        "n_tables": len(table_files),
        "n_figures": len(figure_files),
    }

    meta_p, tab_p, fig_p, baseline_tables_dir = _baseline_paths(baseline_dir)
    baseline_tables_dir.mkdir(parents=True, exist_ok=True)

    tab_p.write_text(json.dumps(tables_manifest, indent=2), encoding="utf-8")
    fig_p.write_text(json.dumps(figures_manifest, indent=2), encoding="utf-8")
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Optional: baseline_tables Snapshot kopieren (nur CSVs)
    for p in table_files:
        target = baseline_tables_dir / p.name
        target.write_bytes(p.read_bytes())

    print("Baseline eingefroren:")
    print(f"- meta: {meta_p}")
    print(f"- tables_manifest: {tab_p}")
    print(f"- figures_manifest: {fig_p}")
    print(f"- baseline_tables/: {baseline_tables_dir}")


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------


@dataclass
class ScopeStats:
    rows_total: int
    rows_in_scope: int
    rows_out_of_scope: int
    rows_unknown: int


def _load_and_features(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # defensive: features may already exist or not
    try:
        df = add_features(df)
    except Exception as e:
        raise RuntimeError(f"add_features fehlgeschlagen: {e}") from e

    return df


def _scope_stats(df: pd.DataFrame) -> ScopeStats:
    if "scope_status" not in df.columns:
        raise KeyError("Spalte 'scope_status' fehlt (add_features muss diese setzen).")

    scope = _normalize_scope_status(df["scope_status"])
    rows_total = len(df)
    rows_in_scope = int((scope == "IN_SCOPE").sum())
    rows_out_of_scope = int((scope == "OUT_OF_SCOPE").sum())
    rows_unknown = int((scope == "UNKNOWN").sum())
    return ScopeStats(rows_total, rows_in_scope, rows_out_of_scope, rows_unknown)


def _scope_unknown_causes(df: pd.DataFrame) -> pd.Series:
    # Ursache: missing source fields (example: model_mlc_source)
    if "model_mlc_source" not in df.columns:
        return pd.Series(dtype=int)
    s = df.loc[_normalize_scope_status(df["scope_status"]) == "UNKNOWN", "model_mlc_source"]
    s = s.fillna("missing").astype(str).str.strip()
    return s.value_counts()


def _coverage_round_task(df: pd.DataFrame) -> pd.DataFrame:
    # Uses canonical task column if present, otherwise task
    if "round" not in df.columns:
        raise KeyError("Spalte 'round' fehlt.")
    task_col = "task_canon" if "task_canon" in df.columns else ("task" if "task" in df.columns else None)
    if task_col is None:
        raise KeyError("Weder 'task_canon' noch 'task' in df vorhanden.")

    tmp = df.copy()
    tmp[task_col] = tmp[task_col].fillna("UNKNOWN")
    tmp["scope_status"] = _normalize_scope_status(tmp.get("scope_status", pd.Series(["UNKNOWN"] * len(tmp))))

    # For coverage table, treat OUT_OF_SCOPE separately
    tmp.loc[tmp["scope_status"] == "OUT_OF_SCOPE", task_col] = "OUT_OF_SCOPE"
    tmp.loc[tmp["scope_status"] == "UNKNOWN", task_col] = "UNKNOWN"

    ct = pd.crosstab(tmp["round"], tmp[task_col]).reset_index()
    return ct


def _metric_coverage(df: pd.DataFrame, metric: str) -> Tuple[int, int, float]:
    # Coverage on IN_SCOPE only (as in your console output)
    scope = _normalize_scope_status(df["scope_status"])
    dfi = df.loc[scope == "IN_SCOPE"].copy()

    if metric not in dfi.columns:
        return (0, len(dfi), 0.0)

    m = pd.to_numeric(dfi[metric], errors="coerce")
    rows_with = int(m.notna().sum())
    rows_total = len(dfi)
    share = float(rows_with / rows_total) if rows_total else 0.0
    return rows_with, rows_total, share


def _missingness_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    scope = _normalize_scope_status(df["scope_status"])
    dfi = df.loc[scope == "IN_SCOPE"].copy()

    rows = []
    for m in metrics:
        if m not in dfi.columns:
            rows.append({"metric": m, "share_na": 1.0})
            continue
        s = pd.to_numeric(dfi[m], errors="coerce")
        rows.append({"metric": m, "share_na": float(s.isna().mean())})
    return pd.DataFrame(rows).sort_values("share_na", ascending=False)


def _plausibility_summary(df: pd.DataFrame, metric: str) -> Optional[str]:
    scope = _normalize_scope_status(df["scope_status"])
    dfi = df.loc[scope == "IN_SCOPE"].copy()
    if metric not in dfi.columns:
        return None
    s = pd.to_numeric(dfi[metric], errors="coerce").dropna()
    if s.empty:
        return None
    q = s.quantile([0.0, 0.5, 1.0])
    return f"{metric}: min={q.loc[0.0]:g}, median={q.loc[0.5]:.3g}, max={q.loc[1.0]:g}"


def _unit_safety_latency_us(df: pd.DataFrame) -> str:
    scope = _normalize_scope_status(df["scope_status"])
    dfi = df.loc[scope == "IN_SCOPE"].copy()

    if "latency_us" not in dfi.columns:
        return "WARN: latency_us fehlt."

    s = pd.to_numeric(dfi["latency_us"], errors="coerce").dropna()
    if s.empty:
        return "WARN: latency_us leer."

    p50 = float(s.quantile(0.50))
    p95 = float(s.quantile(0.95))
    p99 = float(s.quantile(0.99))

    # Heuristik: latency_us sollte typischerweise im Bereich 1e2..1e7 liegen
    if not (1e2 <= p50 <= 1e7 and 1e2 <= p95 <= 1e7 and 1e2 <= p99 <= 1e7):
        return f"WARN: latency_us Skala auffällig (p50={p50:.3g}, p95={p95:.3g}, p99={p99:.3g})."
    return f"OK: latency_us Skala plausibel (p50={p50:.3g}, p95={p95:.3g}, p99={p99:.3g})."


def _duplicates_check(df: pd.DataFrame) -> int:
    # minimal, robust key: only check if columns exist
    key = ["public_id", "round", "system_name", "host_processor_frequency", "model_mlc"]
    key = [c for c in key if c in df.columns]
    if not key:
        return 0
    return int(df.duplicated(subset=key).sum())


def run_checks(
    parquet_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    baseline_dir: Path,
    *,
    strict_baseline_diff: bool = False,
) -> None:
    print("\nchecks: Validierung & Baseline-Abgleich")
    print("=" * 60)

    df = _load_and_features(parquet_path)

    # -----------------------------------------------------------------
    # A) Scope checks
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("A) Scope-Checks (OUT_OF_SCOPE / IN_SCOPE / UNKNOWN)")
    print("=" * 55)

    stats = _scope_stats(df)
    print(f"Rows total: {stats.rows_total}")
    print(f"Rows in_scope: {stats.rows_in_scope} ({stats.rows_in_scope / max(stats.rows_total,1):.3f})")
    print(f"Rows out_of_scope: {stats.rows_out_of_scope} ({stats.rows_out_of_scope / max(stats.rows_total,1):.3f})")
    # UNKNOWN ohne OUT_OF_SCOPE
    print(f"Rows UNKNOWN (ohne out_of_scope): {stats.rows_unknown} ({stats.rows_unknown / max(stats.rows_total,1):.3f})")

    causes = _scope_unknown_causes(df)
    if not causes.empty:
        print("\nUNKNOWN Ursachen (model_mlc_source):")
        for k, v in causes.items():
            print(f"  - {k}: {v}")

    # -----------------------------------------------------------------
    # B) Coverage checks
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("B) Coverage-Checks (Round × Task)")
    print("=" * 55)

    cov = _coverage_round_task(df)
    print(cov.to_string(index=False))

    # Export coverage table
    tables_dir.mkdir(parents=True, exist_ok=True)
    cov_path = tables_dir / "coverage_round_task_all.csv"
    cov.to_csv(cov_path, index=False)
    print(f"[table] coverage_round_task_all: {len(cov)} rows -> {cov_path}")

    # -----------------------------------------------------------------
    # B3) Metric coverage (minimal)
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("B3) Metric Coverage (auf IN_SCOPE + latency Basis)")
    print("=" * 55)

    for metric in ["energy_uj", "power_mw", "accuracy", "auc"]:
        rows_with, rows_total, share = _metric_coverage(df, metric)
        print(f"Metric: {metric} | rows_with_metric={rows_with}/{rows_total} | share={share:.3f}")

    # -----------------------------------------------------------------
    # D) Duplicate check
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("D) Duplikat-Checks")
    print("=" * 55)

    dup_n = _duplicates_check(df)
    print(f"Duplicate count: {dup_n}")

    # -----------------------------------------------------------------
    # C) Missingness & plausibility
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("C) Missingness & Plausibility (in_scope)")
    print("=" * 55)

    miss = _missingness_table(df, metrics=["power_mw", "auc", "accuracy", "energy_uj", "latency_us"])
    print("Missingness (Anteil NA):")
    print(miss.to_string(index=False))

    # Not-NA counts
    scope = _normalize_scope_status(df["scope_status"])
    dfi = df.loc[scope == "IN_SCOPE"].copy()
    if "latency_us" in dfi.columns:
        print(f"latency_us Not-NA: {pd.to_numeric(dfi['latency_us'], errors='coerce').notna().sum()}")
    if "energy_uj" in dfi.columns:
        print(f"energy_uj Not-NA: {pd.to_numeric(dfi['energy_uj'], errors='coerce').notna().sum()}")

    for metric in ["latency_us", "energy_uj"]:
        summ = _plausibility_summary(df, metric)
        if summ:
            print("\n" + summ)

    # -----------------------------------------------------------------
    # C2) Unit-safety (ms->us once)
    # -----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("C2) Unit-Safety Heuristik (ms→µs genau einmal)")
    print("=" * 55)
    print(_unit_safety_latency_us(df))

    # -----------------------------------------------------------------
    # E/G) Release-Checks (nur bei --strict-baseline-diff)
    # -----------------------------------------------------------------
    if strict_baseline_diff:
        meta_p, tab_p, fig_p, _baseline_tables_dir = _baseline_paths(baseline_dir)

        # -----------------------------------------------------------------
        # E) Baseline diff
        # -----------------------------------------------------------------
        print("\n" + "=" * 55)
        print("E) Reports/Tables Diff (gegen Baseline)")
        print("=" * 55)

        if not (meta_p.exists() and tab_p.exists() and fig_p.exists()):
            print(
                "FAIL: Baseline nicht vollständig vorhanden (meta.json / tables_manifest.json / figures_manifest.json fehlt).\n"
                "-> Baseline einfrieren: python -m src.checks --freeze-baseline --baseline-dir <DIR>"
            )
            raise SystemExit(2)

        baseline_tables = json.loads(tab_p.read_text(encoding="utf-8"))
        baseline_figures = json.loads(fig_p.read_text(encoding="utf-8"))

        cur_tables = {p.name: _sha256_of_file(p) for p in _collect_files(tables_dir, (".csv",))}
        cur_figures = {p.name: _sha256_of_file(p) for p in _collect_files(figures_dir, (".png", ".svg"))}

        # Tables diff
        same = [k for k in cur_tables.keys() if k in baseline_tables and cur_tables[k] == baseline_tables[k]]
        changed = [k for k in cur_tables.keys() if k in baseline_tables and cur_tables[k] != baseline_tables[k]]
        missing_baseline = [k for k in cur_tables.keys() if k not in baseline_tables]
        missing_current = [k for k in baseline_tables.keys() if k not in cur_tables]

        print(
            f"Tables: same={len(same)} | changed={len(changed)} | "
            f"missing_baseline={len(missing_baseline)} | missing_current={len(missing_current)}"
        )

        if changed:
            print("\nCHANGED (ggü. Baseline):")
            for k in changed[:50]:
                print(f"- {k} (baseline={baseline_tables[k][:12]}..., current={cur_tables[k][:12]}...)")

        if missing_baseline:
            print("\nMISSING in Baseline (neu):")
            for k in missing_baseline[:50]:
                print(f"- {k}")

        if missing_current:
            print("\nMISSING in Current (Baseline hat mehr):")
            for k in missing_current[:50]:
                print(f"- {k}")

        # Figures diff (kurz)
        same_f = [k for k in cur_figures.keys() if k in baseline_figures and cur_figures[k] == baseline_figures[k]]
        changed_f = [k for k in cur_figures.keys() if k in baseline_figures and cur_figures[k] != baseline_figures[k]]
        missing_baseline_f = [k for k in cur_figures.keys() if k not in baseline_figures]
        missing_current_f = [k for k in baseline_figures.keys() if k not in cur_figures]

        print(
            f"Figures: same={len(same_f)} | changed={len(changed_f)} | "
            f"missing_baseline={len(missing_baseline_f)} | missing_current={len(missing_current_f)}"
        )

        if changed_f:
            print("\nCHANGED Figures (ggü. Baseline):")
            for k in changed_f[:50]:
                print(f"- {k} (baseline={baseline_figures[k][:12]}..., current={cur_figures[k][:12]}...)")

        if missing_baseline_f:
            print("\nMISSING Figures in Baseline (neu):")
            for k in missing_baseline_f[:50]:
                print(f"- {k}")

        if missing_current_f:
            print("\nMISSING Figures in Current (Baseline hat mehr):")
            for k in missing_current_f[:50]:
                print(f"- {k}")

        # Strict-Fail, sobald irgendein Diff existiert
        if changed or missing_baseline or missing_current or changed_f or missing_baseline_f or missing_current_f:
            print("FAIL: Baseline-Diff erkannt. Wenn Änderung beabsichtigt: --freeze-baseline ausführen und committen.")
            raise SystemExit(2)

        # -----------------------------------------------------------------
        # G) Figures presence check
        # -----------------------------------------------------------------
        print("\n" + "=" * 55)
        print("G) Reports/Figures Präsenzcheck")
        print("=" * 55)

        pngs = _collect_files(figures_dir, (".png",))
        svgs = _collect_files(figures_dir, (".svg",))
        print(f"PNG: {len(pngs)} | SVG: {len(svgs)}")

        if pngs:
            print("\nTop PNGs:")
            for p in pngs[:10]:
                print(f"- {p.name}")
    else:
        print("\n" + "=" * 55)
        print("E/G) Baseline-Diff & Figures-Checks: übersprungen")
        print("=" * 55)
        print("Hinweis: Aktivieren mit --strict-baseline-diff (oder Baseline aktualisieren via --freeze-baseline).")

    print("\nChecks abgeschlossen.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation checks & baseline diff for MLPerf-Tiny EDA pipeline.")
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH), help="Path to interim parquet")
    parser.add_argument("--tables-dir", type=str, default=str(TABLES_DIR), help="reports/tables directory")
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR), help="reports/figures directory")
    parser.add_argument("--baseline-dir", type=str, default=str(BASELINE_DIR_DEFAULT), help="docs/baseline directory")
    parser.add_argument(
        "--strict-baseline-diff",
        action="store_true",
        help="Führt Baseline-Diff (Tables/Figures) aus und bricht bei Abweichungen mit Exit-Code 2 ab.",
    )

    parser.add_argument(
        "--set-baseline",
        "--freeze-baseline",
        dest="set_baseline",
        action="store_true",
        help="Baseline einfrieren (Manifeste + Meta aus aktuellen reports/ schreiben)",
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    baseline_dir = Path(args.baseline_dir)

    if args.set_baseline:
        freeze_baseline(tables_dir=tables_dir, figures_dir=figures_dir, baseline_dir=baseline_dir)
        return 0

    run_checks(parquet_path, tables_dir, figures_dir, baseline_dir, strict_baseline_diff=args.strict_baseline_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
