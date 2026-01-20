from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
FIGURES_DIR_DEFAULT = REPORTS_DIR_DEFAULT / "figures"
BASELINE_DIR_DEFAULT = REPORTS_DIR_DEFAULT / "baseline"


# ---------------------------
# Helpers: printing / formatting
# ---------------------------
def _hdr(title: str) -> None:
    print("\n" + title)
    print("=" * max(60, len(title)))


def _sec(title: str) -> None:
    print("\n" + title)
    print("-" * max(60, len(title)))


def _pct(x: float) -> str:
    return f"{x:.3f}"


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _quantiles(s: pd.Series, qs=(0.0, 0.5, 1.0)) -> Dict[float, float]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return {q: float("nan") for q in qs}
    qv = s2.quantile(list(qs)).to_dict()
    return {float(k): float(v) for k, v in qv.items()}


# ---------------------------
# Baseline manifests
# ---------------------------
@dataclass
class FileEntry:
    name: str
    relpath: str
    bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_for_dir(root: Path, patterns: Tuple[str, ...]) -> List[FileEntry]:
    entries: List[FileEntry] = []
    if not root.exists():
        return entries
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if not p.is_file():
                continue
            rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            entries.append(
                FileEntry(
                    name=p.name,
                    relpath=rel,
                    bytes=p.stat().st_size,
                    sha256=_sha256_file(p),
                )
            )
    return entries


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _entries_to_dict(entries: List[FileEntry]) -> Dict[str, Dict[str, object]]:
    return {
        e.relpath: {"name": e.name, "relpath": e.relpath, "bytes": e.bytes, "sha256": e.sha256}
        for e in entries
    }


def _diff_manifests(base: Dict[str, Dict[str, object]], curr: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    base_keys = set(base.keys())
    curr_keys = set(curr.keys())

    missing_current = sorted(base_keys - curr_keys)
    missing_baseline = sorted(curr_keys - base_keys)

    same = []
    changed = []
    for k in sorted(base_keys & curr_keys):
        if base[k]["sha256"] == curr[k]["sha256"] and base[k]["bytes"] == curr[k]["bytes"]:
            same.append(k)
        else:
            changed.append(k)

    return {
        "same": same,
        "changed": changed,
        "missing_baseline": missing_baseline,
        "missing_current": missing_current,
    }


# ---------------------------
# Load + guards
# ---------------------------
def _load_df(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "task_canon" not in df.columns or "model_mlc_canon" not in df.columns or "scope_status" not in df.columns:
        df = add_features(df)
    return df


# ---------------------------
# Checks
# ---------------------------
def _scope_checks(df: pd.DataFrame, *, verbose: bool) -> None:
    _sec("A) Scope-Checks (OUT_OF_SCOPE / IN_SCOPE / UNKNOWN)")

    total = len(df)
    in_scope = int((df["scope_status"] == "IN_SCOPE").sum())
    out_scope = int((df["scope_status"] == "OUT_OF_SCOPE").sum())
    unknown = int((df["scope_status"] == "UNKNOWN").sum())

    print(f"Rows total: {total}")
    print(f"Rows in_scope: {in_scope} ({_pct(in_scope / total)})")
    print(f"Rows out_of_scope: {out_scope} ({_pct(out_scope / total)})")
    print(f"Rows UNKNOWN (ohne out_of_scope): {unknown} ({_pct(unknown / total)})")

    if "model_mlc_source" in df.columns:
        u = df[df["scope_status"] == "UNKNOWN"]
        if not u.empty:
            vc = u["model_mlc_source"].value_counts(dropna=False)
            print("\nUNKNOWN Ursachen (model_mlc_source):")
            for k, v in vc.items():
                print(f"  - {k}: {int(v)}")

    # Guard gegen "leere IN_SCOPE Kategorien" (sollte nie passieren)
    in_scope_df = df[df["scope_status"] == "IN_SCOPE"]
    if not in_scope_df.empty:
        piv = in_scope_df.groupby(["round", "task_canon"], dropna=False).size().unstack(fill_value=0)
        bad_rounds = []
        for rnd in piv.index:
            row = piv.loc[rnd]
            if sum(int(row.get(t, 0)) for t in ["AD", "IC", "KWS", "VWW"]) == 0:
                bad_rounds.append(str(rnd))
        if bad_rounds:
            print("\nFAIL: Rounds mit IN_SCOPE Rows, aber 0 in AD/IC/KWS/VWW (Task Mapping defekt):")
            for r in bad_rounds:
                print(f"  - {r}")
        elif verbose:
            print("\nOK: Keine Round mit IN_SCOPE>0 und leeren Kern-Tasks.")


def _write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[table] {label}: {len(df)} rows -> {path}")


def _coverage_checks(df: pd.DataFrame, tables_dir: Path, *, verbose: bool) -> None:
    _sec("B) Coverage-Checks (Round × Task)")

    cov = round_task_counts(df, include_unknown=True, include_out_of_scope=True)
    print(cov.to_string(index=False))
    _write_csv(cov, tables_dir / "coverage_round_task_all.csv", "coverage_round_task_all")

    _sec("B2) Coverage-Checks (Round × Task × Model)")

    cov_rtm = (
        df.groupby(["round", "task_canon", "model_mlc_canon", "scope_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["round", "task_canon", "model_mlc_canon", "scope_status"])
    )
    _write_csv(cov_rtm, tables_dir / "coverage_round_task_model.csv", "coverage_round_task_model")

    # Terminal: nur Anomalien (statt riesige Liste)
    anom = cov_rtm[
        (cov_rtm["scope_status"] != "IN_SCOPE") | (cov_rtm["task_canon"].isin(["UNKNOWN", "OUT_OF_SCOPE"])) | (cov_rtm["model_mlc_canon"] == "UNKNOWN")
    ].copy()

    if anom.empty:
        print("OK: Keine Anomalien in Round×Task×Model (nur IN_SCOPE + kanonische Modelle).")
    else:
        print("Anomalien (scope!=IN_SCOPE oder UNKNOWN/OUT_OF_SCOPE):")
        # bei verbose komplett, sonst capped
        show = anom if verbose else anom.head(30)
        print(show.to_string(index=False))
        if not verbose and len(anom) > 30:
            print(f"... ({len(anom)} Anomalien total; Details in coverage_round_task_model.csv)")

    _sec("B3) Metric Coverage (auf IN_SCOPE + latency Basis)")

    subsets = get_analysis_subsets(df)
    base = subsets.get("DS_LATENCY")
    if base is None or base.empty:
        base = df[(df["scope_status"] == "IN_SCOPE") & (df.get("latency_us").notna())].copy()

    # Wir schreiben weiterhin CSVs für alle Metrics, aber drucken im Terminal nur sinnvolle Zusammenfassungen.
    for metric in ["energy_uj", "power_mw", "accuracy", "auc"]:
        covm = metric_coverage_by_round_task(base, metric)
        _write_csv(covm, tables_dir / f"coverage_{metric}_by_round_task.csv", f"coverage_{metric}_by_round_task")

        total_rows = int(covm["rows"].sum()) if "rows" in covm.columns else 0
        total_with = int(covm["rows_with_metric"].sum()) if "rows_with_metric" in covm.columns else 0
        share = (total_with / total_rows) if total_rows > 0 else 0.0

        # Terminal: nur eine Summary-Zeile; Detailtabelle nur wenn Coverage > 0 oder verbose
        print(f"\nMetric: {metric} | rows_with_metric={total_with}/{total_rows} | share={share:.3f}")

        if share > 0 or verbose:
            # Optional: statt volle Tabelle nur Top/Bottom Runden zeigen
            if not verbose:
                top = covm.sort_values("share_metric", ascending=False).head(8)
                print("Top share_metric:")
                print(top.to_string(index=False))
            else:
                print(covm.to_string(index=False))


def _unknown_summary(df: pd.DataFrame, tables_dir: Path) -> None:
    _sec("B4) UNKNOWN Summary by Round")

    summ = unknown_summary_by_round(df)
    print(summ.to_string(index=False))
    _write_csv(summ, tables_dir / "unknown_summary_by_round.csv", "unknown_summary_by_round")


def _duplicates_check(df: pd.DataFrame, *, verbose: bool) -> None:
    _sec("D) Duplikat-Checks")

    preferred = ["public_id", "round", "system_name", "host_processor_frequency", "model_mlc"]
    key = [c for c in preferred if c in df.columns]
    if len(key) < 3:
        fallback = ["public_id", "round", "model_mlc_effective", "model_mlc_canon"]
        key = [c for c in fallback if c in df.columns]

    if not key:
        print("WARN: Kein sinnvoller Key für Duplikat-Check gefunden (Spalten fehlen).")
        return

    dup = df.duplicated(subset=key, keep=False)
    ndup = int(dup.sum())
    print(f"Duplicate auf KEY {key}: {ndup}")

    if ndup > 0 and verbose:
        sample = df.loc[dup, key].head(20)
        print("\nBeispiel-Duplikate (Top 20):")
        print(sample.to_string(index=False))


def _missingness_plausibility(df: pd.DataFrame, *, verbose: bool) -> None:
    _sec("C) Missingness & Plausibility (in_scope)")

    in_scope = df[df["scope_status"] == "IN_SCOPE"].copy()
    if in_scope.empty:
        print("WARN: Keine IN_SCOPE Rows vorhanden – Missingness/Plausibility übersprungen.")
        return

    metrics = ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"]
    cols = [c for c in metrics if c in in_scope.columns]
    if not cols:
        print("WARN: Keine Metrikspalten gefunden.")
        return

    miss = in_scope[cols].isna().mean().sort_values(ascending=False)
    print("\nMissingness (Anteil NA):")
    for k, v in miss.items():
        print(f"{k:>12}  {v:.5f}")

    # Not-NA Counts: nur für die zwei Hauptmetriken (redundanzarm)
    for main in ["latency_us", "energy_uj"]:
        if main in in_scope.columns:
            print(f"\n{main} Not-NA: {int(in_scope[main].notna().sum())}")

    # Plausibility (nur Hauptmetriken)
    lat = _safe_num(in_scope, "latency_us")
    if not lat.dropna().empty:
        q = _quantiles(lat)
        print(f"\nlatency_us: min={q[0.0]:.3g}, median={q[0.5]:.3g}, max={q[1.0]:.3g}")
        if (lat <= 0).any():
            print("FAIL: latency_us enthält nicht-positive Werte (<=0).")
        if q[1.0] > 1e8:
            print("WARN: latency_us max > 1e8 µs (Ausreißer oder doppelte Konversion möglich).")

    en = _safe_num(in_scope, "energy_uj")
    if not en.dropna().empty:
        q = _quantiles(en)
        print(f"\nenergy_uj: min={q[0.0]:.3g}, median={q[0.5]:.3g}, max={q[1.0]:.3g}")
        if (en <= 0).any():
            print("FAIL: energy_uj enthält nicht-positive Werte (<=0).")
        if q[1.0] > 1e7:
            print("WARN: energy_uj max > 1e7 µJ (Ausreißer möglich).")

    _sec("C2) Unit-Safety Heuristik (ms→µs genau einmal)")
    if not lat.dropna().empty:
        q = _quantiles(lat, qs=(0.5, 0.95, 0.99))
        p50, p95, p99 = q[0.5], q[0.95], q[0.99]
        if p50 < 500 and p95 < 10_000:
            print(
                f"WARN: latency_us wirkt sehr klein (p50={p50:.3g}, p95={p95:.3g}). "
                "Möglicherweise liegt noch ms vor (Konversion nicht angewendet)."
            )
        elif p50 > 1e7 or p99 > 1e9:
            print(
                f"WARN: latency_us wirkt sehr groß (p50={p50:.3g}, p99={p99:.3g}). "
                "Möglicherweise wurde ms→µs doppelt konvertiert."
            )
        else:
            print(f"OK: latency_us Skala plausibel (p50={p50:.3g}, p95={p95:.3g}, p99={p99:.3g}).")
    else:
        print("SKIP: latency_us hat keine Werte; Unit-Safety nicht prüfbar.")

    if verbose:
        # Optional: zusätzliche Quantile nur bei verbose
        _sec("C3) Extra Quantile (verbose)")
        if not lat.dropna().empty:
            q = _quantiles(lat, qs=(0.25, 0.5, 0.75, 0.95, 0.99))
            print("latency_us quantiles:", q)
        if not en.dropna().empty:
            q = _quantiles(en, qs=(0.25, 0.5, 0.75, 0.95, 0.99))
            print("energy_uj quantiles:", q)


def _clustering_sanity(reports_dir: Path) -> None:
    _sec("F) Clustering-Sanity-Check (hardware_clusters.csv)")

    p1 = reports_dir / "tables" / "hardware_clusters.csv"
    p2 = reports_dir / "hardware_clusters.csv"
    path = p1 if p1.exists() else (p2 if p2.exists() else None)

    if path is None:
        print("INFO: hardware_clusters.csv nicht vorhanden – übersprungen.")
        return

    df = pd.read_csv(path)
    print(f"hardware_clusters.csv gefunden: {path} ({len(df)} rows)")
    if "cluster_id" in df.columns:
        print(f"cluster_id unique: {df['cluster_id'].nunique(dropna=False)}")
    else:
        print("WARN: Spalte 'cluster_id' fehlt.")


def _baseline_freeze(
    baseline_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    parquet_path: Path,
) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.now().isoformat(timespec="seconds"), "parquet": str(parquet_path).replace("\\", "/")}
    _save_json(baseline_dir / "meta.json", meta)

    tables_entries = _manifest_for_dir(tables_dir, ("*.csv",))
    figures_entries = _manifest_for_dir(figures_dir, ("*.png", "*.svg"))

    _save_json(baseline_dir / "tables_manifest.json", _entries_to_dict(tables_entries))
    _save_json(baseline_dir / "figures_manifest.json", _entries_to_dict(figures_entries))

    print(f"OK: Baseline eingefroren in: {baseline_dir}")


def _baseline_diff(baseline_dir: Path, tables_dir: Path, figures_dir: Path) -> None:
    _sec("E) Reports/Tables Diff (gegen Baseline)")

    meta_p = baseline_dir / "meta.json"
    tab_p = baseline_dir / "tables_manifest.json"
    fig_p = baseline_dir / "figures_manifest.json"

    if not (meta_p.exists() and tab_p.exists() and fig_p.exists()):
        print("WARN: Baseline nicht vollständig vorhanden (meta.json / tables_manifest.json / figures_manifest.json fehlt).")
        print("-> Baseline einfrieren: python -m src.checks --freeze-baseline")
        return

    base_tables = _load_json(tab_p)
    base_figs = _load_json(fig_p)

    curr_tables = _entries_to_dict(_manifest_for_dir(tables_dir, ("*.csv",)))
    curr_figs = _entries_to_dict(_manifest_for_dir(figures_dir, ("*.png", "*.svg")))

    dt = _diff_manifests(base_tables, curr_tables)
    df = _diff_manifests(base_figs, curr_figs)

    print(
        f"Tables: same={len(dt['same'])} | changed={len(dt['changed'])} | "
        f"missing_baseline={len(dt['missing_baseline'])} | missing_current={len(dt['missing_current'])}"
    )

    if dt["missing_baseline"]:
        print("\nMISSING in Baseline (neu):")
        for p in dt["missing_baseline"]:
            print(f" - {Path(p).name}")

    if dt["missing_current"]:
        print("\nMISSING aktuell (fehlend ggü. Baseline):")
        for p in dt["missing_current"]:
            print(f" - {Path(p).name}")

    if dt["changed"]:
        print("\nCHANGED (ggü. Baseline):")
        for p in dt["changed"][:15]:
            print(f" - {Path(p).name}")
        if len(dt["changed"]) > 15:
            print(f"... ({len(dt['changed'])} changed total)")

    _sec("G) Reports/Figures Präsenzcheck")
    pngs = sorted(list(figures_dir.glob("*.png"))) if figures_dir.exists() else []
    svgs = sorted(list(figures_dir.glob("*.svg"))) if figures_dir.exists() else []
    print(f"PNG: {len(pngs)} | SVG: {len(svgs)}")
    if pngs:
        print("Top PNGs:", ", ".join([p.name for p in pngs[:3]]))


def run_checks(
    parquet_path: Path,
    reports_dir: Path,
    freeze_baseline: bool = False,
    baseline_dir: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    baseline_dir = baseline_dir or (reports_dir / "baseline")

    _hdr("checks: Validierung & Baseline-Abgleich")

    df = _load_df(parquet_path)

    _scope_checks(df, verbose=verbose)
    _coverage_checks(df, tables_dir=tables_dir, verbose=verbose)
    _unknown_summary(df, tables_dir=tables_dir)
    _duplicates_check(df, verbose=verbose)
    _missingness_plausibility(df, verbose=verbose)
    _clustering_sanity(reports_dir)

    if freeze_baseline:
        _baseline_freeze(baseline_dir, tables_dir, figures_dir, parquet_path)

    _baseline_diff(baseline_dir, tables_dir, figures_dir)

    print("\nChecks abgeschlossen.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validation & baseline checks for MLPerf-Tiny EDA pipeline.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT, help="Path to consolidated parquet")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT, help="Reports directory")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR_DEFAULT, help="Baseline directory")
    parser.add_argument("--freeze-baseline", action="store_true", help="Freeze current reports as baseline")
    parser.add_argument("--verbose", action="store_true", help="Print full tables / extra diagnostics to console")

    args = parser.parse_args(argv)

    run_checks(
        parquet_path=args.parquet,
        reports_dir=args.reports_dir,
        freeze_baseline=args.freeze_baseline,
        baseline_dir=args.baseline_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
