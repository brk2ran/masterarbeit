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

# Baseline-Policy (Option A): Baseline-Kandidaten ab dieser Round zulassen
MIN_BASELINE_ROUND = "v0.7"
ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]


# ---------------------------
# Helpers: printing / formatting
# ---------------------------
def _heading(title: str, char: str = "=", width_min: int = 60) -> None:
    print("\n" + title)
    print(char * max(width_min, len(title)))


def _hdr(title: str) -> None:
    _heading(title, "=")


def _sec(title: str) -> None:
    _heading(title, "-")


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


def _print_df(df: pd.DataFrame, *, verbose: bool, max_rows: int = 30, title: Optional[str] = None) -> None:
    if title:
        print(title)
    if df.empty:
        print("(empty)")
        return
    if verbose or len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows; Details ggf. in CSV)")


def _boolish(series: pd.Series) -> pd.Series:
    """
    Normalize "bool-ish" values to True/False with NaN for unparseable entries.
    Accepted: True/False, 1/0, 'true'/'false', '1'/'0'.
    """
    if series.dtype == bool:
        return series.astype("boolean")
    s = series.astype(str).str.strip().str.lower()
    return s.map({"true": True, "false": False, "1": True, "0": False}).astype("boolean")


def _round_rank(r: object) -> int:
    s = str(r).strip()
    try:
        return ROUND_ORDER.index(s)
    except ValueError:
        return 10_000


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

    same: List[str] = []
    changed: List[str] = []
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

    # UNKNOWN Ursachen kurz
    if "model_mlc_source" in df.columns:
        u = df[df["scope_status"] == "UNKNOWN"]
        if not u.empty:
            vc = u["model_mlc_source"].value_counts(dropna=False)
            print("\nUNKNOWN Ursachen (model_mlc_source):")
            for k, v in vc.items():
                print(f"  - {k}: {int(v)}")

    # Gate: IN_SCOPE darf nicht ausschließlich UNKNOWN/OUT_OF_SCOPE Tasks haben
    in_scope_df = df[df["scope_status"] == "IN_SCOPE"]
    if not in_scope_df.empty:
        piv = in_scope_df.groupby(["round", "task_canon"], dropna=False).size().unstack(fill_value=0)
        core = ["AD", "IC", "KWS", "VWW"]
        bad_rounds = [str(rnd) for rnd in piv.index if sum(int(piv.loc[rnd].get(t, 0)) for t in core) == 0]
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
    _print_df(cov, verbose=True)  # klein, immer vollständig
    _write_csv(cov, tables_dir / "coverage_round_task_all.csv", "coverage_round_task_all")

    _sec("B2) Coverage-Checks (Round × Task × Model)")
    cov_rtm = (
        df.groupby(["round", "task_canon", "model_mlc_canon", "scope_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["round", "task_canon", "model_mlc_canon", "scope_status"])
    )
    _write_csv(cov_rtm, tables_dir / "coverage_round_task_model.csv", "coverage_round_task_model")

    anom = cov_rtm[
        (cov_rtm["scope_status"] != "IN_SCOPE")
        | (cov_rtm["task_canon"].isin(["UNKNOWN", "OUT_OF_SCOPE"]))
        | (cov_rtm["model_mlc_canon"] == "UNKNOWN")
    ].copy()

    if anom.empty:
        print("OK: Keine Anomalien in Round×Task×Model (nur IN_SCOPE + kanonische Modelle).")
    else:
        _print_df(anom, verbose=verbose, max_rows=30, title="Anomalien (scope!=IN_SCOPE oder UNKNOWN/OUT_OF_SCOPE):")

    _sec("B3) Metric Coverage (auf IN_SCOPE + latency Basis)")

    subsets = get_analysis_subsets(df)
    base = subsets.get("DS_LATENCY")
    if base is None or base.empty:
        base = df[(df["scope_status"] == "IN_SCOPE") & (df.get("latency_us").notna())].copy()

    for metric in ["energy_uj", "power_mw", "accuracy", "auc"]:
        covm = metric_coverage_by_round_task(base, metric)
        _write_csv(covm, tables_dir / f"coverage_{metric}_by_round_task.csv", f"coverage_{metric}_by_round_task")

        total_rows = int(covm["rows"].sum()) if "rows" in covm.columns else 0
        total_with = int(covm["rows_with_metric"].sum()) if "rows_with_metric" in covm.columns else 0
        share = (total_with / total_rows) if total_rows > 0 else 0.0
        print(f"Metric: {metric} | rows_with_metric={total_with}/{total_rows} | share={share:.3f}")

        if share > 0 or verbose:
            if verbose:
                _print_df(covm, verbose=True)
            else:
                top = covm.sort_values("share_metric", ascending=False).head(8)
                _print_df(top, verbose=True, title="Top share_metric:")


def _unknown_summary(df: pd.DataFrame, tables_dir: Path) -> None:
    _sec("B4) UNKNOWN Summary by Round")
    summ = unknown_summary_by_round(df)
    _print_df(summ, verbose=True)
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
        _print_df(sample, verbose=True, title="Beispiel-Duplikate (Top 20):")


def _metric_plausibility(name: str, s: pd.Series, *, warn_max: float) -> None:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return
    q = _quantiles(s, qs=(0.0, 0.5, 1.0))
    print(f"\n{name}: min={q[0.0]:.3g}, median={q[0.5]:.3g}, max={q[1.0]:.3g}")
    if (s <= 0).any():
        print(f"FAIL: {name} enthält nicht-positive Werte (<=0).")
    if q[1.0] > warn_max:
        print(f"WARN: {name} max > {warn_max:.3g} (Ausreißer / Mapping prüfen).")


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
    miss_df = miss.reset_index()
    miss_df.columns = ["metric", "share_na"]
    _print_df(miss_df, verbose=True, title="Missingness (Anteil NA):")

    for main in ["latency_us", "energy_uj"]:
        if main in in_scope.columns:
            print(f"{main} Not-NA: {int(in_scope[main].notna().sum())}")

    lat = _safe_num(in_scope, "latency_us")
    en = _safe_num(in_scope, "energy_uj")

    _metric_plausibility("latency_us", lat, warn_max=1e8)
    _metric_plausibility("energy_uj", en, warn_max=1e7)

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
        _sec("C3) Extra Quantile (verbose)")
        if not lat.dropna().empty:
            print("latency_us quantiles:", _quantiles(lat, qs=(0.25, 0.5, 0.75, 0.95, 0.99)))
        if not en.dropna().empty:
            print("energy_uj quantiles:", _quantiles(en, qs=(0.25, 0.5, 0.75, 0.95, 0.99)))


# ---------------------------
# QA Gates for trend tables
# ---------------------------
def _trendtable_schema_gate(
    tables_dir: Path,
    *,
    min_n: int,
    strict: bool,
    verbose: bool,
) -> None:
    """
    Prüft, ob die Trendtabellen die erweiterten Spalten enthalten und plausibel sind.
    """
    _sec("H1) Trendtable Schema Gate (q10/q90/iqr/low_n)")

    checks = [
        ("trend_latency_us_round_task.csv", "latency_us", ["round", "task"]),
        ("trend_energy_uj_round_task.csv", "energy_uj", ["round", "task"]),
    ]

    failures: List[str] = []

    for filename, prefix, key_cols in checks:
        path = tables_dir / filename
        if not path.exists():
            msg = f"FEHLT: {filename} (erwarte nach python -m src.eda)"
            print("WARN:", msg)
            failures.append(msg)
            continue

        df = pd.read_csv(path)

        expected = set(
            key_cols
            + [
                f"{prefix}_n",
                f"{prefix}_median",
                f"{prefix}_q25",
                f"{prefix}_q75",
                f"{prefix}_q10",
                f"{prefix}_q90",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_iqr",
                f"{prefix}_low_n",
            ]
        )
        missing = expected - set(df.columns)
        if missing:
            msg = f"{filename}: Missing columns: {sorted(missing)}"
            print("WARN:", msg)
            failures.append(msg)
            continue

        n = pd.to_numeric(df[f"{prefix}_n"], errors="coerce")
        if n.isna().any():
            msg = f"{filename}: {prefix}_n enthält NaN / nicht-numerisch"
            print("WARN:", msg)
            failures.append(msg)

        low_norm = _boolish(df[f"{prefix}_low_n"])
        if low_norm.isna().any():
            msg = f"{filename}: {prefix}_low_n ist nicht vollständig bool-artig"
            print("WARN:", msg)
            failures.append(msg)

        iqr = pd.to_numeric(df[f"{prefix}_iqr"], errors="coerce")
        iqr_bad = iqr.dropna() < -1e-12
        if iqr_bad.any():
            msg = f"{filename}: {prefix}_iqr hat negative Werte (min={float(iqr.min()):.3g})"
            print("WARN:", msg)
            failures.append(msg)

        # Quantile-Ordnung: q10 ≤ q25 ≤ median ≤ q75 ≤ q90
        q10 = pd.to_numeric(df[f"{prefix}_q10"], errors="coerce")
        q25 = pd.to_numeric(df[f"{prefix}_q25"], errors="coerce")
        med = pd.to_numeric(df[f"{prefix}_median"], errors="coerce")
        q75 = pd.to_numeric(df[f"{prefix}_q75"], errors="coerce")
        q90 = pd.to_numeric(df[f"{prefix}_q90"], errors="coerce")
        mn = pd.to_numeric(df[f"{prefix}_min"], errors="coerce")
        mx = pd.to_numeric(df[f"{prefix}_max"], errors="coerce")

        mask_core = q10.notna() & q25.notna() & med.notna() & q75.notna() & q90.notna()
        if mask_core.any():
            bad_core = (q10[mask_core] > q25[mask_core]) | (q25[mask_core] > med[mask_core]) | (
                med[mask_core] > q75[mask_core]
            ) | (q75[mask_core] > q90[mask_core])
            if bad_core.any():
                ex = df.loc[mask_core].loc[bad_core].head(5)[
                    key_cols
                    + [f"{prefix}_q10", f"{prefix}_q25", f"{prefix}_median", f"{prefix}_q75", f"{prefix}_q90"]
                ]
                print("WARN:", f"{filename}: Quantile-Reihenfolge verletzt. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: quantile order violated (core)")

        mask_edges = mask_core & mn.notna() & mx.notna()
        if mask_edges.any():
            bad_edges = (mn[mask_edges] > q10[mask_edges]) | (q90[mask_edges] > mx[mask_edges])
            if bad_edges.any():
                ex = df.loc[mask_edges].loc[bad_edges].head(5)[key_cols + [f"{prefix}_min", f"{prefix}_q10", f"{prefix}_q90", f"{prefix}_max"]]
                print("WARN:", f"{filename}: Randbedingungen verletzt. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: quantile order violated (edges)")

        # low_n korrekt gemäß min_n?
        m = n.notna() & low_norm.notna()
        if m.any():
            expected_low = (n[m].fillna(0).astype(int) < int(min_n)).astype(bool).values
            got_low = low_norm[m].astype(bool).values
            if np.any(expected_low != got_low):
                msg = f"{filename}: {prefix}_low_n passt nicht zu min_n={min_n} (mind. ein mismatch)"
                print("WARN:", msg)
                failures.append(msg)

        if verbose:
            low_share = float(low_norm.astype("float").mean(skipna=True))
            print(f"OK: {filename} geladen | rows={len(df)} | low_n_share~{low_share:.3f}")

    if failures:
        if strict:
            raise SystemExit(2)
        print("\nTrendtable Schema Gate: WARNINGS (kein Abbruch, da strict=False).")
    else:
        print("OK: Trendtabellen enthalten erwartete erweiterten Spalten und sind plausibel.")


def _indexed_tables_gate(
    tables_dir: Path,
    *,
    min_n: int,
    strict: bool,
    verbose: bool,
    min_baseline_round: str = MIN_BASELINE_ROUND,
) -> None:
    """
    QA-Gate für *_indexed.csv:
    - Baseline-Round darf nicht vor min_baseline_round liegen (Option A -> v0.7)
    - Pre-Baseline (round < baseline_round) muss Index-Spalten NaN haben
    - Baseline-Row: median_index ~= 1.0 (wenn baseline exists)
    - baseline_n >= min_n (wenn baseline exists)
    """
    _sec("H2) Indexed Tables Gate (Baseline ab v0.7, Pre-Baseline NaN)")

    checks = [
        ("trend_latency_us_round_task_indexed.csv", "latency_us", ["round", "task"]),
        ("trend_energy_uj_round_task_indexed.csv", "energy_uj", ["round", "task"]),
    ]

    failures: List[str] = []
    min_base_rank = _round_rank(min_baseline_round)

    for filename, prefix, key_cols in checks:
        path = tables_dir / filename
        if not path.exists():
            msg = f"FEHLT: {filename} (erwarte nach python -m src.eda)"
            print("WARN:", msg)
            failures.append(msg)
            continue

        df = pd.read_csv(path)

        # required columns
        required = set(
            key_cols
            + [
                f"{prefix}_n",
                f"{prefix}_median",
                f"{prefix}_q25",
                f"{prefix}_q75",
                "baseline_round",
                "baseline_n",
                "baseline_median",
                f"{prefix}_median_index",
                f"{prefix}_q25_index",
                f"{prefix}_q75_index",
            ]
        )
        missing = required - set(df.columns)
        if missing:
            msg = f"{filename}: Missing columns: {sorted(missing)}"
            print("WARN:", msg)
            failures.append(msg)
            continue

        # ranks
        rr = df["round"].map(_round_rank)
        br = df["baseline_round"].astype(str).where(df["baseline_round"].notna(), np.nan)
        br_rank = br.map(lambda x: _round_rank(x) if isinstance(x, str) and x != "nan" else np.nan)

        # baseline policy: baseline >= min_baseline_round
        mask_has_base = df["baseline_round"].notna() & df["baseline_median"].notna()
        if mask_has_base.any():
            bad_base = br_rank[mask_has_base] < min_base_rank
            if bad_base.any():
                ex = df.loc[mask_has_base].loc[bad_base].head(10)[key_cols + ["baseline_round", "baseline_n", "baseline_median"]]
                print("WARN:", f"{filename}: baseline_round vor {min_baseline_round} gefunden. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: baseline_round < {min_baseline_round}")

        # baseline_n >= min_n
        bn = pd.to_numeric(df["baseline_n"], errors="coerce")
        if mask_has_base.any():
            bad_bn = bn[mask_has_base] < int(min_n)
            if bad_bn.any():
                ex = df.loc[mask_has_base].loc[bad_bn].head(10)[key_cols + ["baseline_round", "baseline_n"]]
                print("WARN:", f"{filename}: baseline_n < min_n={min_n}. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: baseline_n < min_n")

        # pre-baseline -> index cols must be NaN
        idx_cols = [c for c in df.columns if c.endswith("_index") and c.startswith(prefix)]
        pre = mask_has_base & (rr < br_rank)
        if pre.any():
            any_non_na = df.loc[pre, idx_cols].notna().any(axis=1)
            if any_non_na.any():
                ex = df.loc[pre].loc[any_non_na].head(10)[key_cols + ["baseline_round"] + idx_cols]
                print("WARN:", f"{filename}: Pre-Baseline enthält nicht-NaN in Index-Spalten. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: pre-baseline has non-NaN index")

        # v0.5 sanity: should be pre-baseline and thus NaN (wenn baseline exists)
        mask_v05 = (df["round"].astype(str) == "v0.5") & mask_has_base
        if mask_v05.any():
            any_non_na_v05 = df.loc[mask_v05, idx_cols].notna().any(axis=1)
            if any_non_na_v05.any():
                ex = df.loc[mask_v05].loc[any_non_na_v05].head(10)[key_cols + ["baseline_round"] + idx_cols]
                print("WARN:", f"{filename}: v0.5 hat Indexwerte obwohl Baseline existiert. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: v0.5 has non-NaN index")

        # baseline row -> median_index approx 1.0 (tolerant)
        # baseline row(s): round == baseline_round
        is_base_row = mask_has_base & (df["round"].astype(str) == df["baseline_round"].astype(str))
        if is_base_row.any():
            med_idx = pd.to_numeric(df.loc[is_base_row, f"{prefix}_median_index"], errors="coerce")
            bad = med_idx.notna() & (np.abs(med_idx - 1.0) > 1e-6)
            if bad.any():
                ex = df.loc[is_base_row].loc[bad].head(10)[key_cols + ["baseline_round", f"{prefix}_median_index", "baseline_median", f"{prefix}_median"]]
                print("WARN:", f"{filename}: baseline row median_index != 1.0. Beispiele:\n{ex.to_string(index=False)}")
                failures.append(f"{filename}: baseline row median_index != 1.0")

        if verbose:
            print(f"OK: {filename} geladen | rows={len(df)} | idx_cols={len(idx_cols)} | min_baseline_round={min_baseline_round}")

    if failures:
        if strict:
            raise SystemExit(2)
        print("\nIndexed Tables Gate: WARNINGS (kein Abbruch, da strict=False).")
    else:
        print(f"OK: *_indexed.csv erfüllen Baseline-Policy (>= {min_baseline_round}) und Pre-Baseline=NaN.")


# ---------------------------
# Misc checks
# ---------------------------
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


# ---------------------------
# Baseline freeze/diff
# ---------------------------
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
    strict_trend_schema: bool = False,
    strict_indexed_gate: bool = False,
    min_n: int = 5,
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

    # QA-Gates für Trendtabellen
    _trendtable_schema_gate(tables_dir, min_n=min_n, strict=strict_trend_schema, verbose=verbose)
    _indexed_tables_gate(
        tables_dir,
        min_n=min_n,
        strict=strict_indexed_gate,
        verbose=verbose,
        min_baseline_round=MIN_BASELINE_ROUND,
    )

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

    parser.add_argument("--min-n", type=int, default=5, help="Low-n threshold used in trend schema checks")
    parser.add_argument("--strict-trend-schema", action="store_true", help="Exit non-zero if trend schema gate fails")
    parser.add_argument("--strict-indexed-gate", action="store_true", help="Exit non-zero if indexed tables gate fails")

    args = parser.parse_args(argv)

    run_checks(
        parquet_path=args.parquet,
        reports_dir=args.reports_dir,
        freeze_baseline=args.freeze_baseline,
        baseline_dir=args.baseline_dir,
        verbose=args.verbose,
        strict_trend_schema=args.strict_trend_schema,
        strict_indexed_gate=args.strict_indexed_gate,
        min_n=args.min_n,
    )


if __name__ == "__main__":
    main()
