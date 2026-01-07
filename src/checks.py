from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.features import add_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "figures"
BASELINE_DIR_DEFAULT = PROJECT_ROOT / "docs" / "baseline_tables"


# -------------------------
# Utils
# -------------------------
def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_safe(path: Path) -> pd.DataFrame:
    # encoding="utf-8-sig" passt zu deinem Export
    return pd.read_csv(path, encoding="utf-8-sig")


def _print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _exists_or_warn(path: Path, label: str) -> None:
    if not path.exists():
        print(f"WARN: {label} nicht gefunden: {path}")


@dataclass(frozen=True)
class TableDiff:
    name: str
    status: str  # "same" | "changed" | "missing_baseline" | "missing_current"
    current_hash: str | None = None
    baseline_hash: str | None = None
    current_rows: int | None = None
    baseline_rows: int | None = None


# -------------------------
# Core checks
# -------------------------
def load_parquet(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Canon/Scope-Spalten ggf. ergänzen (falls altes Parquet)
    if "task_canon" not in df.columns or "model_mlc_canon" not in df.columns:
        df = add_features(df)

    return df


def check_scope_consistency(df: pd.DataFrame) -> None:
    """
    Prüft, ob OUT_OF_SCOPE nur dort vorkommt, wo du es erwartest.
    Standardannahme nach deiner Entscheidung:
      - OUT_OF_SCOPE == nur (model_mlc_canon == 1D_DS_CNN) in Round v1.3
    """
    _print_section("A) Scope-Checks (OUT_OF_SCOPE / IN_SCOPE / UNKNOWN)")

    required = {"round", "public_id", "model_mlc", "model_mlc_canon", "task_canon", "out_of_scope", "in_scope"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Fehlende Spalten im Parquet: {sorted(missing)}")
        return

    total = len(df)
    out_count = int(df["out_of_scope"].sum()) if "out_of_scope" in df.columns else 0
    in_count = int(df["in_scope"].sum()) if "in_scope" in df.columns else 0
    unk_count = int((df["task_canon"] == "UNKNOWN").sum())

    print(f"Rows total: {total}")
    print(f"Rows in_scope: {in_count} ({(in_count/total if total else 0):.3f})")
    print(f"Rows out_of_scope: {out_count} ({(out_count/total if total else 0):.3f})")
    print(f"Rows UNKNOWN (ohne out_of_scope): {unk_count} ({(unk_count/total if total else 0):.3f})")

    # Erwartungscheck: OUT_OF_SCOPE sollte nur 1D_DS_CNN in v1.3 sein
    out = df[df["out_of_scope"] == True].copy()  # noqa: E712
    if out.empty:
        print("OK: Keine OUT_OF_SCOPE-Zeilen gefunden.")
        return

    bad = out[~((out["model_mlc_canon"] == "1D_DS_CNN") & (out["round"].astype(str) == "v1.3"))]
    if bad.empty:
        print("OK: OUT_OF_SCOPE entspricht der Erwartung (nur 1D_DS_CNN in v1.3).")
    else:
        print("WARN: OUT_OF_SCOPE enthält unerwartete Fälle. Top 20:")
        cols = ["round", "public_id", "model_mlc", "model_mlc_canon", "task_canon"]
        print(bad[cols].head(20).to_string(index=False))


def check_round_task_coverage(df: pd.DataFrame) -> None:
    _print_section("B) Coverage-Checks (Round × Task)")

    if "round" not in df.columns or "task_canon" not in df.columns:
        print("ERROR: Spalten round/task_canon fehlen.")
        return

    # Coverage nur ohne OUT_OF_SCOPE
    d = df[df.get("out_of_scope", False) == False].copy()  # noqa: E712

    pivot = (
        d.groupby(["round", "task_canon"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .pivot_table(index="round", columns="task_canon", values="rows", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    print(pivot.to_string(index=False))


def check_missingness(df: pd.DataFrame) -> None:
    _print_section("C) Missingness & Plausibility (in_scope)")

    d = df[df.get("in_scope", False) == True].copy()  # noqa: E712
    if d.empty:
        print("WARN: in_scope Subset ist leer – Scope-Filter evtl. zu streng oder Daten fehlen.")
        return

    metrics = [c for c in ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"] if c in d.columns]
    if not metrics:
        print("WARN: Keine Metrikspalten gefunden (latency_us/energy_uj/power_mw/accuracy/auc).")
        return

    miss = d[metrics].isna().mean().sort_values(ascending=False)
    print("Missingness (Anteil NA):")
    print(miss.to_string())

    # Plausibility (sehr defensiv, nur grobe Warnungen)
    if "latency_us" in d.columns:
        s = pd.to_numeric(d["latency_us"], errors="coerce").dropna()
        if not s.empty:
            mn, md, mx = float(s.min()), float(s.median()), float(s.max())
            print(f"\nlatency_us: min={mn:.4g}, median={md:.4g}, max={mx:.4g}")
            if (s <= 0).any():
                print("WARN: latency_us enthält nicht-positive Werte (sollte > 0 sein).")

    if "energy_uj" in d.columns:
        s = pd.to_numeric(d["energy_uj"], errors="coerce").dropna()
        if not s.empty:
            mn, md, mx = float(s.min()), float(s.median()), float(s.max())
            print(f"energy_uj: min={mn:.4g}, median={md:.4g}, max={mx:.4g}")
            if (s <= 0).any():
                print("WARN: energy_uj enthält nicht-positive Werte (sollte > 0 sein).")

    if "power_mw" in d.columns:
        s = pd.to_numeric(d["power_mw"], errors="coerce").dropna()
        if not s.empty:
            mn, md, mx = float(s.min()), float(s.median()), float(s.max())
            print(f"power_mw: min={mn:.4g}, median={md:.4g}, max={mx:.4g}")
            if (s <= 0).any():
                print("WARN: power_mw enthält nicht-positive Werte (sollte > 0 sein).")


def check_duplicates(df: pd.DataFrame) -> None:
    _print_section("D) Duplikat-Checks")

    # Dein „Primary Key“ war zuletzt so definiert:
    # ['public_id', 'round', 'system_name', 'host_processor_frequency', 'model_mlc']
    key = [c for c in ["public_id", "round", "system_name", "host_processor_frequency", "model_mlc"] if c in df.columns]
    if len(key) < 3:
        print(f"WARN: Zu wenige Key-Spalten im Parquet für Duplikat-Check: {key}")
        return

    dup = df.duplicated(key).sum()
    print(f"Duplicate auf KEY {key}: {int(dup)}")
    if dup:
        print("Top 20 Duplikate (sortiert):")
        dups = df[df.duplicated(key, keep=False)].sort_values(key).head(20)
        print(dups[key].to_string(index=False))


def diff_tables(current_dir: Path, baseline_dir: Path) -> List[TableDiff]:
    current = sorted(current_dir.glob("*.csv")) if current_dir.exists() else []
    baseline = sorted(baseline_dir.glob("*.csv")) if baseline_dir.exists() else []

    current_map = {p.name: p for p in current}
    baseline_map = {p.name: p for p in baseline}
    all_names = sorted(set(current_map) | set(baseline_map))

    diffs: List[TableDiff] = []
    for name in all_names:
        c = current_map.get(name)
        b = baseline_map.get(name)

        if c is None and b is not None:
            diffs.append(TableDiff(name=name, status="missing_current", baseline_hash=_md5_file(b)))
            continue
        if b is None and c is not None:
            diffs.append(TableDiff(name=name, status="missing_baseline", current_hash=_md5_file(c)))
            continue

        assert c is not None and b is not None
        ch, bh = _md5_file(c), _md5_file(b)

        # Row counts (nur als Zusatzinfo)
        try:
            cr = int(len(_read_csv_safe(c)))
        except Exception:
            cr = None
        try:
            br = int(len(_read_csv_safe(b)))
        except Exception:
            br = None

        diffs.append(
            TableDiff(
                name=name,
                status="same" if ch == bh else "changed",
                current_hash=ch,
                baseline_hash=bh,
                current_rows=cr,
                baseline_rows=br,
            )
        )
    return diffs


def print_table_diffs(diffs: List[TableDiff]) -> None:
    _print_section("E) Reports/Tables Diff (gegen Baseline)")

    if not diffs:
        print("Keine Tabellen gefunden (oder Verzeichnisse fehlen).")
        return

    changed = [d for d in diffs if d.status == "changed"]
    missing_b = [d for d in diffs if d.status == "missing_baseline"]
    missing_c = [d for d in diffs if d.status == "missing_current"]
    same = [d for d in diffs if d.status == "same"]

    print(f"same: {len(same)} | changed: {len(changed)} | missing_baseline: {len(missing_b)} | missing_current: {len(missing_c)}")

    if changed:
        print("\nCHANGED:")
        for d in changed:
            print(f"  - {d.name} (rows {d.baseline_rows} -> {d.current_rows})")

    if missing_b:
        print("\nMISSING in Baseline (neu):")
        for d in missing_b:
            print(f"  - {d.name}")

    if missing_c:
        print("\nMISSING in Current (evtl. gelöscht):")
        for d in missing_c:
            print(f"  - {d.name}")


def update_baseline(current_dir: Path, baseline_dir: Path) -> None:
    _print_section("F) Baseline Update")
    if not current_dir.exists():
        print(f"ERROR: Current tables dir fehlt: {current_dir}")
        return

    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Clean baseline
    for p in baseline_dir.glob("*.csv"):
        p.unlink()

    for p in current_dir.glob("*.csv"):
        shutil.copy2(p, baseline_dir / p.name)

    print(f"Baseline aktualisiert: {baseline_dir}")


def check_figures_exist(figures_dir: Path) -> None:
    _print_section("G) Reports/Figures Präsenzcheck")
    _exists_or_warn(figures_dir, "Figures-Verzeichnis")
    if not figures_dir.exists():
        return

    pngs = sorted(figures_dir.glob("*.png"))
    svgs = sorted(figures_dir.glob("*.svg"))
    print(f"PNG: {len(pngs)} | SVG: {len(svgs)}")

    if pngs:
        print("Top PNGs:")
        for p in pngs[:10]:
            print(f"  - {p.name}")


# -------------------------
# CLI
# -------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sanity Checks für MLPerf-Tiny EDA Pipeline.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT, help="Pfad zum Parquet (data/interim/...)")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR_DEFAULT, help="Pfad zu reports/tables")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT, help="Pfad zu reports/figures")

    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR_DEFAULT, help="Pfad zur Baseline (CSV Snapshots)")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Aktualisiert Baseline mit aktuellen reports/tables CSVs (überschreibt).",
    )

    args = parser.parse_args(argv)

    # Data checks
    df = load_parquet(args.parquet)
    check_scope_consistency(df)
    check_round_task_coverage(df)
    check_duplicates(df)
    check_missingness(df)

    # Table diff checks
    diffs = diff_tables(args.tables_dir, args.baseline_dir)
    print_table_diffs(diffs)

    # Figure checks (nur Präsenz)
    check_figures_exist(args.figures_dir)

    if args.update_baseline:
        update_baseline(args.tables_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
