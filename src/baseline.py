from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "baseline"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Scope-Decision (wie besprochen)
IN_SCOPE_TASKS = {"AD", "IC", "KWS", "VWW"}
EXCLUDE_MODEL_MLC_CANON = {"1D_DS_CNN"}


README_TEMPLATE = """# Baseline (EDA-Pipeline)

Diese Baseline definiert den Referenzzustand der Datenpipeline und der daraus erzeugten EDA-Artefakte.
Ziel: Änderungen an Preprocessing/Features/EDA/Plots sollen nachvollziehbar und prüfbar sein (Reproduzierbarkeit).

## Pipeline-Schritte (Baseline-Run)
1) Preprocessing:
   - python -m src.data_preprocessing
   - Output: data/interim/mlperf_tiny_raw.parquet

2) EDA-Tabellen:
   - python -m src.eda
   - Output: reports/tables/*.csv

3) Plots:
   - python -m src.plots --all --logy --svg
   - Output: reports/figures/*.(png|svg)

4) Baseline-Freeze:
   - python -m src.baseline --freeze
   - Output: docs/baseline/*

## Scope-Entscheidung (für Vergleichbarkeit)
- IN_SCOPE: AD, IC, KWS, VWW
- OUT_OF_SCOPE: 1D_DS_CNN (nur in v1.3, Streaming Wakeword) wird ausgeschlossen
- UNKNOWN: Zeilen, bei denen task/model nicht eindeutig ableitbar sind

Begründung: Für die Trendanalyse über v0.5–v1.3 soll eine konsistente Aufgaben-/Modellbasis verwendet werden.
1D_DS_CNN ist ein zusätzliches Benchmark-Element (Streaming Wakeword) und nicht über alle Runden vergleichbar.

## Was gilt als "Baseline-stabil"
- Rohdaten (data/raw/*.csv) müssen unverändert sein (raw_hashes.csv)
- Parquet-Metadaten (Zeilen/Spalten/Spaltennamen) müssen plausibel stabil sein (parquet_meta.json)
- Tabellen/Plots müssen vorhanden sein (expected_tables.txt, expected_figures.txt)
- Kennzahlen (Scope-Anteile, Missingness, Quantile, UNKNOWN-Anteil) dürfen sich nur ändern,
  wenn es einen bewussten Code-/Daten-Change gab.
"""


@dataclass(frozen=True)
class Paths:
    out_dir: Path
    raw_dir: Path
    parquet_path: Path
    tables_dir: Path
    figures_dir: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv_rows(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _safe_version(mod, attr: str) -> str:
    try:
        return str(getattr(mod, attr))
    except Exception:
        return ""


def _import_add_features():
    """
    Wir nutzen add_features() aus src.features, um task_canon/model_mlc_canon konsistent zu bekommen.
    Wichtig: Dieses Skript sollte als Modul ausgeführt werden:
      python -m src.baseline
    """
    try:
        from src.features import add_features  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Konnte 'src' nicht importieren. Bitte aus dem Projekt-Root als Modul ausführen:\n"
            "  python -m src.baseline\n"
        ) from e
    return add_features


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet nicht gefunden: {path}")
    return pd.read_parquet(path)


def _scope_label(df: pd.DataFrame) -> pd.Series:
    """
    Liefert pro Zeile: IN_SCOPE / OUT_OF_SCOPE / UNKNOWN
    Regeln:
      - UNKNOWN wenn task_canon == 'UNKNOWN'
      - OUT_OF_SCOPE wenn model_mlc_canon in EXCLUDE_MODEL_MLC_CANON oder task_canon nicht in IN_SCOPE_TASKS
      - sonst IN_SCOPE
    """
    if "task_canon" not in df.columns or "model_mlc_canon" not in df.columns:
        raise ValueError("Benötigte Spalten fehlen: 'task_canon' und/oder 'model_mlc_canon'.")

    task = df["task_canon"].astype("string")
    model = df["model_mlc_canon"].astype("string")

    is_unknown = task.isna() | (task.str.upper() == "UNKNOWN")
    is_out = model.isin(list(EXCLUDE_MODEL_MLC_CANON)) | (~task.isin(list(IN_SCOPE_TASKS)))

    out = pd.Series(["IN_SCOPE"] * len(df), index=df.index, dtype="string")
    out[is_out] = "OUT_OF_SCOPE"
    out[is_unknown] = "UNKNOWN"
    return out


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"min": None, "median": None, "max": None}
    return {"min": float(s.min()), "median": float(s.median()), "max": float(s.max())}


def freeze_baseline(paths: Paths, *, baseline_tag: str) -> None:
    _ensure_dir(paths.out_dir)

    # 1) README (nur anlegen, wenn nicht vorhanden)
    readme_path = paths.out_dir / "README.md"
    if not readme_path.exists():
        _write_text(readme_path, README_TEMPLATE)

    # 2) raw_hashes.csv
    raw_files = sorted(paths.raw_dir.glob("*.csv"))
    raw_hash_rows = []
    for f in raw_files:
        raw_hash_rows.append([str(f.as_posix()), _sha256_file(f)])
    _write_csv_rows(paths.out_dir / "raw_hashes.csv", ["file", "sha256"], raw_hash_rows)

    # 3) parquet_meta.json
    df = _load_parquet(paths.parquet_path)
    parquet_meta = {
        "parquet_path": str(paths.parquet_path.as_posix()),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
    }
    _write_json(paths.out_dir / "parquet_meta.json", parquet_meta)

    # 4) Features + Scope-Snapshots
    add_features = _import_add_features()
    d = add_features(df.copy())

    scope = _scope_label(d)
    d["scope"] = scope

    rows_total = int(len(d))
    rows_in = int((scope == "IN_SCOPE").sum())
    rows_out = int((scope == "OUT_OF_SCOPE").sum())
    rows_unk = int((scope == "UNKNOWN").sum())

    def _share(n: int) -> float:
        return round(n / rows_total, 6) if rows_total else 0.0

    scope_rows = [
        ["IN_SCOPE", rows_in, _share(rows_in)],
        ["OUT_OF_SCOPE", rows_out, _share(rows_out)],
        ["UNKNOWN", rows_unk, _share(rows_unk)],
    ]
    _write_csv_rows(paths.out_dir / "scope_snapshot.csv", ["scope", "rows", "share"], scope_rows)

    # 5) Unknown Snapshot by Round (aus DataFrame, nicht aus reports/tables)
    if "round" in d.columns:
        by_round = (
            d.assign(is_unknown=(d["scope"] == "UNKNOWN"))
            .groupby("round", dropna=False)
            .agg(
                rows_total=("scope", "size"),
                rows_unknown=("is_unknown", "sum"),
            )
            .reset_index()
        )
        by_round["share_unknown"] = (by_round["rows_unknown"] / by_round["rows_total"]).round(6)
        by_round = by_round.sort_values("round")
        by_round.to_csv(paths.out_dir / "unknown_snapshot.csv", index=False, encoding="utf-8-sig")
    else:
        (paths.out_dir / "unknown_snapshot.csv").write_text(
            "round,rows_total,rows_unknown,share_unknown\n", encoding="utf-8-sig"
        )

    # 6) Metrics Snapshot (Missingness + Quantile) auf IN_SCOPE
    din = d[d["scope"] == "IN_SCOPE"].copy()

    def _miss_share(col: str) -> float:
        if col not in din.columns:
            return 1.0
        return round(float(din[col].isna().mean()), 6)

    lat_q = _quantiles(din["latency_us"]) if "latency_us" in din.columns else {"min": None, "median": None, "max": None}
    en_q = _quantiles(din["energy_uj"]) if "energy_uj" in din.columns else {"min": None, "median": None, "max": None}

    metrics_rows = [
        ["rows_total", rows_total],
        ["rows_in_scope", rows_in],
        ["rows_out_of_scope", rows_out],
        ["rows_unknown", rows_unk],
        ["missing_latency_us", _miss_share("latency_us")],
        ["missing_energy_uj", _miss_share("energy_uj")],
        ["missing_power_mw", _miss_share("power_mw")],
        ["missing_accuracy", _miss_share("accuracy")],
        ["missing_auc", _miss_share("auc")],
        ["latency_us_min", lat_q["min"]],
        ["latency_us_median", lat_q["median"]],
        ["latency_us_max", lat_q["max"]],
        ["energy_uj_min", en_q["min"]],
        ["energy_uj_median", en_q["median"]],
        ["energy_uj_max", en_q["max"]],
    ]
    _write_csv_rows(paths.out_dir / "metrics_snapshot.csv", ["metric", "value"], metrics_rows)

    # 7) expected_tables.txt + expected_figures.txt (Freeze vom aktuellen Ist-Zustand)
    tables = sorted([p.name for p in paths.tables_dir.glob("*.csv")]) if paths.tables_dir.exists() else []
    figures = []
    if paths.figures_dir.exists():
        figures = sorted([p.name for p in paths.figures_dir.glob("*.png")] + [p.name for p in paths.figures_dir.glob("*.svg")])

    _write_text(paths.out_dir / "expected_tables.txt", "\n".join(tables) + ("\n" if tables else ""))
    _write_text(paths.out_dir / "expected_figures.txt", "\n".join(figures) + ("\n" if figures else ""))

    # 8) baseline_meta.json
    baseline_meta = {
        "baseline_tag": baseline_tag,
        "baseline_commit": "",
        "created_at": _now_iso(),
        "environment": {
            "python_version": sys.version.split()[0],
            "pandas_version": _safe_version(pd, "__version__"),
            "matplotlib_version": _safe_version(matplotlib, "__version__") if matplotlib is not None else "",
        },
        "scope_decisions": {
            "exclude_model_mlc_canon": sorted(EXCLUDE_MODEL_MLC_CANON),
            "in_scope_tasks": sorted(IN_SCOPE_TASKS),
        },
        "artifacts": {
            "parquet_path": str(paths.parquet_path.as_posix()),
            "tables_dir": str(paths.tables_dir.as_posix()),
            "figures_dir": str(paths.figures_dir.as_posix()),
        },
    }
    _write_json(paths.out_dir / "baseline_meta.json", baseline_meta)

    print(f"[baseline] geschrieben: {paths.out_dir}")
    print("[baseline] Dateien:")
    for p in sorted(paths.out_dir.glob("*")):
        if p.is_file():
            print(f"  - {p.relative_to(PROJECT_ROOT)}")


def verify_baseline(paths: Paths) -> int:
    """
    Prüft Baseline gegen den aktuellen Zustand:
      - raw_hashes.csv stimmt?
      - parquet_meta.json passt (rows/cols/columns)?
      - expected_tables/figures vorhanden?
      - scope/metrics/unknown snapshot neu berechnen und gegen gespeicherte Snapshots vergleichen (soft check)
    Rückgabe: 0 OK / 1 WARN/FAIL
    """
    out = paths.out_dir
    rc = 0

    def warn(msg: str) -> None:
        nonlocal rc
        rc = 1
        print(f"WARN: {msg}")

    # A) raw_hashes.csv
    hashes_path = out / "raw_hashes.csv"
    if not hashes_path.exists():
        warn("raw_hashes.csv fehlt (Baseline unvollständig).")
    else:
        saved = pd.read_csv(hashes_path)
        current = []
        for f in sorted(paths.raw_dir.glob("*.csv")):
            current.append({"file": str(f.as_posix()), "sha256": _sha256_file(f)})
        cur = pd.DataFrame(current)
        merged = saved.merge(cur, on="file", how="outer", suffixes=("_saved", "_cur"), indicator=True)
        changed = merged[(merged["_merge"] != "both") | (merged["sha256_saved"] != merged["sha256_cur"])]
        if not changed.empty:
            warn(f"Rohdaten-Hashes weichen ab: {len(changed)} Datei(en).")

    # B) parquet_meta.json
    pm_path = out / "parquet_meta.json"
    if not pm_path.exists():
        warn("parquet_meta.json fehlt (Baseline unvollständig).")
    else:
        pm = json.loads(pm_path.read_text(encoding="utf-8"))
        df = _load_parquet(paths.parquet_path)
        if int(pm.get("rows", -1)) != int(len(df)):
            warn(f"Parquet rows abweichend: baseline={pm.get('rows')} aktuell={len(df)}")
        if int(pm.get("cols", -1)) != int(len(df.columns)):
            warn(f"Parquet cols abweichend: baseline={pm.get('cols')} aktuell={len(df.columns)}")
        base_cols = list(map(str, pm.get("columns", [])))
        cur_cols = list(map(str, df.columns))
        if base_cols != cur_cols:
            warn("Parquet Spaltenliste abweichend (Reihenfolge/Name geändert).")

    # C) expected_tables/figures presence
    et_path = out / "expected_tables.txt"
    if et_path.exists() and paths.tables_dir.exists():
        expected = [l.strip() for l in et_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        missing = [x for x in expected if not (paths.tables_dir / x).exists()]
        if missing:
            warn(f"Fehlende Tabellen: {missing}")
    else:
        warn("expected_tables.txt oder reports/tables fehlt.")

    ef_path = out / "expected_figures.txt"
    if ef_path.exists() and paths.figures_dir.exists():
        expected = [l.strip() for l in ef_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        missing = [x for x in expected if not (paths.figures_dir / x).exists()]
        if missing:
            warn(f"Fehlende Figuren: {missing}")
    else:
        warn("expected_figures.txt oder reports/figures fehlt.")

    # D) Soft check: Snapshots neu berechnen und grob vergleichen
    # (bewusst "soft": falls du bewusst Code änderst, soll es als Warnung sichtbar sein)
    try:
        add_features = _import_add_features()
        df = _load_parquet(paths.parquet_path)
        d = add_features(df.copy())
        d["scope"] = _scope_label(d)

        # scope_snapshot.csv Vergleich
        ss_path = out / "scope_snapshot.csv"
        if ss_path.exists():
            saved = pd.read_csv(ss_path)
            cur = pd.DataFrame(
                {
                    "scope": ["IN_SCOPE", "OUT_OF_SCOPE", "UNKNOWN"],
                    "rows": [
                        int((d["scope"] == "IN_SCOPE").sum()),
                        int((d["scope"] == "OUT_OF_SCOPE").sum()),
                        int((d["scope"] == "UNKNOWN").sum()),
                    ],
                }
            )
            m = saved.merge(cur, on="scope", how="outer", suffixes=("_saved", "_cur"))
            diff = m[m["rows_saved"] != m["rows_cur"]]
            if not diff.empty:
                warn(f"Scope-Snapshot abweichend: {diff[['scope','rows_saved','rows_cur']].to_dict(orient='records')}")

    except Exception as e:
        warn(f"Snapshot-Softcheck konnte nicht ausgeführt werden: {e}")

    if rc == 0:
        print("OK: Baseline-Verify ohne Abweichungen.")
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Baseline freeze/verify für MLPerf-Tiny EDA Pipeline.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Baseline output directory (docs/baseline)")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw CSV directory (data/raw)")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET, help="Parquet path (data/interim/*.parquet)")
    p.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Tables directory (reports/tables)")
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Figures directory (reports/figures)")
    p.add_argument("--tag", type=str, default="baseline-eda-01", help="Baseline tag name (stored in baseline_meta.json)")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Freeze current state into docs/baseline/*")
    mode.add_argument("--verify", action="store_true", help="Verify current state against docs/baseline/*")

    args = p.parse_args(argv)

    paths = Paths(
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        parquet_path=args.parquet,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )

    if args.freeze:
        freeze_baseline(paths, baseline_tag=args.tag)
        return 0
    if args.verify:
        return verify_baseline(paths)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
