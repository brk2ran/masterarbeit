from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Wichtig: nur für Features/Scope-Flags; hält den Scope zentral in src/features.py
from src.features import add_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DOCS_DIR = PROJECT_ROOT / "docs"

PARQUET_OUT = INTERIM_DIR / "mlperf_tiny_raw.parquet"

ROUND_FILE_PATTERN = re.compile(r"raw_(v\d+\.\d+)\.csv$", re.IGNORECASE)


# -----------------------------
# Parsing / Normalisierung
# -----------------------------
def _to_float(x: object) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    # Dezimalkomma robust
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_metric(units: object) -> Tuple[str | None, str | None]:
    if units is None or (isinstance(units, float) and pd.isna(units)):
        return None, None
    u = str(units).strip().lower().replace("µ", "u")

    # Latency
    if ("latency" in u and "ms" in u) or u in {"ms", "latency (ms)"}:
        return "latency_us", "ms"
    if ("latency" in u and "us" in u) or u in {"us", "latency (us)"}:
        return "latency_us", "us"

    # Energy / Power
    if ("energy" in u and "uj" in u) or u in {"uj", "energy (uj)", "energy (µj)"}:
        return "energy_uj", "uj"
    if ("power" in u and "mw" in u) or u in {"mw", "power (mw)"}:
        return "power_mw", "mw"

    # Quality
    if "accuracy" in u:
        return "accuracy", None
    if "auc" in u:
        return "auc", None

    return None, None


def _convert_value(metric: str | None, unit: str | None, value: object) -> float:
    v = _to_float(value)
    if pd.isna(v) or metric is None:
        return float("nan")

    if metric == "latency_us":
        if unit == "ms":
            return v * 1000.0
        return v

    return v


def _detect_round_from_filename(path: Path) -> str:
    m = ROUND_FILE_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Dateiname passt nicht zum erwarteten Muster raw_vX.Y.csv: {path.name}")
    return m.group(1)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Public ID": "public_id",
        "Organization": "organization",
        "Availability": "availability",
        "System Name (Click + for details)": "system_name",
        "Processor": "processor",
        "Accelerator": "accelerator",
        "Model MLC": "model_mlc",
        "Units": "units",
        "Avg. Result": "avg_result",
        "Software": "software",
        "Host Processor Frequency": "host_processor_frequency",
        "Benchmark": "benchmark",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()

    # sicherstellen, dass alle erwarteten Felder existieren
    for col in rename.values():
        if col not in out.columns:
            out[col] = pd.NA

    return out


def load_raw_csvs() -> List[Tuple[str, pd.DataFrame]]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR nicht gefunden: {RAW_DIR}")

    files = sorted([p for p in RAW_DIR.glob("raw_v*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"Keine raw_v*.csv in {RAW_DIR} gefunden.")

    out: List[Tuple[str, pd.DataFrame]] = []
    for p in files:
        rnd = _detect_round_from_filename(p)
        df = pd.read_csv(p)
        df = _standardize_columns(df)
        df["round"] = rnd
        out.append((p.name, df))

    return out


def normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-Format: pro Zeile eine Metrik (Units + Avg. Result).
    Wir mappen auf metric + value_num und machen danach Wide-Pivot.
    """
    d = df.copy()

    # Model MLC: NA oder leer/Whitespace -> NULL (entspricht "Null"-Spalte in MLCommons Tabellen)
    d["model_mlc"] = d["model_mlc"].astype("string").str.strip()
    d["model_mlc"] = d["model_mlc"].where(d["model_mlc"].notna() & (d["model_mlc"] != ""), pd.NA)
    d["model_mlc"] = d["model_mlc"].fillna("NULL")

    parsed = d["units"].apply(_parse_metric)
    d["metric"] = parsed.apply(lambda t: t[0])
    d["metric_unit"] = parsed.apply(lambda t: t[1])
    d["value_num"] = [_convert_value(m, u, v) for m, u, v in zip(d["metric"], d["metric_unit"], d["avg_result"])]

    return d


def pivot_to_wide(d_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot auf eine Zeile pro (round, public_id, system_name, model_mlc, ...).
    Aggregation bei Duplikaten: median (robust).
    """
    d = d_long.copy()

    # groupby mit dropna=False ist stabiler, wenn Spalten fehlen
    for c in ["system_name", "organization", "processor", "accelerator", "host_processor_frequency", "benchmark", "availability", "software"]:
        d[c] = d[c].fillna("")

    key_cols = [
        "round",
        "public_id",
        "system_name",
        "organization",
        "processor",
        "accelerator",
        "host_processor_frequency",
        "software",
        "availability",
        "benchmark",
        "model_mlc",
    ]

    g = d.groupby(key_cols + ["metric"], dropna=False)["value_num"].median()
    wide = g.unstack("metric").reset_index()

    # Task-Spalte kann (je nach Export) fehlen. Für Konsistenz legen wir sie an.
    if "task" not in wide.columns:
        wide["task"] = pd.NA

    return wide


def _print_file_stats(filename: str, df_wide: pd.DataFrame) -> None:
    metrics = ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"]
    stats = {"rows_in": int(df_wide.shape[0]), "rows_norm": int(df_wide.shape[0])}
    for m in metrics:
        stats[m] = int(df_wide[m].notna().sum()) if m in df_wide.columns else 0
    print(f"  -> {filename}: {stats}")


def run() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_csvs()
    print(f"CSV-Dateien gefunden: {len(raw)}")

    wide_parts: List[pd.DataFrame] = []

    for filename, df in raw:
        d_long = normalize_long(df)
        d_wide = pivot_to_wide(d_long)

        # Features/Scope-Spalten hinzufügen (inkl. out_of_scope / in_scope)
        d_wide = add_features(d_wide)

        _print_file_stats(filename, d_wide)
        wide_parts.append(d_wide)

    out = pd.concat(wide_parts, ignore_index=True)

    # Speichern
    out.to_parquet(PARQUET_OUT, index=False)
    print(f"Gespeichert: {PARQUET_OUT}")


if __name__ == "__main__":
    run()
