from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


# -----------------------------
# Paths / Defaults
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR_DEFAULT = PROJECT_ROOT / "data" / "raw"
OUT_PATH_DEFAULT = PROJECT_ROOT / "data" / "interim" / "mlperf_tiny_raw.parquet"


# -----------------------------
# Helpers
# -----------------------------
def _detect_csv_format(first_line: str) -> tuple[str, str]:
    """
    Heuristik für Separator und Dezimalzeichen.
    - MLCommons-Exporte sind typischerweise ','-separiert mit '.' als Dezimal.
    - Manche Exporte (z.B. aus Excel) sind ';'-separiert mit ',' als Dezimal.
    """
    semi = first_line.count(";")
    comma = first_line.count(",")
    if semi > comma:
        return ";", ","
    return ",", "."


def _read_csv_smart(path: Path) -> pd.DataFrame:
    """Robustes Einlesen der MLPerf-CSV-Exports (Encoding/Separator/Decimal)."""
    first_line = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    sep_guess, dec_guess = _detect_csv_format(first_line)

    # engine="python" toleriert gemischte CSVs besser als der default (C engine),
    # ist dafür etwas langsamer – bei den kleinen MLPerf-Exports unkritisch.
    return pd.read_csv(
        path,
        sep=sep_guess,
        decimal=dec_guess,
        encoding="utf-8-sig",
        engine="python",
    )


def _to_num(series: pd.Series) -> pd.Series:
    """Konvertiert Werte robust nach float (kommas/leerstrings)."""
    s = series.astype("string")
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _first_non_null(s: pd.Series):
    """Aggregation: erster nicht-null Wert (für wide-Join je Submission×Model)."""
    for v in s:
        if pd.notna(v):
            return v
    return pd.NA


def _infer_task_from_model(model: str) -> Optional[str]:
    """Task-Ableitung aus Model MLC (robuste Heuristik)."""
    if model is None or pd.isna(model):
        return None
    m = str(model).lower()
    if any(k in m for k in ["dscnn", "ds-cnn", "ds cnn", "1d ds-cnn"]):
        return "KWS"
    if "mobilenet" in m:
        return "VWW"
    if "resnet" in m:
        return "IC"
    if "autoencoder" in m:
        return "AD"
    return None


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Normalization
# -----------------------------
def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisiert MLPerf-Tiny CSV-Exports auf ein einheitliches (long) Schema.

    Erwartete Spalten (typisch MLCommons Export):
      - Units
      - Avg. Result (oder Varianten)
      - Model MLC (oder Varianten)
      - Public ID, Organization, System Name, Processor, Accelerator, Software, ...

    Output (long): pro Originalzeile je nach Units genau eine Metrikspalte befüllt:
      latency_us, energy_uj, power_mw, accuracy, auc
      + Metadaten (public_id, organization, processor, accelerator, software, model_mlc, ...)
    """
    df = df.copy()

    # Column mapping (robust gegen Varianten)
    units_col = _pick_col(df, ["Units", "Unit", "units"])
    res_col = _pick_col(df, ["Avg. Result", "Average Result", "Avg Result", "Result", "Value"])
    model_col = _pick_col(df, ["Model MLC", "Model", "Model MLC (Closed)", "Model Name"])
    pid_col = _pick_col(df, ["Public ID", "public_id", "Submission ID", "Submission"])
    org_col = _pick_col(df, ["Organization", "Org"])
    proc_col = _pick_col(df, ["Processor", "Processor Name", "Processor (Click + for details)"])
    acc_col = _pick_col(df, ["Accelerator", "Accelerator Name", "Accelerator (Click + for details)"])
    sw_col = _pick_col(df, ["Software", "SW", "Runtime"])
    bench_col = _pick_col(df, ["Benchmark", "Suite"])
    avail_col = _pick_col(df, ["Availability", "Available?"])
    div_col = _pick_col(df, ["Division", "division"])
    sys_col = _pick_col(df, ["System Name (Click + for details)", "System Name", "System"])
    board_col = _pick_col(df, ["Board Name", "Board"])
    freq_col = _pick_col(df, ["Host Processor Frequency", "Host Frequency", "Host Clock"])


    out = pd.DataFrame(index=df.index)

    # Metadaten (String dtype -> NA bleibt NA, kein "nan"-String)
    if pid_col:
        out["public_id"] = df[pid_col].astype("string").str.strip()
    if org_col:
        out["organization"] = df[org_col].astype("string").str.strip()
    if proc_col:
        out["processor"] = df[proc_col].astype("string").str.strip()
    if acc_col:
        out["accelerator"] = df[acc_col].astype("string").str.strip()
    if sw_col:
        out["software"] = df[sw_col].astype("string").str.strip()
    if bench_col:
        out["benchmark"] = df[bench_col].astype("string").str.strip()
    if model_col:
        out["model_mlc"] = df[model_col].astype("string").str.strip()
    if div_col:
        out["division"] = df[div_col].astype("string").str.strip()
    if avail_col:
        out["availability"] = df[avail_col].astype("string").str.strip()
    if sys_col:
        out["system_name"] = df[sys_col].astype("string").str.strip()
    if board_col:
        out["board_name"] = df[board_col].astype("string").str.strip()
    if freq_col:
        out["host_processor_frequency"] = df[freq_col].astype("string").str.strip()

    # Numeric base
    res_vals = _to_num(df[res_col]) if res_col else pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")

    # Metric columns
    out["latency_us"] = pd.NA
    out["energy_uj"] = pd.NA
    out["power_mw"] = pd.NA
    out["accuracy"] = pd.NA
    out["auc"] = pd.NA

    if units_col:
        units = df[units_col].astype("string").str.strip().str.lower()

        # Latency -> µs (ms -> *1000; s -> *1e6)
        is_lat = units.str.contains("latency")
        ms = is_lat & units.str.contains("ms")
        us = is_lat & (units.str.contains("us") | units.str.contains("µs"))
        sec = is_lat & (
            units.str.fullmatch(r".*\(s\).*") | units.str.endswith(" s") | units.str.endswith("(s)")
        ) & ~ms & ~us

        out.loc[ms, "latency_us"] = res_vals.loc[ms] * 1000.0
        out.loc[us, "latency_us"] = res_vals.loc[us]
        out.loc[sec, "latency_us"] = res_vals.loc[sec] * 1_000_000.0

        # Energy -> µJ
        is_en = units.str.contains("energy")
        uj = is_en & (units.str.contains("uj") | units.str.contains("µj"))
        mj = is_en & units.str.contains("mj")
        nj = is_en & units.str.contains("nj")

        out.loc[uj, "energy_uj"] = res_vals.loc[uj]
        out.loc[mj, "energy_uj"] = res_vals.loc[mj] * 1000.0
        out.loc[nj, "energy_uj"] = res_vals.loc[nj] / 1000.0

        # Power -> mW
        # In den MLPerf-Tiny Exports taucht Power teils nur als "mW" ohne "power" im Units-String auf.
        is_pw = units.str.contains("power") | units.eq("mw") | units.eq("w") | units.eq("uw")
        mw = is_pw & units.eq("mw")
        w = is_pw & units.eq("w")
        uw = is_pw & units.eq("uw")

        out.loc[mw, "power_mw"] = res_vals.loc[mw]
        out.loc[w, "power_mw"] = res_vals.loc[w] * 1000.0
        out.loc[uw, "power_mw"] = res_vals.loc[uw] / 1000.0

        # Quality
        is_acc = units.str.fullmatch("accuracy") | units.str.contains("top-1")
        is_auc = units.str.fullmatch("auc")

        out.loc[is_acc, "accuracy"] = res_vals.loc[is_acc]
        out.loc[is_auc, "auc"] = res_vals.loc[is_auc]
        # AD nutzt AUC als Qualitätsmetrik -> zusätzlich in accuracy spiegeln
        out.loc[is_auc, "accuracy"] = res_vals.loc[is_auc]

    # Task ableiten
    if "model_mlc" in out.columns:
        out["task"] = out["model_mlc"].map(_infer_task_from_model)
    else:
        out["task"] = pd.NA

    return out


# -----------------------------
# Validation / Reporting
# -----------------------------
def _validation_report(
    df_wide: pd.DataFrame,
    *,
    primary_key: list[str],
    coarse_key: Optional[list[str]] = None,
) -> None:
    """Kompakter Validitätsreport für reproduzierbare Runs."""
    metric_cols = [c for c in ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"] if c in df_wide.columns]

    print("\n--- VALIDATION REPORT ---")

    # Primary key duplicates (must be 0)
    pk_dups = int(df_wide.duplicated(subset=primary_key).sum()) if primary_key else -1
    print(f"Duplikate auf PRIMARY KEY {primary_key}: {pk_dups}")

    # Coarse key duplicates (expected >0 if system variants exist)
    if coarse_key:
        ck_dups = int(df_wide.duplicated(subset=coarse_key).sum())
        print(
            f"INFO: Duplikate auf GROBEM KEY {coarse_key}: {ck_dups} "
            "(würde bei zu grobem Grouping zusammenfallen)"
        )

    # Coverage Round×Task (count rows, not a metric column)
    if "round" in df_wide.columns and "task" in df_wide.columns:
        cov = df_wide.assign(task=df_wide["task"].fillna("UNKNOWN")).groupby(["round", "task"]).size().unstack(fill_value=0)
        print("\nCoverage (Round×Task) – Anzahl Zeilen:")
        print(cov)

        unk = int((df_wide["task"].isna() | (df_wide["task"] == "UNKNOWN")).sum())
        if unk:
            print(f"WARN: {unk} Zeilen mit unbekanntem Task (task = UNKNOWN/NA).")
    else:
        print("WARN: Coverage-Check übersprungen (round/task fehlen).")

    # Missingness
    if metric_cols:
        miss = df_wide[metric_cols].isna().mean().sort_values(ascending=False)
        print("\nMissingness (Anteil NA pro Metrik):")
        print(miss)

    # Plausibility (broad flags)
    def _range(col: str):
        s = pd.to_numeric(df_wide[col], errors="coerce")
        return float(s.min()), float(s.median()), float(s.max())

    if "latency_us" in df_wide.columns:
        mn, md, mx = _range("latency_us")
        print(f"\nlatency_us: min={mn:.3g}, median={md:.3g}, max={mx:.3g}")
        if mn <= 0 or mx > 1e8:
            print("WARN: latency_us außerhalb plausibler Grobgrenzen (<=0 oder >1e8 µs).")

    if "energy_uj" in df_wide.columns:
        s = pd.to_numeric(df_wide["energy_uj"], errors="coerce")
        if s.notna().any():
            mn, md, mx = float(s.min()), float(s.median()), float(s.max())
            print(f"energy_uj: min={mn:.3g}, median={md:.3g}, max={mx:.3g}")
            if mn <= 0 or mx > 1e9:
                print("WARN: energy_uj außerhalb plausibler Grobgrenzen (<=0 oder >1e9 µJ).")

    if "power_mw" in df_wide.columns:
        s = pd.to_numeric(df_wide["power_mw"], errors="coerce")
        if s.notna().any():
            mn, md, mx = float(s.min()), float(s.median()), float(s.max())
            print(f"power_mw: min={mn:.3g}, median={md:.3g}, max={mx:.3g}")
            if mn <= 0 or mx > 1e6:
                print("WARN: power_mw außerhalb plausibler Grobgrenzen (<=0 oder >1e6 mW).")

    print("--- END REPORT ---\n")

def _write_unknown_reports(df_wide: pd.DataFrame, docs_dir: Path) -> None:
    """
    Schreibt zwei CSVs für die Dokumentation der UNKNOWN/NULL-Fälle:
      - validation_unknown_rows.csv: Zeilenliste zur händischen Prüfung in Excel
      - validation_unknown_summary_by_round.csv: Round-level Summary
    """
    docs_dir.mkdir(parents=True, exist_ok=True)

    # UNKNOWN-Definition: task fehlt/UNKNOWN ODER model_mlc fehlt (NULL-Spalte in Pivot)
    task_col_exists = "task" in df_wide.columns
    model_col_exists = "model_mlc" in df_wide.columns

    is_unknown = pd.Series(False, index=df_wide.index)
    if task_col_exists:
        is_unknown = is_unknown | df_wide["task"].isna() | (df_wide["task"] == "UNKNOWN")
    if model_col_exists:
        is_unknown = is_unknown | df_wide["model_mlc"].isna()

    unknown_df = df_wide[is_unknown].copy()

    # Zeilenliste (Excel-freundlich)
    cols_preferred = [
        "round",
        "public_id",
        "organization",
        "system_name",
        "host_processor_frequency",
        "processor",
        "accelerator",
        "software",
        "benchmark",
        "model_mlc",
        "task",
        "latency_us",
        "energy_uj",
        "power_mw",
        "accuracy",
        "auc",
    ]
    cols = [c for c in cols_preferred if c in unknown_df.columns]

    out_rows = docs_dir / "validation_unknown_rows.csv"
    unknown_df[cols].sort_values([c for c in ["round", "public_id", "system_name", "model_mlc"] if c in cols]) \
        .to_csv(out_rows, index=False, encoding="utf-8-sig")

    # Summary pro Round
    if "round" in df_wide.columns:
        total_by_round = df_wide.groupby("round").size().rename("rows_total")
        unknown_by_round = unknown_df.groupby("round").size().rename("rows_unknown")
        summary = pd.concat([total_by_round, unknown_by_round], axis=1).fillna(0).astype(int)
        summary["share_unknown"] = (summary["rows_unknown"] / summary["rows_total"]).round(4)

        out_summary = docs_dir / "validation_unknown_summary_by_round.csv"
        summary.reset_index().sort_values("round").to_csv(out_summary, index=False, encoding="utf-8-sig")

        print(f"UNKNOWN-Reports geschrieben: {out_rows} | {out_summary}")
    else:
        print(f"UNKNOWN-Report geschrieben: {out_rows} (kein Round-Feld für Summary gefunden)")



# -----------------------------
# Pipeline
# -----------------------------
def load_and_clean_csvs(
    raw_dir: Path = RAW_DIR_DEFAULT,
    out_path: Path = OUT_PATH_DEFAULT,
    *,
    write_parquet: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    paths = sorted(raw_dir.glob("raw_v*.csv"))
    if not paths:
        raise FileNotFoundError(f"Keine raw_v*.csv in {raw_dir.resolve()} gefunden.")

    print(f"CSV-Dateien gefunden: {len(paths)}")

    parts: list[pd.DataFrame] = []
    for p in paths:
        df = _read_csv_smart(p)
        round_tag = p.stem.replace("raw_", "")

        norm_long = normalize_metrics(df)
        norm_long["round"] = round_tag
        parts.append(norm_long)

        # Minimaler Smoke-Check je Datei
        present = {
            "rows_in": int(df.shape[0]),
            "rows_norm": int(norm_long.shape[0]),
            "latency_us": int(pd.to_numeric(norm_long["latency_us"], errors="coerce").notna().sum()),
            "energy_uj": int(pd.to_numeric(norm_long["energy_uj"], errors="coerce").notna().sum()),
            "power_mw": int(pd.to_numeric(norm_long["power_mw"], errors="coerce").notna().sum()),
            "accuracy": int(pd.to_numeric(norm_long["accuracy"], errors="coerce").notna().sum()),
            "auc": int(pd.to_numeric(norm_long["auc"], errors="coerce").notna().sum()),
        }
        print(f"  -> {p.name}: {present}")

    long_df = pd.concat(parts, ignore_index=True)

    # Konsolidierung: pro Submission×Model eine Zeile (sonst vermischt man Tasks/Metriken!)
    group_keys = [
        k for k in [
            "public_id",
            "round",
            "system_name",
            "host_processor_frequency",
            "model_mlc",
            ]
            if k in long_df.columns
    ]


    agg_map = {col: _first_non_null for col in long_df.columns if col not in group_keys}
    wide_df = long_df.groupby(group_keys, as_index=False, dropna=False).agg(agg_map)

    # Metrics explizit numerisch machen (groupby+custom agg -> sonst oft object dtype)
    for col in ["latency_us", "energy_uj", "power_mw", "accuracy", "auc"]:
        if col in wide_df.columns:
            wide_df[col] = pd.to_numeric(wide_df[col], errors="coerce")

    # Mindestens Latenz oder Energie erforderlich
    wide_df = wide_df[(wide_df["latency_us"].notna()) | (wide_df["energy_uj"].notna())].copy()

    # Validierungsreport (pflichtig)
    _validation_report(
        wide_df,
        primary_key=group_keys,
        coarse_key=[k for k in ["public_id", "round", "model_mlc"] if k in wide_df.columns],
    )

        # Dokumentation UNKNOWN/NULL (für Excel + Methodik)
    docs_dir = PROJECT_ROOT / "docs"
    _write_unknown_reports(wide_df, docs_dir)


    if write_parquet:
        try:
            wide_df.to_parquet(out_path, index=False)
        except Exception as e:
            raise RuntimeError(
                "Parquet-Export fehlgeschlagen. Installiere eine Parquet-Engine, z.B.: pip install pyarrow"
            ) from e

        print(f"Gespeichert: {out_path} ({len(wide_df)} Zeilen)")

    return wide_df


if __name__ == "__main__":
    load_and_clean_csvs()
