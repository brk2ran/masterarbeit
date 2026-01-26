import re
from typing import Dict

import numpy as np
import pandas as pd

TASKS_IN_SCOPE = ["AD", "IC", "KWS", "VWW"]
TASKS_ALL_ORDER = ["AD", "IC", "KWS", "VWW", "UNKNOWN", "OUT_OF_SCOPE"]

# Benchmark darf nur als Ersatz-Model dienen, wenn es wie ein Modellstring aussieht.
# In deinen Daten ist benchmark i.d.R. "Tiny" -> greift damit NICHT.
MODEL_LIKE_PATTERN = re.compile(
    r"DS\s*-?\s*CNN|1D\s*-?\s*DS\s*-?\s*CNN|RESNET|MOBILENET|AUTOENCODER|FC\s+AUTOENCODER",
    re.IGNORECASE,
)


def _canon_task_from_model(model_mlc_canon: str) -> str:
    if model_mlc_canon == "DS_CNN":
        return "KWS"
    if model_mlc_canon == "RESNET":
        return "IC"
    if model_mlc_canon == "MBNET":
        return "VWW"
    if model_mlc_canon == "FC_AE":
        return "AD"
    return "UNKNOWN"


def _canon_task_from_task_col(raw: object) -> str:
    """Mappt eine Task-Spalte (falls vorhanden) robust auf AD/IC/KWS/VWW/UNKNOWN."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "UNKNOWN"
    s = str(raw).strip().upper()
    task_map = {
        "AD": "AD",
        "ANOMALY": "AD",
        "ANOMALY DETECTION": "AD",
        "IC": "IC",
        "IMAGE": "IC",
        "IMAGE CLASSIFICATION": "IC",
        "KWS": "KWS",
        "KEYWORD": "KWS",
        "KEYWORD SPOTTING": "KWS",
        "VWW": "VWW",
        "VISUAL WAKE WORDS": "VWW",
    }
    return task_map.get(s, "UNKNOWN")


def _canon_model(raw: object) -> str:
    """
    Robust kanonisieren:
    - wenige direkte Mappings
    - ansonsten token/regex-basiert
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "UNKNOWN"
    s = str(raw).strip()
    if not s:
        return "UNKNOWN"

    u = s.upper().replace("_", " ").strip()
    u = re.sub(r"\s+", " ", u)

    direct = {
        "DSCNN": "DS_CNN",
        "DS-CNN": "DS_CNN",
        "DS CNN": "DS_CNN",
        "FC AUTOENCODER": "FC_AE",
        "AUTOENCODER": "FC_AE",
        "FC AE": "FC_AE",
        "MOBILENET": "MBNET",
        "MOBILENETV1": "MBNET",
        "MOBILENET-V1": "MBNET",
        "MOBILENET V1": "MBNET",
        "RESNET": "RESNET",
        "RESNET-V1": "RESNET",
        "RESNET V1": "RESNET",
        "RESNET_V1": "RESNET",
        "1D-DS-CNN": "1D_DS_CNN",
        "1D DS-CNN": "1D_DS_CNN",
        "1D DS CNN": "1D_DS_CNN",
        "1D DSCNN": "1D_DS_CNN",
    }
    if u in direct:
        return direct[u]

    # Heuristik für neue Varianten
    if re.search(r"\b1D\b", u) and re.search(r"\bDS\b", u) and re.search(r"\bCNN\b", u):
        return "1D_DS_CNN"
    if re.search(r"\bDS\b", u) and re.search(r"\bCNN\b", u):
        return "DS_CNN"
    if "AUTO" in u and "ENC" in u:
        return "FC_AE"
    if "MOBILE" in u and "NET" in u:
        return "MBNET"
    if "RES" in u and "NET" in u:
        return "RESNET"

    return "UNKNOWN"


# -----------------------------
# FF1b / FF1c: minimal robuste Heuristiken
# -----------------------------
_WS_RE = re.compile(r"\s+")
_FREQ_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(GHZ|MHZ)?", re.IGNORECASE)


def _norm_str(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return _WS_RE.sub(" ", str(x).strip())


def _infer_processor_family(processor_raw: object) -> str:
    s = _norm_str(processor_raw).upper()
    if not s:
        return "UNKNOWN"
    if "CORTEX-M" in s or "CORTEX M" in s:
        return "Cortex-M"
    if "CORTEX-A" in s or "CORTEX A" in s:
        return "Cortex-A"
    if "RISC-V" in s or "RISCV" in s or "RISC V" in s:
        return "RISC-V"
    if "XTENSA" in s:
        return "Xtensa"
    return "OTHER"


def _parse_freq_mhz(freq_raw: object) -> float:
    s = _norm_str(freq_raw)
    if not s:
        return float("nan")
    m = _FREQ_RE.search(s)
    if not m:
        return float("nan")
    val = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "").upper()
    if unit == "GHZ":
        return val * 1000.0
    return val  # default MHz


def _bucket_freq_mhz(freq_mhz: float) -> str:
    if pd.isna(freq_mhz):
        return "UNKNOWN"
    if freq_mhz < 100:
        return "<100"
    if freq_mhz < 200:
        return "100-199"
    if freq_mhz < 400:
        return "200-399"
    return ">=400"


def _infer_software_flags(software_raw: object) -> Dict[str, bool]:
    s = _norm_str(software_raw).upper()
    return {
        "sw_cmsis_nn": ("CMSIS" in s),
        "sw_tflm": ("TFLM" in s) or ("TENSORFLOW LITE MICRO" in s) or ("TFLITE MICRO" in s),
        "sw_tvm": ("TVM" in s) or ("MICROTVM" in s),
        "sw_onnx": ("ONNX" in s),
    }


def _infer_software_stack(software_raw: object) -> str:
    flags = _infer_software_flags(software_raw)
    if flags["sw_cmsis_nn"]:
        return "CMSIS"
    if flags["sw_tflm"]:
        return "TFLM"
    if flags["sw_tvm"]:
        return "TVM"
    if flags["sw_onnx"]:
        return "ONNX"
    if _norm_str(software_raw) == "":
        return "UNKNOWN"
    return "OTHER"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt kanonische Task-/Model-/Scope-Spalten.
    OUT_OF_SCOPE (1D-DS-CNN) wird explizit überschrieben.

    Erweiterung (FF1b/FF1c):
      - accelerator_present
      - processor_family
      - cpu_freq_mhz + cpu_freq_bucket
      - software_stack + sw_* Flags
    """
    out = df.copy()

    if "model_mlc" not in out.columns:
        out["model_mlc"] = pd.NA

    model_raw = out["model_mlc"].astype("string").str.strip()
    model_valid = model_raw.notna() & (model_raw != "") & (model_raw.str.upper() != "NULL")

    # Benchmark nur als Model-Ersatz zulassen, wenn es wie ein Modellstring aussieht.
    if "benchmark" in out.columns:
        bench_raw = out["benchmark"].astype("string").str.strip()
        bench_valid = bench_raw.notna() & (bench_raw != "") & bench_raw.str.contains(MODEL_LIKE_PATTERN, regex=True)
    else:
        bench_raw = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
        bench_valid = pd.Series([False] * len(out), index=out.index, dtype="bool")

    model_effective = model_raw.where(model_valid, bench_raw.where(bench_valid, pd.NA))
    out["model_mlc_effective"] = model_effective
    out["model_mlc_source"] = np.where(model_valid, "model_mlc", np.where(bench_valid, "benchmark", "missing"))

    out["model_mlc_canon"] = out["model_mlc_effective"].apply(_canon_model)

    if "task" in out.columns:
        task_from_col = out["task"].apply(_canon_task_from_task_col)
        out["task_canon"] = np.where(
            task_from_col != "UNKNOWN",
            task_from_col,
            out["model_mlc_canon"].apply(_canon_task_from_model),
        )
    else:
        out["task_canon"] = out["model_mlc_canon"].apply(_canon_task_from_model)

    out["in_scope"] = out["task_canon"].isin(TASKS_IN_SCOPE)
    out["out_of_scope"] = False
    out["scope_status"] = np.where(out["in_scope"], "IN_SCOPE", "UNKNOWN")

    # OUT_OF_SCOPE Override: 1D_DS_CNN bleibt explizit ausgeschlossen
    is_1d = out["model_mlc_canon"].eq("1D_DS_CNN")
    out.loc[is_1d, "in_scope"] = False
    out.loc[is_1d, "out_of_scope"] = True
    out.loc[is_1d, "task_canon"] = "OUT_OF_SCOPE"
    out.loc[is_1d, "scope_status"] = "OUT_OF_SCOPE"

    # -----------------------------
    # FF1b: Hardware features (minimal)
    # -----------------------------
    if "accelerator" in out.columns:
        out["accelerator_present"] = out["accelerator"].apply(lambda x: _norm_str(x) != "")
    else:
        out["accelerator_present"] = False

    if "processor" in out.columns:
        out["processor_family"] = out["processor"].apply(_infer_processor_family)
    else:
        out["processor_family"] = "UNKNOWN"

    if "host_processor_frequency" in out.columns:
        out["cpu_freq_mhz"] = out["host_processor_frequency"].apply(_parse_freq_mhz)
        out["cpu_freq_bucket"] = out["cpu_freq_mhz"].apply(_bucket_freq_mhz)
    else:
        out["cpu_freq_mhz"] = np.nan
        out["cpu_freq_bucket"] = "UNKNOWN"

    # -----------------------------
    # FF1c: Software features (minimal)
    # -----------------------------
    if "software" in out.columns:
        flags = out["software"].apply(_infer_software_flags)
        out["sw_cmsis_nn"] = flags.apply(lambda d: bool(d["sw_cmsis_nn"]))
        out["sw_tflm"] = flags.apply(lambda d: bool(d["sw_tflm"]))
        out["sw_tvm"] = flags.apply(lambda d: bool(d["sw_tvm"]))
        out["sw_onnx"] = flags.apply(lambda d: bool(d["sw_onnx"]))
        out["software_stack"] = out["software"].apply(_infer_software_stack)
    else:
        out["sw_cmsis_nn"] = False
        out["sw_tflm"] = False
        out["sw_tvm"] = False
        out["sw_onnx"] = False
        out["software_stack"] = "UNKNOWN"

    return out


def get_analysis_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Liefert standardisierte Subsets für EDA/Checks.
    DS_LATENCY/DS_ENERGY werden explizit bereitgestellt.
    """
    if "scope_status" not in df.columns or "task_canon" not in df.columns or "model_mlc_canon" not in df.columns:
        df = add_features(df)

    ds_in_scope = df[df["scope_status"].eq("IN_SCOPE")].copy()
    ds_out_of_scope = df[df["scope_status"].eq("OUT_OF_SCOPE")].copy()
    ds_unknown = df[df["scope_status"].eq("UNKNOWN")].copy()
    ds_non_oos = df[~df["scope_status"].eq("OUT_OF_SCOPE")].copy()

    has_lat = ds_in_scope["latency_us"].notna() if "latency_us" in ds_in_scope.columns else pd.Series(False, index=ds_in_scope.index)
    has_en = ds_in_scope["energy_uj"].notna() if "energy_uj" in ds_in_scope.columns else pd.Series(False, index=ds_in_scope.index)

    ds_in_scope_le = ds_in_scope[has_lat & has_en].copy()
    ds_latency = ds_in_scope[has_lat].copy() if "latency_us" in ds_in_scope.columns else ds_in_scope.iloc[0:0].copy()
    ds_energy = ds_in_scope[has_en].copy() if "energy_uj" in ds_in_scope.columns else ds_in_scope.iloc[0:0].copy()

    return {
        "DS_IN_SCOPE": ds_in_scope,
        "DS_OUT_OF_SCOPE": ds_out_of_scope,
        "DS_UNKNOWN": ds_unknown,
        "DS_NON_OOS": ds_non_oos,
        "DS_IN_SCOPE_LATENCY_ENERGY": ds_in_scope_le,
        "DS_LATENCY": ds_latency,
        "DS_ENERGY": ds_energy,
    }


def sort_rounds(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # feste Reihenfolge (falls neue Runden kommen, bleiben sie am Ende)
    order = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def round_task_counts(df: pd.DataFrame, *, include_unknown: bool = True, include_out_of_scope: bool = True) -> pd.DataFrame:
    """
    Counts pro Round×Task im Wide-Format.
    Rückgabe wird mit index=False gespeichert (Round ist daher eine normale Spalte).
    """
    if "round" not in df.columns or "task_canon" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'task_canon' fehlen.")

    d = df.copy()

    allowed = TASKS_IN_SCOPE.copy()
    if include_unknown:
        allowed.append("UNKNOWN")
    if include_out_of_scope:
        allowed.append("OUT_OF_SCOPE")

    d = d[d["task_canon"].isin(allowed)]
    piv = d.groupby(["round", "task_canon"], dropna=False).size().unstack(fill_value=0)

    # Spalten vollständig/geordnet
    ordered_cols = [c for c in TASKS_ALL_ORDER if c in piv.columns]
    piv = piv.reindex(columns=ordered_cols, fill_value=0)

    out = piv.reset_index().rename_axis(None, axis=1)
    out = sort_rounds(out, "round")
    return out.sort_values("round")


def metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Für jede Round×Task-Gruppe:
      - rows: Anzahl Zeilen
      - rows_with_metric: Anzahl nicht-NA für metric
      - share_metric: Anteil mit metric
    """
    if "round" not in df.columns or "task_canon" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'task_canon' fehlen.")

    if metric not in df.columns:
        out = (
            df.groupby(["round", "task_canon"], dropna=False)
            .size()
            .reset_index(name="rows")
            .rename(columns={"task_canon": "task"})
        )
        out["rows_with_metric"] = 0
        out["share_metric"] = 0.0
        out = sort_rounds(out, "round")
        return out.sort_values(["round", "task"])

    grp = df.groupby(["round", "task_canon"], dropna=False)
    out = grp.size().reset_index(name="rows")
    out["rows_with_metric"] = grp[metric].apply(lambda s: int(s.notna().sum())).values
    out["share_metric"] = np.where(out["rows"] > 0, out["rows_with_metric"] / out["rows"], 0.0)
    out = out.rename(columns={"task_canon": "task"})
    out = sort_rounds(out, "round")
    return out.sort_values(["round", "task"])


def unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pro Round:
      - rows_total
      - rows_unknown
      - share_unknown
    """
    if "round" not in df.columns or "scope_status" not in df.columns:
        raise ValueError("Spalten 'round' und/oder 'scope_status' fehlen.")

    grp = df.groupby("round", dropna=False)
    out = grp.size().reset_index(name="rows_total")
    out["rows_unknown"] = grp["scope_status"].apply(lambda s: int((s == "UNKNOWN").sum())).values
    out["share_unknown"] = np.where(out["rows_total"] > 0, out["rows_unknown"] / out["rows_total"], 0.0)
    out = sort_rounds(out, "round")
    return out.sort_values(["round"])
