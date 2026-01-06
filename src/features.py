# src/features.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ------------------------------------------------------------
# Canonical vocab / configuration
# ------------------------------------------------------------

PRIMARY_TASKS = ("AD", "IC", "KWS", "VWW")  # Fokus deiner bisherigen Trends
EXTENDED_TASKS = ("SWW",)  # Streaming Wakeword (v1.3)
TASK_ORDER = (*PRIMARY_TASKS, *EXTENDED_TASKS, "UNKNOWN")

MODEL_ORDER = (
    "DS-CNN",
    "1D DS-CNN",
    "FC AutoEncoder",
    "MobileNetV1 (0.25x)",
    "ResNet-V1",
    "Dense",
    "NULL",
    "UNKNOWN",
)

MODE_ORDER = ("single_stream", "offline", "streaming", "unknown")


@dataclass(frozen=True)
class CanonicalColumns:
    # raw inputs (optional)
    task_raw: str = "task"
    model_raw: str = "model_mlc"
    units: str = "units"
    benchmark: str = "benchmark"

    # metrics (optional)
    latency_us: str = "latency_us"
    energy_uj: str = "energy_uj"
    power_mw: str = "power_mw"
    accuracy: str = "accuracy"
    auc: str = "auc"

    # metadata (optional)
    processor: str = "processor"
    accelerator: str = "accelerator"
    software: str = "software"
    round: str = "round"


COL = CanonicalColumns()


# ------------------------------------------------------------
# Helpers: string normalization
# ------------------------------------------------------------

def _norm_str(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _norm_key(x: object) -> str:
    """Lowercase, collapse whitespace, normalize hyphens."""
    s = _norm_str(x).lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    return s


def _is_null_like(model: str) -> bool:
    s = _norm_key(model)
    return s in ("", "nan", "none", "null", "<na>")


# ------------------------------------------------------------
# Canonicalization: model_mlc
# ------------------------------------------------------------

def canon_model_mlc(model: object) -> str:
    """
    Mappt heterogene Model-Labels aus MLCommons-Exports auf kanonische Namen.

    Wichtig:
      - "1D DS-CNN" wird NICHT zu "DS-CNN" gemappt.
        Grund: Streaming Wakeword (streaming) ist konzeptionell ein eigener Benchmark/Mode.
    """
    raw = _norm_str(model)
    if _is_null_like(raw):
        return "NULL"

    s = _norm_key(raw)

    # 1D DS-CNN (Streaming Wakeword)
    if "1d" in s and ("ds-cnn" in s or "dscnn" in s or "ds cnn" in s):
        return "1D DS-CNN"

    # DS-CNN (Keyword Spotting)
    if any(k in s for k in ("ds-cnn", "dscnn", "ds cnn")):
        return "DS-CNN"

    # Autoencoder / Dense
    if "autoencoder" in s:
        return "FC AutoEncoder"
    if s == "dense" or " dense" in s or s.endswith("dense"):
        return "Dense"

    # Vision models
    if "mobilenet" in s:
        # häufig "MobileNetV1 (0.25x)" (VWW)
        return "MobileNetV1 (0.25x)"
    if "resnet" in s:
        # häufig "ResNet-V1" (IC)
        return "ResNet-V1"

    # Fallback
    return "UNKNOWN"


def model_family(model_canon: str) -> str:
    """
    Grobe Modellfamilie für Gruppierungen/EDA.
    """
    m = model_canon
    if m in ("DS-CNN", "1D DS-CNN"):
        return "DS-CNN"
    if m in ("FC AutoEncoder", "Dense"):
        return "AutoEncoder/Dense"
    if m == "MobileNetV1 (0.25x)":
        return "MobileNet"
    if m == "ResNet-V1":
        return "ResNet"
    if m == "NULL":
        return "NULL"
    return "UNKNOWN"


def model_dim(model_canon: str) -> str:
    """
    Dimensions-Tag aus der kanonischen Modellbezeichnung.
    - "1D" für 1D DS-CNN
    - sonst "2D_or_unspecified" (weil Exporte oft nicht explizit sind)
    """
    if model_canon == "1D DS-CNN":
        return "1D"
    if model_canon in ("NULL", "UNKNOWN"):
        return "unknown"
    return "2D_or_unspecified"


# ------------------------------------------------------------
# Canonicalization: task
# ------------------------------------------------------------

def _canon_task_from_raw(task: object) -> str:
    s = _norm_key(task)
    if s in ("ad", "anomaly detection", "anomaly"):
        return "AD"
    if s in ("ic", "image classification", "cifar-10", "cifar10"):
        return "IC"
    if s in ("kws", "keyword spotting", "keyword"):
        return "KWS"
    if s in ("vww", "visual wake words", "person detection", "person"):
        return "VWW"
    if s in ("sww", "streaming wakeword", "streaming wake word", "streaming"):
        return "SWW"
    if s in ("unknown", "na", "n/a", ""):
        return "UNKNOWN"
    # unbekannte Labels nicht wegwerfen, aber als UNKNOWN behandeln
    return "UNKNOWN"


def infer_task_from_model(model_canon: str) -> str:
    """
    Inferenz von Task aus kanonischem Modellnamen (Fallback).
    """
    if model_canon == "1D DS-CNN":
        return "SWW"
    if model_canon == "DS-CNN":
        return "KWS"
    if model_canon == "MobileNetV1 (0.25x)":
        return "VWW"
    if model_canon == "ResNet-V1":
        return "IC"
    if model_canon in ("FC AutoEncoder", "Dense"):
        return "AD"
    return "UNKNOWN"


def canon_task(task: object, model_canon: str) -> str:
    """
    Korrigiert/vereinheitlicht Task:
      1) Raw task normalisieren (falls vorhanden)
      2) Falls UNKNOWN oder inkonsistent, aus model_canon ableiten
      3) Speziell: Wenn raw als KWS kommt, aber model_canon == "1D DS-CNN" => SWW
    """
    t = _canon_task_from_raw(task)

    # Spezifische Korrektur: 1D DS-CNN ist Streaming Wakeword (nicht KWS)
    if model_canon == "1D DS-CNN":
        return "SWW"

    # wenn raw task fehlt/unknown -> aus Modell ableiten
    if t == "UNKNOWN":
        return infer_task_from_model(model_canon)

    # optionaler Konsistenz-Guard: Wenn raw task stark widerspricht, Modell-Fallback
    inferred = infer_task_from_model(model_canon)
    if inferred != "UNKNOWN" and t != inferred:
        # Beispiel: raw sagt KWS, aber Modell ist ResNet -> dann ist raw vermutlich falsch/Artefakt
        return inferred

    return t


# ------------------------------------------------------------
# Canonicalization: mode (optional)
# ------------------------------------------------------------

def infer_mode(task_canon: str, model_canon: str, units: object = None) -> str:
    """
    Mode ist in deinen CSV-Exports nicht immer direkt vorhanden.
    Minimaler, robuster Heuristik-Ansatz:
      - Streaming Wakeword => streaming
      - KWS => offline (in MLPerf Tiny häufig als "Single-stream, Offline" geführt)
      - sonst => single_stream
    """
    if task_canon == "SWW" or model_canon == "1D DS-CNN":
        return "streaming"
    if task_canon == "KWS":
        return "offline"
    if task_canon in ("AD", "IC", "VWW"):
        return "single_stream"
    return "unknown"


# ------------------------------------------------------------
# Numeric helpers (safe parsing)
# ------------------------------------------------------------

_FREQ_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>ghz|mhz|khz)\b", re.IGNORECASE)


def parse_freq_mhz(text: object) -> Optional[float]:
    """
    Extrahiert eine Frequenz aus Freitext und gibt MHz zurück.
    Beispiele: "80MHz", "1.2 GHz", "500 khz"
    """
    s = _norm_str(text)
    if not s:
        return None
    m = _FREQ_RE.search(s)
    if not m:
        return None
    val = float(m.group("val"))
    unit = m.group("unit").lower()
    if unit == "ghz":
        return val * 1000.0
    if unit == "mhz":
        return val
    if unit == "khz":
        return val / 1000.0
    return None


# ------------------------------------------------------------
# Public API: feature engineering
# ------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt robuste, reproduzierbare Features für EDA/Trends/Clustering.

    Output-Spalten (neu):
      - model_mlc_canon, model_family, model_dim
      - task_canon, mode_canon
      - has_latency, has_energy, has_power, has_quality
      - quality_value, quality_metric
      - has_accelerator, software_norm
      - processor_freq_mhz (best-effort aus Text)
      - round_value (numeric für Sorting)
    """
    out = df.copy()

    # --- canonical model ---
    model_raw = out[COL.model_raw] if COL.model_raw in out.columns else pd.Series([""] * len(out))
    out["model_mlc_canon"] = model_raw.map(canon_model_mlc)
    out["model_family"] = out["model_mlc_canon"].map(model_family)
    out["model_dim"] = out["model_mlc_canon"].map(model_dim)

    # --- canonical task ---
    task_raw = out[COL.task_raw] if COL.task_raw in out.columns else pd.Series([""] * len(out))
    out["task_canon"] = [
        canon_task(t, m) for t, m in zip(task_raw.tolist(), out["model_mlc_canon"].tolist())
    ]

    # --- mode ---
    units = out[COL.units] if COL.units in out.columns else pd.Series([""] * len(out))
    out["mode_canon"] = [
        infer_mode(t, m, u) for t, m, u in zip(out["task_canon"], out["model_mlc_canon"], units)
    ]

    # --- metric availability flags ---
    if COL.latency_us in out.columns:
        out["has_latency"] = out[COL.latency_us].notna()
    else:
        out["has_latency"] = False

    if COL.energy_uj in out.columns:
        out["has_energy"] = out[COL.energy_uj].notna()
    else:
        out["has_energy"] = False

    if COL.power_mw in out.columns:
        out["has_power"] = out[COL.power_mw].notna()
    else:
        out["has_power"] = False

    # Quality: accuracy/auc
    acc = out[COL.accuracy] if COL.accuracy in out.columns else pd.Series([pd.NA] * len(out))
    auc = out[COL.auc] if COL.auc in out.columns else pd.Series([pd.NA] * len(out))

    # prefer AUC for AD / Streaming Wakeword? (SWW hat andere Quality; in Exports oft nicht als AUC/accuracy drin)
    quality_value = acc.copy()
    quality_metric = pd.Series(["accuracy"] * len(out))

    use_auc = (out["task_canon"] == "AD") & auc.notna()
    quality_value.loc[use_auc] = auc.loc[use_auc]
    quality_metric.loc[use_auc] = "auc"

    out["quality_value"] = quality_value
    out["quality_metric"] = quality_metric
    out["has_quality"] = out["quality_value"].notna()

    # --- metadata convenience ---
    if COL.accelerator in out.columns:
        out["has_accelerator"] = out[COL.accelerator].astype(str).map(_norm_key).ne("")
        # "null" strings -> False
        out["has_accelerator"] = out["has_accelerator"] & ~out[COL.accelerator].astype(str).map(_is_null_like)
    else:
        out["has_accelerator"] = False

    if COL.software in out.columns:
        out["software_norm"] = out[COL.software].astype(str).map(_norm_key)
    else:
        out["software_norm"] = ""

    # --- best-effort frequency parse from processor text ---
    proc_text = out[COL.processor] if COL.processor in out.columns else pd.Series([""] * len(out))
    out["processor_freq_mhz"] = proc_text.map(parse_freq_mhz)

    # --- numeric round for sorting ---
    if COL.round in out.columns:
        out["round_value"] = out[COL.round].astype(str).str.replace("v", "", regex=False).apply(
            lambda x: float(x) if re.fullmatch(r"\d+(\.\d+)?", x.strip()) else float("nan")
        )
    else:
        out["round_value"] = float("nan")

    return out


# Backwards-compatible aliases (falls eda.py o.ä. andere Namen nutzt)
apply_features = add_features
build_features = add_features
enrich_features = add_features
