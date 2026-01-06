from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd


# Primärer Analyse-Scope (dein Fokus)
PRIMARY_TASKS = ("AD", "IC", "KWS", "VWW")

# Interne Labels
TASK_UNKNOWN = "UNKNOWN"
TASK_OUT_OF_SCOPE = "OUT_OF_SCOPE"

MODEL_UNKNOWN = "UNKNOWN"
MODEL_NULL = "NULL"

# Canonical Model Tokens (kompakt, stabil für Groupbys)
MODEL_DS_CNN = "DS_CNN"
MODEL_1D_DS_CNN = "1D_DS_CNN"
MODEL_FC_AE = "FC_AE"
MODEL_MOBILENETV1 = "MOBILENETV1"
MODEL_RESNET_V1 = "RESNET_V1"

MODE_UNKNOWN = "UNKNOWN"
MODE_SINGLE_STREAM = "single_stream"
MODE_OFFLINE = "offline"
MODE_OUT_OF_SCOPE = "out_of_scope"


# -----------------------------
# Canonicalization helpers
# -----------------------------
def _norm(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def canon_model_mlc(model_mlc: object) -> str:
    """
    Canonicalize Model MLC to a compact token.
    Wichtig: 1D DS-CNN bleibt erkennbar (MODEL_1D_DS_CNN), wird aber später als OUT_OF_SCOPE markiert.
    """
    s = _norm(model_mlc)
    if not s or s.lower() in {"null", "none", "nan", "<na>"}:
        return MODEL_NULL

    s_low = s.lower()

    # 1D DS-CNN (Streaming Wakeword)
    if "1d" in s_low and ("ds-cnn" in s_low or "ds cnn" in s_low or "ds_cnn" in s_low):
        return MODEL_1D_DS_CNN

    # DS-CNN (Keyword Spotting)
    # Achtung: "DS-CNN" ist in MLPerf Tiny üblich als DS-CNN / DS CNN / DSCNN
    if "ds-cnn" in s_low or "ds cnn" in s_low or re.fullmatch(r"dscnn", s_low):
        return MODEL_DS_CNN

    # FC AutoEncoder / Dense AE
    if "autoencoder" in s_low or "fc autoencoder" in s_low or "fc_autoencoder" in s_low or "dense" == s_low:
        return MODEL_FC_AE

    # MobileNetV1
    if "mobilenet" in s_low:
        return MODEL_MOBILENETV1

    # ResNet-V1
    if "resnet" in s_low:
        return MODEL_RESNET_V1

    return MODEL_UNKNOWN


def infer_task_from_model(model_canon: str) -> str:
    """
    Fallback: wenn Task fehlt/unklar, kann man ihn aus Model MLC ableiten (MLPerf Tiny Konvention).
    """
    if model_canon == MODEL_FC_AE:
        return "AD"
    if model_canon == MODEL_RESNET_V1:
        return "IC"
    if model_canon == MODEL_MOBILENETV1:
        return "VWW"
    if model_canon == MODEL_DS_CNN:
        return "KWS"
    if model_canon == MODEL_1D_DS_CNN:
        # bewusst als out-of-scope markiert (Streaming Wakeword)
        return TASK_OUT_OF_SCOPE
    return TASK_UNKNOWN


def canon_task(task: object, *, model_canon: str) -> str:
    """
    Canonicalize task.
    - Wenn Model 1D_DS_CNN -> OUT_OF_SCOPE.
    - Wenn Task fehlt, inferiere aus Model.
    """
    if model_canon == MODEL_1D_DS_CNN:
        return TASK_OUT_OF_SCOPE

    t = _norm(task).upper()
    if not t or t in {"N/A", "NA", "NONE", "NAN", "<NA>"}:
        return infer_task_from_model(model_canon)

    # akzeptiere nur die primären Tasks
    if t in PRIMARY_TASKS:
        return t

    # alles andere ist für deine Arbeit "nicht zuordenbar"
    return TASK_UNKNOWN


def infer_mode(task_canon: str) -> str:
    """
    Mode ist in MLPerf Tiny nicht immer sauber im CSV; wir leiten ihn aus dem Task ab.
    """
    if task_canon == TASK_OUT_OF_SCOPE:
        return MODE_OUT_OF_SCOPE
    if task_canon == "KWS":
        return MODE_OFFLINE
    if task_canon in {"AD", "IC", "VWW"}:
        return MODE_SINGLE_STREAM
    return MODE_UNKNOWN


# -----------------------------
# Public API: add_features + EDA utilities
# -----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - model_mlc_canon
      - task_canon
      - mode_canon
      - out_of_scope (bool)
      - in_scope (bool; PRIMARY_TASKS only)
    """
    d = df.copy()

    if "model_mlc" not in d.columns:
        d["model_mlc"] = pd.NA
    if "task" not in d.columns:
        d["task"] = pd.NA

    d["model_mlc_canon"] = d["model_mlc"].map(canon_model_mlc)

    d["task_canon"] = [
        canon_task(t, model_canon=m)
        for t, m in zip(d["task"], d["model_mlc_canon"])
    ]

    d["mode_canon"] = d["task_canon"].map(infer_mode)

    d["out_of_scope"] = d["task_canon"].eq(TASK_OUT_OF_SCOPE)
    d["in_scope"] = d["task_canon"].isin(PRIMARY_TASKS)

    return d


def get_analysis_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Liefert konsistente Subsets, die überall gleich verwendet werden.
    OUT_OF_SCOPE ist explizit ausgeschlossen.
    """
    d = add_features(df)

    ds_base = d[d["in_scope"]].copy()
    ds_latency = ds_base[ds_base["latency_us"].notna()].copy()

    ds_energy = ds_base[ds_base["energy_uj"].notna()].copy() if "energy_uj" in ds_base.columns else ds_base.iloc[0:0].copy()
    ds_power = ds_base[ds_base["power_mw"].notna()].copy() if "power_mw" in ds_base.columns else ds_base.iloc[0:0].copy()
    ds_accuracy = ds_base[ds_base["accuracy"].notna()].copy() if "accuracy" in ds_base.columns else ds_base.iloc[0:0].copy()
    ds_auc = ds_base[ds_base["auc"].notna()].copy() if "auc" in ds_base.columns else ds_base.iloc[0:0].copy()

    ds_out = d[d["out_of_scope"]].copy()
    ds_unknown = d[(~d["out_of_scope"]) & (d["task_canon"] == TASK_UNKNOWN)].copy()

    return {
        "DS_BASE": ds_base,
        "DS_LATENCY": ds_latency,
        "DS_ENERGY": ds_energy,
        "DS_POWER": ds_power,
        "DS_ACCURACY": ds_accuracy,
        "DS_AUC": ds_auc,
        "DS_OUT_OF_SCOPE": ds_out,
        "DS_UNKNOWN": ds_unknown,
    }


def round_task_counts(
    df: pd.DataFrame,
    *,
    include_unknown: bool = True,
    include_out_of_scope: bool = False,
) -> pd.DataFrame:
    """
    Coverage Round×Task (Zeilenanzahl) als Pivot.
    Default: UNKNOWN drin, OUT_OF_SCOPE raus.
    """
    d = add_features(df)

    if not include_out_of_scope:
        d = d[~d["out_of_scope"]].copy()

    if not include_unknown:
        d = d[d["task_canon"] != TASK_UNKNOWN].copy()

    g = (
        d.groupby(["round", "task_canon"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )

    # Pivot für schnell lesbare Coverage-Übersicht
    out = g.pivot_table(index="round", columns="task_canon", values="rows", aggfunc="sum", fill_value=0)
    out = out.reset_index()

    # stabile Spaltenreihenfolge
    ordered_cols = ["round"] + [t for t in PRIMARY_TASKS if t in out.columns]
    if include_unknown and TASK_UNKNOWN in out.columns:
        ordered_cols.append(TASK_UNKNOWN)
    if include_out_of_scope and TASK_OUT_OF_SCOPE in out.columns:
        ordered_cols.append(TASK_OUT_OF_SCOPE)

    # plus alle ggf. übrigen
    rest = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + rest]


def unknown_summary_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNKNOWN Summary by Round:
    - OUT_OF_SCOPE wird aus der Betrachtung rausgenommen (wichtig für deine Entscheidung).
    - rows_total = total ohne OUT_OF_SCOPE
    """
    d = add_features(df)
    d = d[~d["out_of_scope"]].copy()

    g = d.groupby("round", dropna=False).agg(
        rows_total=("round", "size"),
        rows_unknown=("task_canon", lambda s: int((s == TASK_UNKNOWN).sum())),
    )
    g["share_unknown"] = g["rows_unknown"] / g["rows_total"].where(g["rows_total"] != 0, pd.NA)
    return g.reset_index().sort_values("round")


def metric_coverage_by_round_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Anteil nicht-null je Metrik (Round×Task) innerhalb des PRIMARY_SCOPE.
    Erwartet i. d. R. ein in-scope Subset (z.B. DS_LATENCY als Basis).
    """
    d = add_features(df)
    d = d[d["in_scope"]].copy()

    if metric not in d.columns:
        return pd.DataFrame(columns=["round", "task", "rows", "rows_with_metric", "share_metric"])

    g = d.groupby(["round", "task_canon"], dropna=False).agg(
        rows=("round", "size"),
        rows_with_metric=(metric, lambda s: int(s.notna().sum())),
    )
    g["share_metric"] = g["rows_with_metric"] / g["rows"].where(g["rows"] != 0, pd.NA)
    out = g.reset_index().rename(columns={"task_canon": "task"})
    return out.sort_values(["round", "task"])
