from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORE_TASKS = ["AD", "IC", "KWS", "VWW"]
ROUND_ORDER = ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2", "v1.3"]


def _round_rank(r: str) -> int:
    try:
        return ROUND_ORDER.index(str(r))
    except ValueError:
        return 10_000


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _save(fig, out_path: Path, also_svg: bool) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[figure] {out_path}")
    if also_svg:
        out_svg = out_path.with_suffix(".svg")
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"[figure] {out_svg}")


def _prep_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["round_rank"] = df["round"].map(_round_rank)
    df["task_rank"] = df["task"].map({t: i for i, t in enumerate(CORE_TASKS)})
    df = (
        df.sort_values(["round_rank", "task_rank"])
        .drop(columns=["round_rank", "task_rank"])
        .reset_index(drop=True)
    )
    return df


def _plot_trend_raw(
    df: pd.DataFrame,
    metric: str,
    figures_dir: Path,
    min_n: int,
    logy: bool,
    also_svg: bool,
) -> None:
    
    df = _prep_trend(df)

    y_med = f"{metric}_median"
    y_q25 = f"{metric}_q25"
    y_q75 = f"{metric}_q75"
    low_col = f"{metric}_low_n"

    fig, ax = plt.subplots(figsize=(10, 6))

    for task in CORE_TASKS:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue

        x = sub["round"].to_list()
        y = sub[y_med].to_numpy(dtype=float)
        q25 = sub[y_q25].to_numpy(dtype=float)
        q75 = sub[y_q75].to_numpy(dtype=float)

        line = ax.plot(x, y, marker="o", linewidth=2, label=task)[0]
        color = line.get_color()
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

        if low_col in sub.columns:
            low = sub[low_col].fillna(False).to_numpy(dtype=bool)
            if low.any():
                ax.scatter(
                    np.array(x)[low],
                    y[low],
                    s=90,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )

    ax.set_title(f"Trend: {metric} (Median) pro Round × Task (Low-n n<{min_n} markiert)")
    ax.set_xlabel("Round")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    if logy:
        ax.set_yscale("log")

    ax.legend(title="Task", loc="best")
    out_path = figures_dir / f"trend_{metric}.png"
    _save(fig, out_path, also_svg)
    plt.close(fig)


def _plot_trend_index(
    df: pd.DataFrame,
    metric: str,
    figures_dir: Path,
    min_n: int,
    also_svg: bool,
) -> None:
    
    df = _prep_trend(df)

    y_med = f"{metric}_median_index"
    y_q25 = f"{metric}_q25_index"
    y_q75 = f"{metric}_q75_index"
    low_col = f"{metric}_low_n"

    fig, ax = plt.subplots(figsize=(10, 6))

    for task in CORE_TASKS:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue

        x = sub["round"].to_list()
        y = sub[y_med].to_numpy(dtype=float)
        q25 = sub[y_q25].to_numpy(dtype=float)
        q75 = sub[y_q75].to_numpy(dtype=float)

        line = ax.plot(x, y, marker="o", linewidth=2, label=task)[0]
        color = line.get_color()
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

        if low_col in sub.columns:
            low = sub[low_col].fillna(False).to_numpy(dtype=bool)
            mask = low & ~np.isnan(y)
            if mask.any():
                ax.scatter(
                    np.array(x)[mask],
                    y[mask],
                    s=90,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )

    ax.axhline(1.0, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_title(f"Trend (Index): {metric} pro Round × Task (Low-n n<{min_n} markiert)")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{metric}_median_index (baseline=1.0)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.legend(title="Task", loc="best")
    out_path = figures_dir / f"trend_{metric}_index.png"
    _save(fig, out_path, also_svg)
    plt.close(fig)


def _plot_unknown_share(unknown_df: pd.DataFrame, figures_dir: Path, also_svg: bool) -> None:
    df = unknown_df.copy()
    df["round_rank"] = df["round"].map(_round_rank)
    df = df.sort_values("round_rank").drop(columns=["round_rank"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["round"], df["share_unknown"])
    ax.set_title("Datenqualität: Anteil UNKNOWN (task/model nicht zuordenbar)")
    ax.set_xlabel("Round")
    ax.set_ylabel("share_unknown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    out_path = figures_dir / "unknown_share_by_round.png"
    _save(fig, out_path, also_svg)
    plt.close(fig)


def _read_csv_safe(tables_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = tables_dir / name
    if not p.exists():
        print(f"WARN: Tabelle fehlt: {p} (übersprungen)")
        return None
    return pd.read_csv(p)


def _cluster_label_map(profiles: Optional[pd.DataFrame]) -> Dict[int, str]:
    
    if profiles is None or profiles.empty or "cluster_id" not in profiles.columns:
        return {}

    m: Dict[int, str] = {}
    for _, r in profiles.iterrows():
        cid = int(r["cluster_id"])
        cclass = str(r.get("cluster_class", f"C{cid}"))

        acc = str(r.get("accelerator_present", "UNK"))
        fam = str(r.get("processor_family", "UNK"))
        fb = str(r.get("cpu_freq_bucket", "UNK"))

        if acc == "True":
            short = f"{cclass}: Accel / {fam}"
        elif acc == "False":
            short = f"{cclass}: No-Accel / {fam}"
        else:
            short = f"{cclass}: {fam}"

        if fb not in ("UNKNOWN", "nan", "NaN"):
            short = f"{short} ({fb})"

        m[cid] = short

    return m

def _plot_ff2_pareto_scatter_4panel(
    pareto_points: pd.DataFrame,
    figures_dir: Path,
    also_svg: bool,
    profiles: Optional[pd.DataFrame] = None,
) -> None:
    
    df = pareto_points.copy()

    required = {"task", "latency_us", "energy_uj", "cluster_id", "pareto_flag"}
    missing = required - set(df.columns)
    if missing:
        print(f"WARN: ff2_pareto_points missing columns {sorted(list(missing))} -> skip FF2 pareto plot")
        return

    df = df[df["latency_us"].notna() & df["energy_uj"].notna()].copy()
    if df.empty:
        print("WARN: Keine Zeilen mit latency_us & energy_uj vorhanden -> FF2 Pareto-Plot übersprungen")
        return

    df["cluster_id"] = df["cluster_id"].astype(int)
    df["pareto_flag"] = df["pareto_flag"].fillna(0).astype(int)
    df["task"] = df["task"].astype(str)

    cmap = _cluster_label_map(profiles)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    cluster_ids = sorted(df["cluster_id"].unique().tolist())

    for i, task in enumerate(CORE_TASKS):

        ax = axes[i]
        sub = df[df["task"] == task].copy()

        ax.set_title(task)

        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            continue

        for cid in cluster_ids:

            s2 = sub[sub["cluster_id"] == cid]

            if s2.empty:
                continue

            label = cmap.get(cid, f"Cluster {cid}")

            ax.scatter(
                s2["latency_us"].to_numpy(dtype=float),
                s2["energy_uj"].to_numpy(dtype=float),
                s=18,
                alpha=0.75,
                label=label,
            )

            s_p = s2[s2["pareto_flag"] == 1]

            if not s_p.empty:

                ax.scatter(
                    s_p["latency_us"].to_numpy(dtype=float),
                    s_p["energy_uj"].to_numpy(dtype=float),
                    s=70,
                    marker="X",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.3,
                    zorder=5,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle(
        "FF2: Pareto-Trade-off (Latency vs Energy) – 4 Tasks (Farbe=Cluster, Pareto umrandet)",
        y=0.98,
    )

    fig.text(0.5, 0.04, "latency_us (µs / inference, log)", ha="center")
    fig.text(0.04, 0.5, "energy_uj (µJ / inference, log)", va="center", rotation="vertical")

    all_h, all_l = [], []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_h.extend(h)
        all_l.extend(l)

    if all_h:

        uniq = {}

        for h, l in zip(all_h, all_l):
            if l not in uniq:
                uniq[l] = h

        def _cluster_order(label: str) -> int:
            try:
                return int(label.split(":")[0].replace("C", ""))
            except Exception:
                return 999

        labels_sorted = sorted(uniq.keys(), key=_cluster_order)
        handles_sorted = [uniq[l] for l in labels_sorted]

        fig.legend(
            handles_sorted,
            labels_sorted,
            loc="center right",
            bbox_to_anchor=(1.18, 0.5),
            title="Cluster",
        )

    plt.tight_layout(rect=[0.06, 0.06, 0.86, 0.94])

    out_path = figures_dir / "ff2_pareto_scatter_latency_vs_energy_by_task.png"

    _save(fig, out_path, also_svg)

    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="reports/tables")
    ap.add_argument("--figures-dir", default="reports/figures")
    ap.add_argument("--min-n", type=int, default=5)
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    _ensure_dirs(figures_dir)

    min_n = int(args.min_n)

    also_svg = True
    logy = True

    # FF1 raw trends
    lat = _read_csv_safe(tables_dir, "trend_latency_us_round_task.csv")
    en = _read_csv_safe(tables_dir, "trend_energy_uj_round_task.csv")
    if lat is not None:
        _plot_trend_raw(lat, "latency_us", figures_dir, min_n=min_n, logy=logy, also_svg=also_svg)
    if en is not None:
        _plot_trend_raw(en, "energy_uj", figures_dir, min_n=min_n, logy=logy, also_svg=also_svg)

    # FF1 index trends
    lat_i = _read_csv_safe(tables_dir, "trend_latency_us_round_task_indexed.csv")
    en_i = _read_csv_safe(tables_dir, "trend_energy_uj_round_task_indexed.csv")
    if lat_i is not None:
        _plot_trend_index(lat_i, "latency_us", figures_dir, min_n=min_n, also_svg=also_svg)
    if en_i is not None:
        _plot_trend_index(en_i, "energy_uj", figures_dir, min_n=min_n, also_svg=also_svg)

    # UNKNOWN share
    unk = _read_csv_safe(tables_dir, "unknown_summary_by_round.csv")
    if unk is not None:
        _plot_unknown_share(unk, figures_dir, also_svg=also_svg)

    # FF2 pareto scatter
    pareto_pts = _read_csv_safe(tables_dir, "ff2_pareto_points_round_task.csv")
    prof = _read_csv_safe(tables_dir, "ff2_cluster_profiles.csv")
    if pareto_pts is not None:
        _plot_ff2_pareto_scatter_4panel(
            pareto_pts,
            figures_dir,
            also_svg=also_svg,
            profiles=prof,
        )


if __name__ == "__main__":
    main()