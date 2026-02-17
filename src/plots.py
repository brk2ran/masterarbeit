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
    """
    Plot: Median pro Round×Task + IQR-Band (q25-q75), low-n Markierung.
    """
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

        ax.plot(x, y, marker="o", linewidth=2, label=task)
        ax.fill_between(x, q25, q75, alpha=0.15)

        # low-n markieren (n<min_n)
        low = sub[low_col].fillna(False).to_numpy(dtype=bool)
        if low.any():
            ax.scatter(
                np.array(x)[low],
                y[low],
                s=90,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
            )

    ax.set_title(f"Trend: {metric} (Median) pro Round × Task (Low-n n<{min_n} markiert)")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{metric}")
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
    """
    Plot: Index (Median / baseline_median), baseline=1.0 Linie, IQR-Band im Indexraum.
    Pre-Baseline ist NaN -> wird nicht gezeichnet.
    """
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

        ax.plot(x, y, marker="o", linewidth=2, label=task)
        ax.fill_between(x, q25, q75, alpha=0.15)

        # low-n markieren (aber nur wo y nicht NaN, also nicht pre-baseline)
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
                zorder=3,
            )

    ax.axhline(1.0, linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_title(f"Trend (Index): {metric} (Median-Index) pro Round × Task (Low-n n<{min_n} markiert)")
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


# -------------------------
# FF2 plots
# -------------------------
def _cluster_label_map(profiles: Optional[pd.DataFrame]) -> Dict[int, str]:
    """
    Create concise, deterministic labels for clusters.
    If profiles missing, fallback to "Cluster <id>".
    """
    if profiles is None or profiles.empty or "cluster_id" not in profiles.columns:
        return {}

    m: Dict[int, str] = {}
    for _, r in profiles.iterrows():
        cid = int(r["cluster_id"])
        acc = str(r.get("accelerator_present", "UNK"))
        fam = str(r.get("processor_family", "UNK"))
        fb = str(r.get("cpu_freq_bucket", "UNK"))

        # keep label short; still informative
        if acc == "True":
            short = f"C{cid}: Accel / {fam}"
        elif acc == "False":
            short = f"C{cid}: No-Accel / {fam}"
        else:
            short = f"C{cid}: {fam}"

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
    """
    4-Panel Scatter (AD/IC/KWS/VWW):
    x=latency_us (log), y=energy_uj (log), color=cluster_id, pareto points outlined.
    Uses only rows with both metrics.
    """
    df = pareto_points.copy()

    # columns sanity
    required = {"task", "latency_us", "energy_uj", "cluster_id", "pareto_flag"}
    missing = required - set(df.columns)
    if missing:
        print(f"WARN: ff2_pareto_points missing columns {sorted(list(missing))} -> skip FF2 pareto plot")
        return

    # only rows with both metrics (Pareto candidates)
    df = df[df["latency_us"].notna() & df["energy_uj"].notna()].copy()
    if df.empty:
        print("WARN: Keine Zeilen mit latency_us & energy_uj vorhanden -> FF2 Pareto-Plot übersprungen")
        return

    # stable types
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["pareto_flag"] = df["pareto_flag"].fillna(0).astype(int)
    df["task"] = df["task"].astype(str)

    # label mapping
    cmap = _cluster_label_map(profiles)

    # prepare panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    # consistent set of clusters across panels
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

        # plot each cluster separately to get legend entries
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

            # highlight pareto points (outline)
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

    # common labels
    fig.suptitle("FF2: Pareto-Trade-off (Latency vs Energy) – 4 Tasks (Farbe=Cluster, Pareto umrandet)", y=0.98)
    fig.text(0.5, 0.04, "latency_us (µs / inference, log)", ha="center")
    fig.text(0.04, 0.5, "energy_uj (µJ / inference, log)", va="center", rotation="vertical")

    # one legend for all panels (right side)
    # collect legend entries from ALL panels (otherwise clusters missing if not present in AD)
    all_h, all_l = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_h.extend(h)
        all_l.extend(l)

    if all_h:
        # de-duplicate labels while preserving order
        seen = set()
        h2, l2 = [], []
        for h, l in zip(all_h, all_l):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h)
            l2.append(l)

        # place legend outside (no overlap)
        fig.legend(h2, l2, loc="center right", bbox_to_anchor=(1.18, 0.5), title="Cluster")

    plt.tight_layout(rect=[0.06, 0.06, 0.86, 0.94])
    out_path = figures_dir / "ff2_pareto_scatter_latency_vs_energy_by_task.png"
    _save(fig, out_path, also_svg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="reports/tables")
    ap.add_argument("--figures-dir", default="reports/figures")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--logy", action="store_true", help="log-scale for raw trend plots")
    ap.add_argument("--svg", action="store_true", help="also export svg (baseline expects this if you froze with svg)")
    ap.add_argument("--all", action="store_true", help="generate core figures (trend raw+index + unknown share + FF2 if available)")
    ap.add_argument("--no-index", action="store_true", help="do not generate index plots")
    ap.add_argument("--ff2", action="store_true", help="generate FF2 pareto scatter (4 panels)")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    _ensure_dirs(figures_dir)

    min_n = int(args.min_n)

    if args.all and not args.svg:
        print("INFO: --svg ist aus. Wenn deine Baseline SVGs enthält, nutze --svg, sonst meldet checks 'missing_current'.")

    if not args.all and not args.ff2:
        print("Nutze --all für FF1-Plots oder --ff2 für den FF2-Pareto-Plot.")
        return

    # -----------------
    # FF1 core plots (existing behavior)
    # -----------------
    if args.all:
        # raw trends
        lat = _read_csv_safe(tables_dir, "trend_latency_us_round_task.csv")
        en = _read_csv_safe(tables_dir, "trend_energy_uj_round_task.csv")
        if lat is not None:
            _plot_trend_raw(lat, "latency_us", figures_dir, min_n=min_n, logy=True, also_svg=args.svg)
        if en is not None:
            _plot_trend_raw(en, "energy_uj", figures_dir, min_n=min_n, logy=True, also_svg=args.svg)

        # index trends
        if not args.no_index:
            lat_i = _read_csv_safe(tables_dir, "trend_latency_us_round_task_indexed.csv")
            en_i = _read_csv_safe(tables_dir, "trend_energy_uj_round_task_indexed.csv")
            if lat_i is not None:
                _plot_trend_index(lat_i, "latency_us", figures_dir, min_n=min_n, also_svg=args.svg)
            if en_i is not None:
                _plot_trend_index(en_i, "energy_uj", figures_dir, min_n=min_n, also_svg=args.svg)

        # unknown share
        unk = _read_csv_safe(tables_dir, "unknown_summary_by_round.csv")
        if unk is not None:
            _plot_unknown_share(unk, figures_dir, also_svg=args.svg)

    # -----------------
    # FF2 plot (Pareto scatter 4-panel)
    # -----------------
    if args.ff2 or args.all:
        pareto_pts = _read_csv_safe(tables_dir, "ff2_pareto_points_round_task.csv")
        prof = _read_csv_safe(tables_dir, "ff2_cluster_profiles.csv")
        if pareto_pts is not None:
            _plot_ff2_pareto_scatter_4panel(pareto_pts, figures_dir, also_svg=args.svg, profiles=prof)


if __name__ == "__main__":
    main()
