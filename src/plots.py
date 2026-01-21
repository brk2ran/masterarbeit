from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

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
    df = df.sort_values(["round_rank", "task_rank"]).drop(columns=["round_rank", "task_rank"]).reset_index(drop=True)
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
    n_col = f"{metric}_n"
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
            ax.scatter(np.array(x)[low], y[low], s=90, facecolors="none", edgecolors="black", linewidths=1.5, zorder=3)

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

        # low-n markieren (n<min_n) – aber nur wo y nicht NaN (also nicht pre-baseline)
        low = sub[low_col].fillna(False).to_numpy(dtype=bool)
        mask = low & ~np.isnan(y)
        if mask.any():
            ax.scatter(np.array(x)[mask], y[mask], s=90, facecolors="none", edgecolors="black", linewidths=1.5, zorder=3)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="reports/tables")
    ap.add_argument("--figures-dir", default="reports/figures")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--logy", action="store_true", help="log-scale for raw trend plots")
    ap.add_argument("--svg", action="store_true", help="also export svg")
    ap.add_argument("--all", action="store_true", help="generate all figures")
    ap.add_argument("--no-index", action="store_true", help="do not generate index plots")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    _ensure_dirs(figures_dir)

    min_n = int(args.min_n)

    def read_csv(name: str) -> pd.DataFrame:
        p = tables_dir / name
        return pd.read_csv(p)

    if args.all:
        # raw trends
        lat = read_csv("trend_latency_us_round_task.csv")
        en = read_csv("trend_energy_uj_round_task.csv")
        _plot_trend_raw(lat, "latency_us", figures_dir, min_n=min_n, logy=True, also_svg=args.svg)
        _plot_trend_raw(en, "energy_uj", figures_dir, min_n=min_n, logy=True, also_svg=args.svg)

        # index trends
        if not args.no_index:
            lat_i = read_csv("trend_latency_us_round_task_indexed.csv")
            en_i = read_csv("trend_energy_uj_round_task_indexed.csv")
            _plot_trend_index(lat_i, "latency_us", figures_dir, min_n=min_n, also_svg=args.svg)
            _plot_trend_index(en_i, "energy_uj", figures_dir, min_n=min_n, also_svg=args.svg)

        # unknown share
        unk = read_csv("unknown_summary_by_round.csv")
        _plot_unknown_share(unk, figures_dir, also_svg=args.svg)

        return

    # (Optional: wenn du später Einzelflags willst, kannst du hier erweitern.)
    print("Nutze --all für den vollständigen Plot-Lauf.")


if __name__ == "__main__":
    main()
