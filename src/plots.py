from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "figures"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_round_value(r: str) -> float:
    """
    Rounds kommen typischerweise als 'v0.5', 'v1.3' etc.
    Fallback: sehr groß, damit Unbekanntes ans Ende sortiert.
    """
    if not isinstance(r, str):
        return 1e9
    s = r.strip()
    if s.startswith("v"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return 1e9


def _sort_rounds(df: pd.DataFrame, round_col: str = "round") -> pd.DataFrame:
    if round_col not in df.columns:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].astype(str).map(_parse_round_value)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


def _load_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tabelle nicht gefunden: {path}")
    return pd.read_csv(path)


def plot_latency_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_latency_us.png",
) -> Path:
    """
    Plottet Median (und IQR-Band) von latency_us pro round×task aus der Tabelle
    'trend_latency_us_round_task.csv'.
    """
    df = _load_table(tables_dir, "trend_latency_us_round_task")
    required = {"round", "task", "latency_us_median", "latency_us_q25", "latency_us_q75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in trend_latency_us_round_task.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")

    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    for task, d in df.groupby("task"):
        x = d["round"].astype(str).tolist()
        y = d["latency_us_median"].astype(float).tolist()

        line = ax.plot(x, y, marker="o", label=str(task))[0]
        color = line.get_color()

        q25 = d["latency_us_q25"].astype(float).tolist()
        q75 = d["latency_us_q75"].astype(float).tolist()
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

    ax.set_title("Trend: Latenz (Median) pro Round × Task")
    ax.set_xlabel("Round")
    ax.set_ylabel("latency_us (µs)")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[figure] latency trend -> {out_path}")
    return out_path


def plot_energy_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_energy_uj.png",
) -> Path:
    """
    Plottet Median (und IQR-Band) von energy_uj pro round×task aus der Tabelle
    'trend_energy_uj_round_task.csv'. (Energy-Subset)
    """
    df = _load_table(tables_dir, "trend_energy_uj_round_task")
    required = {"round", "task", "energy_uj_median", "energy_uj_q25", "energy_uj_q75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in trend_energy_uj_round_task.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")

    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    for task, d in df.groupby("task"):
        x = d["round"].astype(str).tolist()
        y = d["energy_uj_median"].astype(float).tolist()

        line = ax.plot(x, y, marker="o", label=str(task))[0]
        color = line.get_color()

        q25 = d["energy_uj_q25"].astype(float).tolist()
        q75 = d["energy_uj_q75"].astype(float).tolist()
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

    ax.set_title("Trend: Energie (Median) pro Round × Task")
    ax.set_xlabel("Round")
    ax.set_ylabel("energy_uj (µJ)")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[figure] energy trend -> {out_path}")
    return out_path


def plot_unknown_share_by_round(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "unknown_share_by_round.png",
) -> Path:
    """
    Plottet share_unknown pro Round aus 'unknown_summary_by_round.csv'.
    """
    df = _load_table(tables_dir, "unknown_summary_by_round")
    required = {"round", "rows_total", "rows_unknown", "share_unknown"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in unknown_summary_by_round.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")

    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    x = df["round"].astype(str).tolist()
    y = df["share_unknown"].astype(float).tolist()
    ax.bar(x, y)

    ax.set_title("Datenqualität: Anteil UNKNOWN (task/model nicht zuordenbar)")
    ax.set_xlabel("Round")
    ax.set_ylabel("share_unknown")
    ax.set_ylim(0, max(0.05, max(y) * 1.1 if y else 0.05))
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[figure] unknown share -> {out_path}")
    return out_path


def run_all(tables_dir: Path, figures_dir: Path) -> None:
    plot_latency_trend(tables_dir, figures_dir)
    plot_energy_trend(tables_dir, figures_dir)
    plot_unknown_share_by_round(tables_dir, figures_dir)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate plots from reports/tables CSVs.")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR_DEFAULT, help="Path to reports/tables")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT, help="Path to reports/figures")

    parser.add_argument("--latency", action="store_true", help="Generate latency trend plot")
    parser.add_argument("--energy", action="store_true", help="Generate energy trend plot")
    parser.add_argument("--unknown", action="store_true", help="Generate unknown share plot")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if no flags set)")

    args = parser.parse_args(argv)

    flags = [args.latency, args.energy, args.unknown, args.all]
    if not any(flags) or args.all:
        run_all(args.tables_dir, args.figures_dir)
        return

    if args.latency:
        plot_latency_trend(args.tables_dir, args.figures_dir)
    if args.energy:
        plot_energy_trend(args.tables_dir, args.figures_dir)
    if args.unknown:
        plot_unknown_share_by_round(args.tables_dir, args.figures_dir)


if __name__ == "__main__":
    main()