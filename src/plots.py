from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "figures"


# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _round_key(r: object) -> float:
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
    if df.empty or round_col not in df.columns:
        return df
    out = df.copy()
    out["_round_order"] = out[round_col].astype(str).map(_round_key)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


def _load_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tabelle nicht gefunden: {path}")
    return pd.read_csv(path)


def _safe_set_logy(ax: plt.Axes, series_list: list[pd.Series], *, context: str) -> bool:
    """
    Setzt y-Achse auf log, wenn alle Werte (ohne NaN) > 0 sind.
    """
    for s in series_list:
        if s is None or s.empty:
            continue
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            continue
        if (s2 <= 0).any():
            print(f"WARN: logy deaktiviert ({context}) – nicht-positive Werte vorhanden.")
            return False
    ax.set_yscale("log")
    return True


def _save_figure(fig: plt.Figure, out_path: Path, *, save_svg: bool) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"))


def _low_n_mask(d: pd.DataFrame, *, low_n_col: str, n_col: str, min_n: int) -> tuple[pd.Series, bool]:
    """
    Liefert (mask_low_n, has_low_info).

    Priorität:
      1) explizite low_n Spalte (bool)
      2) n Spalte (n < min_n)
      3) fallback: keine Infos -> alles False
    """
    if low_n_col in d.columns:
        m = d[low_n_col].astype(bool)
        return m.fillna(False), True
    if n_col in d.columns:
        nvals = pd.to_numeric(d[n_col], errors="coerce").fillna(0).astype(int)
        return (nvals < int(min_n)), True
    return pd.Series([False] * len(d), index=d.index), False


# ---------------------------
# Generic Trend Plot
# ---------------------------
def plot_trend_metric(
    tables_dir: Path,
    figures_dir: Path,
    *,
    table_name: str,
    metric_prefix: str,
    y_label: str,
    title: str,
    filename: str,
    logy: bool = False,
    min_n: int = 5,
    hide_low_n: bool = False,
    save_svg: bool = False,
) -> Path:
    df = _load_table(tables_dir, table_name)

    required = {"round", "task", f"{metric_prefix}_median", f"{metric_prefix}_q25", f"{metric_prefix}_q75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in {table_name}.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")
    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    # Low-n-Info vorhanden?
    any_low_info = (f"{metric_prefix}_low_n" in df.columns) or (f"{metric_prefix}_n" in df.columns)
    title_suffix = ""
    if any_low_info:
        title_suffix = f" (Low-n n<{min_n} {'ausgeblendet' if hide_low_n else 'markiert'})"

    for task, d in df.groupby("task"):
        d = d.copy()

        mask_low, has_low_info_task = _low_n_mask(
            d, low_n_col=f"{metric_prefix}_low_n", n_col=f"{metric_prefix}_n", min_n=min_n
        )

        if hide_low_n and has_low_info_task:
            d = d.loc[~mask_low].copy()
            mask_low = pd.Series([False] * len(d), index=d.index)

        if d.empty:
            continue

        x = d["round"].astype(str).tolist()
        y = pd.to_numeric(d[f"{metric_prefix}_median"], errors="coerce").astype(float).tolist()

        line = ax.plot(x, y, marker="o", label=str(task))[0]
        color = line.get_color()

        q25 = pd.to_numeric(d[f"{metric_prefix}_q25"], errors="coerce").astype(float).tolist()
        q75 = pd.to_numeric(d[f"{metric_prefix}_q75"], errors="coerce").astype(float).tolist()
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

        # Low-n Marker (hollow), nur wenn nicht ausgeblendet
        if not hide_low_n and has_low_info_task:
            if mask_low.any():
                ax.scatter(
                    d.loc[mask_low, "round"].astype(str),
                    pd.to_numeric(d.loc[mask_low, f"{metric_prefix}_median"], errors="coerce").astype(float),
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )

    ax.set_title(title + title_suffix)
    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", loc="best")

    if logy:
        _safe_set_logy(
            ax,
            [
                pd.to_numeric(df[f"{metric_prefix}_q25"], errors="coerce"),
                pd.to_numeric(df[f"{metric_prefix}_q75"], errors="coerce"),
                pd.to_numeric(df[f"{metric_prefix}_median"], errors="coerce"),
            ],
            context=metric_prefix,
        )

    _save_figure(fig, out_path, save_svg=save_svg)
    plt.close(fig)

    print(f"[figure] {metric_prefix} trend -> {out_path}")
    return out_path


# ---------------------------
# Specific plots (thin wrappers)
# ---------------------------
def plot_latency_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_latency_us.png",
    logy: bool = False,
    min_n: int = 5,
    hide_low_n: bool = False,
    save_svg: bool = False,
) -> Path:
    return plot_trend_metric(
        tables_dir,
        figures_dir,
        table_name="trend_latency_us_round_task",
        metric_prefix="latency_us",
        y_label="latency_us (µs)",
        title="Trend: Latenz (Median) pro Round × Task",
        filename=filename,
        logy=logy,
        min_n=min_n,
        hide_low_n=hide_low_n,
        save_svg=save_svg,
    )


def plot_energy_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_energy_uj.png",
    logy: bool = False,
    min_n: int = 5,
    hide_low_n: bool = False,
    save_svg: bool = False,
) -> Path:
    return plot_trend_metric(
        tables_dir,
        figures_dir,
        table_name="trend_energy_uj_round_task",
        metric_prefix="energy_uj",
        y_label="energy_uj (µJ)",
        title="Trend: Energie (Median) pro Round × Task",
        filename=filename,
        logy=logy,
        min_n=min_n,
        hide_low_n=hide_low_n,
        save_svg=save_svg,
    )


def plot_unknown_share_by_round(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "unknown_share_by_round.png",
    save_svg: bool = False,
) -> Path:
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
    y = pd.to_numeric(df["share_unknown"], errors="coerce").fillna(0.0).tolist()

    ax.bar(x, y)
    ax.set_title("Datenqualität: Anteil UNKNOWN (task/model nicht zuordenbar)")
    ax.set_xlabel("Round")
    ax.set_ylabel("share_unknown")
    ax.set_ylim(0, max(0.05, max(y) * 1.1 if y else 0.05))
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    _save_figure(fig, out_path, save_svg=save_svg)
    plt.close(fig)

    print(f"[figure] unknown share -> {out_path}")
    return out_path


def run_all(tables_dir: Path, figures_dir: Path, *, logy: bool, min_n: int, hide_low_n: bool, save_svg: bool) -> None:
    plot_latency_trend(tables_dir, figures_dir, logy=logy, min_n=min_n, hide_low_n=hide_low_n, save_svg=save_svg)
    plot_energy_trend(tables_dir, figures_dir, logy=logy, min_n=min_n, hide_low_n=hide_low_n, save_svg=save_svg)
    plot_unknown_share_by_round(tables_dir, figures_dir, save_svg=save_svg)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate plots from reports/tables CSVs.")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR_DEFAULT, help="Path to reports/tables")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT, help="Path to reports/figures")

    parser.add_argument("--latency", action="store_true", help="Generate latency trend plot")
    parser.add_argument("--energy", action="store_true", help="Generate energy trend plot")
    parser.add_argument("--unknown", action="store_true", help="Generate unknown share plot")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if no flags set)")

    parser.add_argument("--logy", action="store_true", help="Use logarithmic y-axis for trend plots")
    parser.add_argument("--min-n", type=int, default=5, help="Low-n threshold (n < MIN_N)")
    parser.add_argument("--hide-low-n", action="store_true", help="Hide low-n points instead of marking them")
    parser.add_argument("--svg", action="store_true", help="Also export SVG (in addition to PNG)")

    args = parser.parse_args(argv)

    flags = [args.latency, args.energy, args.unknown, args.all]
    if not any(flags) or args.all:
        run_all(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            hide_low_n=args.hide_low_n,
            save_svg=args.svg,
        )
        return

    if args.latency:
        plot_latency_trend(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            hide_low_n=args.hide_low_n,
            save_svg=args.svg,
        )
    if args.energy:
        plot_energy_trend(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            hide_low_n=args.hide_low_n,
            save_svg=args.svg,
        )
    if args.unknown:
        plot_unknown_share_by_round(args.tables_dir, args.figures_dir, save_svg=args.svg)


if __name__ == "__main__":
    main()
