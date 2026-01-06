from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "reports" / "figures"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_round_value(r: object) -> float:
    """
    Rounds kommen typischerweise als 'v0.5', 'v1.3' etc.
    Fallback: sehr groß, damit Unbekanntes ans Ende sortiert.
    """
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return 1e9
    s = str(r).strip()
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
    out["_round_order"] = out[round_col].map(_parse_round_value)
    out = out.sort_values(["_round_order", round_col]).drop(columns=["_round_order"])
    return out


def _load_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tabelle nicht gefunden: {path}")
    df = pd.read_csv(path)
    return df


def _save_figure(fig: plt.Figure, out_path: Path, *, save_svg: bool) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"))


def _safe_set_logy(ax: plt.Axes, series_list: list[pd.Series], *, context: str) -> bool:
    """
    Setzt y-Achse auf log, wenn alle übergebenen Serien ausschließlich > 0 sind.
    Gibt True zurück, wenn log gesetzt wurde, sonst False (mit Warnung).
    """
    for s in series_list:
        if s is None or s.empty:
            continue
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if s_num.empty:
            continue
        if (s_num <= 0).any():
            print(f"[warn] logy deaktiviert ({context}) – nicht-positive Werte vorhanden.")
            return False
    ax.set_yscale("log")
    return True


def _has_enough_points(df: pd.DataFrame, *, group_cols: list[str], min_points_per_series: int) -> bool:
    """
    Prüft, ob pro Serie (z. B. Task) mindestens min_points_per_series Punkte existieren.
    """
    if df.empty:
        return False
    if not group_cols:
        return len(df) >= min_points_per_series
    counts = df.groupby(group_cols).size()
    return bool((counts >= min_points_per_series).any())


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

def plot_latency_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_latency_us.png",
    logy: bool = False,
    min_n: int = 5,
    min_points_per_task: int = 2,
    save_svg: bool = False,
) -> Optional[Path]:
    """
    Plottet Median (und IQR-Band) von latency_us pro round×task aus:
      - reports/tables/trend_latency_us_round_task.csv

    Guardrails:
      - Skip bei leerer Tabelle
      - Skip wenn pro Task weniger als min_points_per_task Punkte vorhanden sind

    Optional:
      - logy=True: logarithmische y-Achse (nur wenn alle Werte > 0 sind)
      - min_n: Punkte mit n < min_n (falls *_n vorhanden) werden als hollow markers markiert
      - save_svg=True: zusätzlich SVG exportieren
    """
    df = _load_table(tables_dir, "trend_latency_us_round_task")
    if df.empty:
        print("[skip] latency trend – Tabelle leer.")
        return None

    required = {"round", "task", "latency_us_median", "latency_us_q25", "latency_us_q75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in trend_latency_us_round_task.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")

    if not _has_enough_points(df, group_cols=["task"], min_points_per_series=int(min_points_per_task)):
        print(f"[skip] latency trend – zu wenige Punkte pro Task (min_points_per_task={min_points_per_task}).")
        return None

    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    has_n = "latency_us_n" in df.columns
    title_suffix = f" (hollow markers: n<{min_n})" if has_n else ""

    for task, d in df.groupby("task", dropna=False):
        x = d["round"].astype(str).tolist()
        y = pd.to_numeric(d["latency_us_median"], errors="coerce")

        line = ax.plot(x, y, marker="o", label=str(task))[0]
        color = line.get_color()

        q25 = pd.to_numeric(d["latency_us_q25"], errors="coerce")
        q75 = pd.to_numeric(d["latency_us_q75"], errors="coerce")
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

        if has_n:
            nvals = pd.to_numeric(d["latency_us_n"], errors="coerce")
            mask_low = nvals < int(min_n)
            if mask_low.any():
                ax.scatter(
                    d.loc[mask_low, "round"].astype(str),
                    pd.to_numeric(d.loc[mask_low, "latency_us_median"], errors="coerce"),
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )

    ax.set_title("Trend: Latenz (Median) pro Round × Task" + title_suffix)
    ax.set_xlabel("Round")
    ax.set_ylabel("latency_us (µs)")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", loc="best")

    if logy:
        _safe_set_logy(
            ax,
            [
                pd.to_numeric(df["latency_us_q25"], errors="coerce"),
                pd.to_numeric(df["latency_us_q75"], errors="coerce"),
                pd.to_numeric(df["latency_us_median"], errors="coerce"),
            ],
            context="latency_us",
        )

    _save_figure(fig, out_path, save_svg=save_svg)
    plt.close(fig)

    print(f"[figure] latency trend -> {out_path}")
    return out_path


def plot_energy_trend(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "trend_energy_uj.png",
    logy: bool = False,
    min_n: int = 5,
    min_points_per_task: int = 2,
    save_svg: bool = False,
) -> Optional[Path]:
    """
    Plottet Median (und IQR-Band) von energy_uj pro round×task aus:
      - reports/tables/trend_energy_uj_round_task.csv

    Guardrails:
      - Skip bei leerer Tabelle
      - Skip wenn pro Task weniger als min_points_per_task Punkte vorhanden sind

    Optional:
      - logy=True: logarithmische y-Achse (nur wenn alle Werte > 0 sind)
      - min_n: Punkte mit n < min_n (falls *_n vorhanden) werden als hollow markers markiert
      - save_svg=True: zusätzlich SVG exportieren
    """
    df = _load_table(tables_dir, "trend_energy_uj_round_task")
    if df.empty:
        print("[skip] energy trend – Tabelle leer.")
        return None

    required = {"round", "task", "energy_uj_median", "energy_uj_q25", "energy_uj_q75"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in trend_energy_uj_round_task.csv: {sorted(missing)}")

    df = _sort_rounds(df, "round")

    if not _has_enough_points(df, group_cols=["task"], min_points_per_series=int(min_points_per_task)):
        print(f"[skip] energy trend – zu wenige Punkte pro Task (min_points_per_task={min_points_per_task}).")
        return None

    _ensure_dir(figures_dir)
    out_path = figures_dir / filename

    fig, ax = plt.subplots()

    has_n = "energy_uj_n" in df.columns
    title_suffix = f" (hollow markers: n<{min_n})" if has_n else ""

    for task, d in df.groupby("task", dropna=False):
        x = d["round"].astype(str).tolist()
        y = pd.to_numeric(d["energy_uj_median"], errors="coerce")

        line = ax.plot(x, y, marker="o", label=str(task))[0]
        color = line.get_color()

        q25 = pd.to_numeric(d["energy_uj_q25"], errors="coerce")
        q75 = pd.to_numeric(d["energy_uj_q75"], errors="coerce")
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

        if has_n:
            nvals = pd.to_numeric(d["energy_uj_n"], errors="coerce")
            mask_low = nvals < int(min_n)
            if mask_low.any():
                ax.scatter(
                    d.loc[mask_low, "round"].astype(str),
                    pd.to_numeric(d.loc[mask_low, "energy_uj_median"], errors="coerce"),
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )

    ax.set_title("Trend: Energie (Median) pro Round × Task" + title_suffix)
    ax.set_xlabel("Round")
    ax.set_ylabel("energy_uj (µJ)")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", loc="best")

    if logy:
        _safe_set_logy(
            ax,
            [
                pd.to_numeric(df["energy_uj_q25"], errors="coerce"),
                pd.to_numeric(df["energy_uj_q75"], errors="coerce"),
                pd.to_numeric(df["energy_uj_median"], errors="coerce"),
            ],
            context="energy_uj",
        )

    _save_figure(fig, out_path, save_svg=save_svg)
    plt.close(fig)

    print(f"[figure] energy trend -> {out_path}")
    return out_path


def plot_unknown_share_by_round(
    tables_dir: Path,
    figures_dir: Path,
    *,
    filename: str = "unknown_share_by_round.png",
    save_svg: bool = False,
) -> Optional[Path]:
    """
    Plottet share_unknown pro Round aus:
      - reports/tables/unknown_summary_by_round.csv

    Guardrails:
      - Skip bei leerer Tabelle
    """
    df = _load_table(tables_dir, "unknown_summary_by_round")
    if df.empty:
        print("[skip] unknown share – Tabelle leer.")
        return None

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

    # Y-Limit sinnvoll setzen: mind. 0.05, sonst 10% Puffer über max
    ymax = max(y) if y else 0.0
    ax.set_ylim(0, max(0.05, ymax * 1.1 if ymax > 0 else 0.05))

    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    _save_figure(fig, out_path, save_svg=save_svg)
    plt.close(fig)

    print(f"[figure] unknown share -> {out_path}")
    return out_path


def run_all(tables_dir: Path, figures_dir: Path, *, logy: bool, min_n: int, min_points_per_task: int, save_svg: bool) -> None:
    plot_latency_trend(
        tables_dir,
        figures_dir,
        logy=logy,
        min_n=min_n,
        min_points_per_task=min_points_per_task,
        save_svg=save_svg,
    )
    plot_energy_trend(
        tables_dir,
        figures_dir,
        logy=logy,
        min_n=min_n,
        min_points_per_task=min_points_per_task,
        save_svg=save_svg,
    )
    plot_unknown_share_by_round(tables_dir, figures_dir, save_svg=save_svg)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate plots from reports/tables CSVs.")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR_DEFAULT, help="Pfad zu reports/tables")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT, help="Pfad zu reports/figures")

    parser.add_argument("--latency", action="store_true", help="Latenz-Trend plotten")
    parser.add_argument("--energy", action="store_true", help="Energie-Trend plotten")
    parser.add_argument("--unknown", action="store_true", help="UNKNOWN-Anteil plotten")
    parser.add_argument("--all", action="store_true", help="Alle Plots erzeugen (Default wenn keine Flags gesetzt)")

    parser.add_argument("--logy", action="store_true", help="Trendplots mit logarithmischer y-Achse")
    parser.add_argument("--min-n", type=int, default=5, help="Punkte mit n < MIN_N als hollow markers markieren")
    parser.add_argument("--min-points-per-task", type=int, default=2, help="Mindestens so viele Punkte pro Task für Plot")
    parser.add_argument("--svg", action="store_true", help="Zusätzlich SVG exportieren (neben PNG)")

    args = parser.parse_args(argv)

    flags = [args.latency, args.energy, args.unknown, args.all]
    if not any(flags) or args.all:
        run_all(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            min_points_per_task=args.min_points_per_task,
            save_svg=args.svg,
        )
        return

    if args.latency:
        plot_latency_trend(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            min_points_per_task=args.min_points_per_task,
            save_svg=args.svg,
        )
    if args.energy:
        plot_energy_trend(
            args.tables_dir,
            args.figures_dir,
            logy=args.logy,
            min_n=args.min_n,
            min_points_per_task=args.min_points_per_task,
            save_svg=args.svg,
        )
    if args.unknown:
        plot_unknown_share_by_round(args.tables_dir, args.figures_dir, save_svg=args.svg)


if __name__ == "__main__":
    # Wichtig: bevorzugt als Modul ausführen: python -m src.plots
    main()
