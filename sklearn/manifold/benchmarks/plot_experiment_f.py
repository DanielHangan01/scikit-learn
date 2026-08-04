"""
Experiment F: Pivot k-Sensitivity Sweep (Plotting).

Reads:
  results/experiment_f/k_sweep/<dataset>/results.csv

Generates per dataset:
  1. stress_vs_k.pdf/png        -- absolute stress vs k, SGD-cycle baseline
  2. time_vs_k.pdf/png          -- runtime vs k, SGD-cycle baseline
  3. stress_ratio_vs_k.pdf/png  -- stress / SGD-cycle-stress vs k (quality gap)

And a combined figure with both panels side-by-side across all datasets.

Usage:
    python plot_experiment_f.py                   # all DEFAULT_DATASETS
    python plot_experiment_f.py coil20 s_curve
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results" / "experiment_f" / "k_sweep"

N_INTERP = 200


def get_all_datasets() -> list[str]:
    """Discover all datasets that have results under RESULTS_DIR."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "results.csv").exists()
    )

ALGO_ORDER = ["SGD-pivot-lazy", "SGD-pivot-euclidean"]
ALGO_LABEL = {
    "SGD-pivot-lazy":       "Pivot-lazy",
    "SGD-pivot-euclidean":  "Pivot-euclidean",
}
ALGO_COLOR = {
    "SGD-pivot-lazy":       "#1f77b4",
    "SGD-pivot-euclidean":  "#ff7f0e",
}
BASELINE_COLOR = "#2ca02c"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def load(dataset: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    df = pd.read_csv(path)
    df["k"] = df["k"].astype(int)
    df["stress"] = df["stress"].astype(float)
    df["time_solver"] = df["time_solver"].astype(float)
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (pivot_summary_df, baseline_dict).

    baseline_dict keys: stress_median, stress_q25, stress_q75,
                        time_median, time_q25, time_q75
    """
    baseline_rows = df[df["algo"] == "SGD-cycle"]
    baseline = {
        "stress_median": baseline_rows["stress"].median(),
        "stress_q25":    baseline_rows["stress"].quantile(0.25),
        "stress_q75":    baseline_rows["stress"].quantile(0.75),
        "time_median":   baseline_rows["time_solver"].median(),
        "time_q25":      baseline_rows["time_solver"].quantile(0.25),
        "time_q75":      baseline_rows["time_solver"].quantile(0.75),
    }

    pivot_df = df[df["algo"].isin(ALGO_ORDER)].copy()
    g = pivot_df.groupby(["algo", "k"])
    summary = g.agg(
        stress_median=("stress", "median"),
        stress_q25=("stress", lambda x: x.quantile(0.25)),
        stress_q75=("stress", lambda x: x.quantile(0.75)),
        time_median=("time_solver", "median"),
        time_q25=("time_solver", lambda x: x.quantile(0.25)),
        time_q75=("time_solver", lambda x: x.quantile(0.75)),
    ).reset_index()

    summary["stress_ratio"] = summary["stress_median"] / baseline["stress_median"]
    summary["stress_ratio_q25"] = summary["stress_q25"] / baseline["stress_median"]
    summary["stress_ratio_q75"] = summary["stress_q75"] / baseline["stress_median"]

    return summary, baseline


def save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)


def _draw_baseline_hline(ax, value, label, color=BASELINE_COLOR, linestyle="--"):
    ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.6, label=label, zorder=3)


# ──────────────────────────────────────────────
# PER-DATASET PLOTS
# ──────────────────────────────────────────────
def plot_stress_vs_k(summary: pd.DataFrame, baseline: dict, dataset: str, n: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGO_ORDER:
        s = summary[summary["algo"] == algo].sort_values("k")
        if s.empty:
            continue
        k   = s["k"].to_numpy()
        med = s["stress_median"].to_numpy()
        q25 = s["stress_q25"].to_numpy()
        q75 = s["stress_q75"].to_numpy()
        c = ALGO_COLOR[algo]
        ax.plot(k, med, marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
        ax.fill_between(k, q25, q75, alpha=0.15, color=c)

    _draw_baseline_hline(
        ax,
        baseline["stress_median"],
        label=f"SGD-cycle (full, median={baseline['stress_median']:.2e})",
    )
    ax.fill_between(
        ax.get_xlim(),
        baseline["stress_q25"], baseline["stress_q75"],
        alpha=0.10, color=BASELINE_COLOR,
    )

    ax.set_title(f"{dataset} (N={n:,}): Stress vs pivot count k")
    ax.set_xlabel("k (number of pivots)")
    ax.set_ylabel("Final stress (median ± IQR)")
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "stress_vs_k")


def plot_time_vs_k(summary: pd.DataFrame, baseline: dict, dataset: str, n: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGO_ORDER:
        s = summary[summary["algo"] == algo].sort_values("k")
        if s.empty:
            continue
        k   = s["k"].to_numpy()
        med = s["time_median"].to_numpy()
        q25 = s["time_q25"].to_numpy()
        q75 = s["time_q75"].to_numpy()
        c = ALGO_COLOR[algo]
        ax.plot(k, med, marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
        ax.fill_between(k, q25, q75, alpha=0.15, color=c)

    _draw_baseline_hline(
        ax,
        baseline["time_median"],
        label=f"SGD-cycle (full, median={baseline['time_median']:.2f}s)",
    )

    ax.set_title(f"{dataset} (N={n:,}): Runtime vs pivot count k")
    ax.set_xlabel("k (number of pivots)")
    ax.set_ylabel("Runtime (s, median ± IQR)")
    ax.legend(loc="upper left")
    save_fig(fig, out_dir / "time_vs_k")


def plot_stress_ratio_vs_k(summary: pd.DataFrame, dataset: str, n: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGO_ORDER:
        s = summary[summary["algo"] == algo].sort_values("k")
        if s.empty:
            continue
        k   = s["k"].to_numpy()
        rat = s["stress_ratio"].to_numpy()
        lo  = s["stress_ratio_q25"].to_numpy()
        hi  = s["stress_ratio_q75"].to_numpy()
        c = ALGO_COLOR[algo]
        ax.plot(k, rat, marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
        ax.fill_between(k, lo, hi, alpha=0.15, color=c)

    ax.axhline(1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1.6,
               label="SGD-cycle baseline (ratio = 1.0)", zorder=3)

    ax.set_title(f"{dataset} (N={n:,}): Stress ratio vs k  (pivot / full-cycle)")
    ax.set_xlabel("k (number of pivots)")
    ax.set_ylabel("Stress ratio (1.0 = full-cycle quality)")
    ax.legend(loc="upper right")
    save_fig(fig, out_dir / "stress_ratio_vs_k")


# ──────────────────────────────────────────────
# COMBINED FIGURES  (3-column vertical grid, one figure per metric)
# ──────────────────────────────────────────────
import math

def _make_combined_grid(n_datasets: int, ncols: int = 3):
    """Return (nrows, ncols) for a ceil(n/ncols) × ncols grid."""
    nrows = math.ceil(n_datasets / ncols)
    return nrows, ncols


def plot_combined(datasets: list[str], summaries: dict, baselines: dict, ns: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_combined_stress_ratio(datasets, summaries, ns, out_dir)
    _plot_combined_stress_absolute(datasets, summaries, baselines, ns, out_dir)
    _plot_combined_runtime(datasets, summaries, baselines, ns, out_dir)


def _plot_combined_stress_absolute(
    datasets: list[str], summaries: dict, baselines: dict, ns: dict, out_dir: Path
) -> None:
    nrows, ncols = _make_combined_grid(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        summary  = summaries[dataset]
        baseline = baselines[dataset]

        for algo in ALGO_ORDER:
            s = summary[summary["algo"] == algo].sort_values("k")
            if s.empty:
                continue
            c = ALGO_COLOR[algo]
            ax.plot(s["k"].to_numpy(), s["stress_median"].to_numpy(),
                    marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
            ax.fill_between(s["k"].to_numpy(),
                            s["stress_q25"].to_numpy(),
                            s["stress_q75"].to_numpy(),
                            alpha=0.15, color=c)
        ax.axhline(baseline["stress_median"], color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.6, zorder=3,
                   label=f"SGD-cycle ({baseline['stress_median']:.2e})")
        ax.fill_between(ax.get_xlim(),
                        baseline["stress_q25"], baseline["stress_q75"],
                        alpha=0.10, color=BASELINE_COLOR)
        ax.set_title(f"{dataset} (N={ns[dataset]:,})")
        ax.set_xlabel("k (number of pivots)")
        ax.set_ylabel("Final stress" if col == 0 else "")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.04, hspace=0.55, wspace=0.25)
    fig.suptitle("Experiment F: Absolute stress vs k", fontsize=14, y=0.97)
    save_fig(fig, out_dir / "combined_stress_absolute")


# ──────────────────────────────────────────────
# CONVERGENCE (per-epoch histories)
# ──────────────────────────────────────────────
def load_histories(dataset: str) -> dict:
    path = RESULTS_DIR / dataset / "histories.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _interp_history(times, stresses, t_grid):
    return np.interp(t_grid, np.asarray(times), np.asarray(stresses))


def convergence_summary(histories: dict, *, normalize_by_initial: bool = False) -> dict:
    """Group histories by (algo, k) and return median + IQR on a common time grid.

    When ``normalize_by_initial`` is True, each replicate's stress trajectory is
    divided by *its own* initial (t=0) stress before interpolation. Useful when
    absolute final stresses misrepresent optimisation progress relative to the
    scale of the problem.
    """
    buckets: dict[tuple, list] = {}
    for rec in histories.values():
        h = rec.get("history")
        if not h:
            continue
        if normalize_by_initial:
            init = float(h[0][1])
            if init <= 0:
                continue
            h = [[t, s / init] for t, s in h]
        buckets.setdefault((rec["algo"], int(rec["k"])), []).append(h)

    result: dict[tuple, dict] = {}
    for key, hist_list in buckets.items():
        t_max = max(h[-1][0] for h in hist_list if h)
        if t_max <= 0:
            continue
        t_grid = np.linspace(0, t_max, N_INTERP)
        interped = np.stack([
            _interp_history([p[0] for p in h], [p[1] for p in h], t_grid)
            for h in hist_list
        ])
        result[key] = {
            "t_grid":        t_grid,
            "stress_median": np.median(interped, axis=0),
            "stress_q25":    np.percentile(interped, 25, axis=0),
            "stress_q75":    np.percentile(interped, 75, axis=0),
        }
    return result


def plot_convergence_by_k(
    conv: dict, baseline: dict, dataset: str, n: int,
    k_values: list[int], out_dir: Path, *, relative: bool,
) -> None:
    """One subplot per k; curves are the pivot variants + SGD-cycle baseline.

    The supplied ``conv`` is expected to already be in the right form: absolute
    stresses when ``relative=False``, per-replicate stress / initial-stress
    ratios when ``relative=True``. The cycle baseline is shown as a horizontal
    line at cycle's final value (absolute) or cycle's final ratio (relative).
    """
    cycle_key = ("SGD-cycle", n)
    if relative:
        if cycle_key in conv:
            cycle_ref = float(conv[cycle_key]["stress_median"][-1])
            cycle_label = f"SGD-cycle (final/init = {cycle_ref:.3f})"
        else:
            cycle_ref = None
            cycle_label = None
    else:
        cycle_ref = float(baseline["stress_median"])
        cycle_label = f"SGD-cycle ({cycle_ref:.2e})"

    ncols = min(len(k_values), 3)
    nrows = math.ceil(len(k_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)

    for idx, k in enumerate(k_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        plotted_anything = False
        for algo in ALGO_ORDER:
            key = (algo, k)
            if key not in conv:
                continue
            c = conv[key]
            color = ALGO_COLOR[algo]
            ax.plot(c["t_grid"], c["stress_median"],
                    color=color, linewidth=2, label=ALGO_LABEL[algo])
            ax.fill_between(c["t_grid"],
                            c["stress_q25"], c["stress_q75"],
                            alpha=0.15, color=color)
            plotted_anything = True

        if cycle_ref is not None:
            ax.axhline(cycle_ref, color=BASELINE_COLOR, linestyle="--",
                       linewidth=1.6, zorder=3, label=cycle_label)

        ax.set_title(f"k={k}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(("Stress / initial stress" if relative else "Stress") if col == 0 else "")
        if plotted_anything:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    for idx in range(len(k_values), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    mode = "relative" if relative else "absolute"
    fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.5, wspace=0.25)
    fig.suptitle(f"{dataset} (N={n:,}): Convergence by k ({mode})",
                 fontsize=13, y=0.97)
    save_fig(fig, out_dir / f"convergence_by_k_{mode}")


def _plot_combined_stress_ratio(
    datasets: list[str], summaries: dict, ns: dict, out_dir: Path
) -> None:
    nrows, ncols = _make_combined_grid(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        summary = summaries[dataset]

        for algo in ALGO_ORDER:
            s = summary[summary["algo"] == algo].sort_values("k")
            if s.empty:
                continue
            c = ALGO_COLOR[algo]
            ax.plot(s["k"].to_numpy(), s["stress_ratio"].to_numpy(),
                    marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
            ax.fill_between(s["k"].to_numpy(),
                            s["stress_ratio_q25"].to_numpy(),
                            s["stress_ratio_q75"].to_numpy(),
                            alpha=0.15, color=c)
        ax.axhline(1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1.6,
                   label="SGD-cycle (ratio = 1.0)", zorder=3)
        ax.set_title(f"{dataset} (N={ns[dataset]:,})")
        ax.set_xlabel("k (number of pivots)")
        ax.set_ylabel("Stress ratio" if col == 0 else "")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    # Hide unused axes
    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.04, hspace=0.55, wspace=0.25)
    fig.suptitle("Experiment F: Stress ratio vs k  (pivot / full-cycle)", fontsize=14, y=0.97)
    save_fig(fig, out_dir / "combined_stress_ratio")


def _plot_combined_runtime(
    datasets: list[str], summaries: dict, baselines: dict, ns: dict, out_dir: Path
) -> None:
    nrows, ncols = _make_combined_grid(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        summary  = summaries[dataset]
        baseline = baselines[dataset]

        for algo in ALGO_ORDER:
            s = summary[summary["algo"] == algo].sort_values("k")
            if s.empty:
                continue
            c = ALGO_COLOR[algo]
            ax.plot(s["k"].to_numpy(), s["time_median"].to_numpy(),
                    marker="o", linewidth=2, color=c, label=ALGO_LABEL[algo])
            ax.fill_between(s["k"].to_numpy(),
                            s["time_q25"].to_numpy(),
                            s["time_q75"].to_numpy(),
                            alpha=0.15, color=c)
        ax.axhline(baseline["time_median"], color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.6, label=f"SGD-cycle ({baseline['time_median']:.2f}s)", zorder=3)
        ax.set_title(f"{dataset} (N={ns[dataset]:,})")
        ax.set_xlabel("k (number of pivots)")
        ax.set_ylabel("Runtime (s)" if col == 0 else "")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.7)

    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    fig.subplots_adjust(top=0.94, bottom=0.04, hspace=0.55, wspace=0.25)
    fig.suptitle("Experiment F: Runtime vs k", fontsize=14, y=0.97)
    save_fig(fig, out_dir / "combined_runtime")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main(datasets: list[str]) -> None:
    summaries: dict = {}
    baselines: dict = {}
    ns: dict = {}

    for dataset in datasets:
        print(f"Processing {dataset}...")
        df = load(dataset)
        n = int(df["n_samples"].iloc[0])
        summary, baseline = summarize(df)
        summaries[dataset] = summary
        baselines[dataset] = baseline
        ns[dataset] = n

        out_dir = RESULTS_DIR / dataset
        plot_stress_vs_k(summary, baseline, dataset, n, out_dir)
        plot_time_vs_k(summary, baseline, dataset, n, out_dir)
        plot_stress_ratio_vs_k(summary, dataset, n, out_dir)

        histories = load_histories(dataset)
        if histories:
            conv_abs = convergence_summary(histories, normalize_by_initial=False)
            conv_rel = convergence_summary(histories, normalize_by_initial=True)
            k_values = sorted(set(
                int(k) for (algo, k) in conv_abs.keys() if algo in ALGO_ORDER
            ))
            if k_values:
                plot_convergence_by_k(conv_abs, baseline, dataset, n, k_values, out_dir,
                                      relative=False)
                plot_convergence_by_k(conv_rel, baseline, dataset, n, k_values, out_dir,
                                      relative=True)
        print(f"  Saved per-dataset plots → {out_dir}")

    combined_dir = RESULTS_DIR / "_plots_combined"
    plot_combined(datasets, summaries, baselines, ns, combined_dir)
    print(f"  Saved combined plots → {combined_dir / 'combined_stress_ratio.*'}, {combined_dir / 'combined_runtime.*'}")


if __name__ == "__main__":
    targets = sys.argv[1:] or get_all_datasets()
    main(targets)
