"""
Experiment D: Convergence + Quality Plots.

Reads: results/experiment_d/<dataset>/results.json

Per dataset:
  1. convergence_by_k.pdf   -- one subplot per k; each shows SGD-cycle vs pivot-lazy vs pivot-euc
  2. final_stress_box.pdf   -- final stress box plots grouped by k, colored by algo

Combined (3-column grid):
  _plots_combined/combined_stress_ratio.pdf  -- stress ratio (pivot / cycle) vs k, one panel per dataset
  _plots_combined/combined_convergence_<k_idx>.pdf  -- convergence at each k fraction

Usage:
    python plot_experiment_d.py                    # all datasets with results
    python plot_experiment_d.py coil20 s_curve
"""

from __future__ import annotations

import sys
import math
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_d"

PIVOT_ALGO = "SGD-pivot-euc"

ALGO_LABEL = {
    "SGD-cycle":     "SGD-cycle (euclidean)",
    "SGD-pivot-euc": "Pivot-euclidean",
}
ALGO_COLOR = {
    "SGD-cycle":     "#2ca02c",
    "SGD-pivot-euc": "#ff7f0e",
}

N_INTERP = 200

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def get_all_datasets() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "results.json").exists()
    )


def load(dataset: str) -> dict:
    path = RESULTS_DIR / dataset / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# SUMMARIZATION
# ──────────────────────────────────────────────

def _interp_history(times, stresses, t_grid):
    return np.interp(t_grid, np.asarray(times), np.asarray(stresses))


def convergence_summary(data: dict) -> dict:
    """Returns {(algo, k): {t_grid, stress_median, stress_q25, stress_q75}}."""
    buckets: dict[tuple, list] = {}
    for run in data["runs"]:
        key = (run["algo"], run.get("k"))
        if run.get("history"):
            buckets.setdefault(key, []).append(run["history"])

    result = {}
    for key, histories in buckets.items():
        t_max = max(h[-1][0] for h in histories if h)
        if t_max == 0:
            continue
        t_grid = np.linspace(0, t_max, N_INTERP)
        interped = np.stack([
            _interp_history([p[0] for p in h], [p[1] for p in h], t_grid)
            for h in histories
        ])
        result[key] = {
            "t_grid":        t_grid,
            "stress_median": np.median(interped, axis=0),
            "stress_q25":    np.percentile(interped, 25, axis=0),
            "stress_q75":    np.percentile(interped, 75, axis=0),
        }
    return result


def final_stress_summary(data: dict) -> dict:
    """Returns {(algo, k): np.ndarray of final stresses across seeds}."""
    buckets: dict[tuple, list] = {}
    for run in data["runs"]:
        key = (run["algo"], run.get("k"))
        buckets.setdefault(key, []).append(run["stress"])
    return {k: np.array(v) for k, v in buckets.items()}


# ──────────────────────────────────────────────
# SAVE HELPER
# ──────────────────────────────────────────────

def save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)


# ──────────────────────────────────────────────
# PER-DATASET PLOTS
# ──────────────────────────────────────────────

def plot_convergence_by_k(conv: dict, data: dict, dataset: str, out_dir: Path) -> None:
    """One subplot per k; each panel: SGD-cycle + pivot-lazy + pivot-euc."""
    k_values = data["k_values"]
    n = data["n_samples"]
    ncols = min(len(k_values), 3)
    nrows = math.ceil(len(k_values) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)

    cycle_key = ("SGD-cycle", None)

    for idx, k in enumerate(k_values):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Baseline
        if cycle_key in conv:
            c = conv[cycle_key]
            ax.plot(c["t_grid"], c["stress_median"], color=ALGO_COLOR["SGD-cycle"],
                    linewidth=2, label=ALGO_LABEL["SGD-cycle"])
            ax.fill_between(c["t_grid"], c["stress_q25"], c["stress_q75"],
                            alpha=0.15, color=ALGO_COLOR["SGD-cycle"])

        key = (PIVOT_ALGO, k)
        if key in conv:
            c = conv[key]
            ax.plot(c["t_grid"], c["stress_median"], color=ALGO_COLOR[PIVOT_ALGO],
                    linewidth=2, label=ALGO_LABEL[PIVOT_ALGO])
            ax.fill_between(c["t_grid"], c["stress_q25"], c["stress_q75"],
                            alpha=0.15, color=ALGO_COLOR[PIVOT_ALGO])

        ax.set_title(f"k={k} ({k/n:.0%} of N)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stress" if col == 0 else "")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    for idx in range(len(k_values), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.5, wspace=0.25)
    fig.suptitle(f"{dataset} (N={n:,}): Convergence by k", fontsize=13, y=0.97)
    save_fig(fig, out_dir / "convergence_by_k")


def plot_final_stress_box(finals: dict, data: dict, dataset: str, out_dir: Path) -> None:
    """Box plots: x-axis = k value, one box per algo per k, plus cycle baseline."""
    k_values = data["k_values"]
    n = data["n_samples"]
    width = 0.5
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(k_values)), 5))

    # Cycle baseline as a horizontal band
    cycle_vals = finals.get(("SGD-cycle", None), np.array([]))
    if len(cycle_vals):
        ax.axhline(np.median(cycle_vals), color=ALGO_COLOR["SGD-cycle"],
                   linestyle="--", linewidth=1.8, label=ALGO_LABEL["SGD-cycle"], zorder=3)
        ax.fill_between([-0.5, len(k_values) - 0.5],
                        np.percentile(cycle_vals, 25), np.percentile(cycle_vals, 75),
                        alpha=0.10, color=ALGO_COLOR["SGD-cycle"])

    x_pos = np.arange(len(k_values))
    boxes = [finals.get((PIVOT_ALGO, k), np.array([])) for k in k_values]
    boxes = [b for b in boxes if len(b)]
    if boxes:
        bp = ax.boxplot(boxes, positions=x_pos[:len(boxes)], widths=width,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(ALGO_COLOR[PIVOT_ALGO])
            patch.set_alpha(0.6)
        bp["boxes"][0].set_label(ALGO_LABEL[PIVOT_ALGO])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"k={k}\n({k/n:.0%})" for k in k_values])
    ax.set_ylabel("Final stress")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{dataset} (N={n:,}): Final stress by k")
    save_fig(fig, out_dir / "final_stress_box")


# ──────────────────────────────────────────────
# COMBINED GRIDS
# ──────────────────────────────────────────────

def _make_grid(n: int, ncols: int = 3):
    return math.ceil(n / ncols), ncols


def plot_combined_stress_ratio(
    datasets: list[str],
    finals_all: dict,
    metas: dict,
    out_dir: Path,
) -> None:
    """Stress ratio (pivot / cycle median) vs k, one panel per dataset."""
    nrows, ncols = _make_grid(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        finals = finals_all[dataset]
        n, k_values = metas[dataset]

        cycle_median = np.median(finals.get(("SGD-cycle", None), np.array([np.nan])))

        ks, ratios, q25s, q75s = [], [], [], []
        for k in k_values:
            vals = finals.get((PIVOT_ALGO, k), np.array([]))
            if not len(vals):
                continue
            ks.append(k)
            ratios.append(np.median(vals) / cycle_median)
            q25s.append(np.percentile(vals, 25) / cycle_median)
            q75s.append(np.percentile(vals, 75) / cycle_median)
        if ks:
            c = ALGO_COLOR[PIVOT_ALGO]
            ax.plot(ks, ratios, marker="o", linewidth=2, color=c, label=ALGO_LABEL[PIVOT_ALGO])
            ax.fill_between(ks, q25s, q75s, alpha=0.15, color=c)

        ax.axhline(1.0, color=ALGO_COLOR["SGD-cycle"], linestyle="--",
                   linewidth=1.6, label="SGD-cycle (ratio=1.0)", zorder=3)
        ax.set_title(f"{dataset} (N={n:,})")
        ax.set_xlabel("k (number of pivots)")
        ax.set_ylabel("Stress ratio" if col == 0 else "")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(top=0.94, bottom=0.04, hspace=0.55, wspace=0.25)
    fig.suptitle("Experiment D: Stress ratio vs k  (pivot / SGD-cycle)", fontsize=14, y=0.97)
    save_fig(fig, out_dir / "combined_stress_ratio")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main(datasets: list[str]) -> None:
    finals_all: dict = {}
    metas:      dict = {}

    for dataset in datasets:
        print(f"Processing {dataset}...")
        data   = load(dataset)
        n      = data["n_samples"]
        k_vals = data["k_values"]

        conv   = convergence_summary(data)
        finals = final_stress_summary(data)

        finals_all[dataset] = finals
        metas[dataset]      = (n, k_vals)

        out_dir = RESULTS_DIR / dataset
        plot_convergence_by_k(conv, data, dataset, out_dir)
        plot_final_stress_box(finals, data, dataset, out_dir)
        print(f"  Saved per-dataset plots → {out_dir}")

    combined_dir = RESULTS_DIR / "_plots_combined"
    plot_combined_stress_ratio(datasets, finals_all, metas, combined_dir)
    print(f"  Saved combined plots → {combined_dir}")


if __name__ == "__main__":
    targets = sys.argv[1:] or get_all_datasets()
    if not targets:
        print("No results found. Run run_experiment_d.py first.")
        sys.exit(0)
    main(targets)
