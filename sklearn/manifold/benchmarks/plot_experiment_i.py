"""
Experiment I plotting: budget knob on sampling_strategy='random'.

Reads:
  results/experiment_i/<dataset>/results.csv

All plots use ratios anchored at SGD-cycle (= 1.0) so cross-dataset
comparison is meaningful and small-but-real differences are visible.

Per dataset (3-col combined grids saved to ``_plots_combined/``):
  A. stress_overhead_vs_budget.{pdf,png}
        x = n_updates_per_epoch (log), y = random_stress / cycle_stress
        (LINEAR, anchored at 1.0). Cycle reference at y=1.
  B. speedup_vs_budget.{pdf,png}
        x = n_updates_per_epoch (log), y = cycle_time / random_time
        (linear). Above 1 = faster than cycle, below = slower.

Cross-dataset (single figures saved to ``_plots_combined/``):
  C. pareto_normalized_all.{pdf,png}
        x = time / cycle_time (log, 1.0 = cycle speed)
        y = stress / cycle_stress (linear, 1.0 = cycle stress)
        All 7 datasets on ONE axes, distinct colours per dataset.
        Solid = max_iter=30, dashed = max_iter=15. Bottom-left of
        (1, 1) is the win quadrant (shaded green).
  D. scaling_vs_N.{pdf,png}
        Two side-by-side panels. Left: speedup vs N (log x) at
        fixed k ∈ {50, 100}, max_iter=30. Right: stress overhead
        vs N at the same k. Direct visualisation of the scaling
        claim ("speedup grows ~linearly with N at fixed k while
        quality stays bounded").

Usage:
    python plot_experiment_i.py                       # all datasets
    python plot_experiment_i.py coil20 hiva           # subset

Old artefact filenames (stress_vs_budget, time_vs_budget,
stress_vs_time) are cleaned up by this script on each run.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = Path(__file__).parent / "results" / "experiment_i"
COMBINED_DIR = RESULTS_DIR / "_plots_combined"

# Plot D — fixed-k scaling story. Two k values shown side-by-side.
FIXED_K_VALUES = [50, 100]
FIXED_MAX_ITER = 30   # cleanest series for the scaling story

# Pareto plot (C) — distinct qualitative colours per dataset.
DATASET_PALETTE = {
    "fmd":            "#1f77b4",  # blue
    "cnae9":          "#ff7f0e",  # orange
    "coil20":         "#2ca02c",  # green
    "fashion_mnist":  "#d62728",  # red
    "hiva":           "#9467bd",  # purple
    "cifar10":        "#8c564b",  # brown
    "epileptic":      "#e377c2",  # pink
}

OLD_PER_DATASET_STEMS = ["stress_vs_budget", "time_vs_budget", "stress_vs_time"]
OLD_COMBINED_STEMS = [
    "combined_stress_vs_budget",
    "combined_time_vs_budget",
    "combined_stress_vs_time",
    "speedup_at_quality_vs_N",   # superseded by scaling_vs_N
]


def get_all_datasets() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "results.csv").exists()
    )


# ──────────────────────────────────────────────
# style
# ──────────────────────────────────────────────

MAX_ITER_COLOR = {
    15: "#1f77b4",   # blue
    30: "#ff7f0e",   # orange
}
MAX_ITER_LABEL = {
    15: "max_iter=15",
    30: "max_iter=30",
}
MAX_ITER_LS = {
    15: "--",
    30: "-",
}
CYCLE_COLOR = "#2ca02c"

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
    "legend.fontsize": 10,
})


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def load(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / dataset / "results.csv")
    df["n_updates_per_epoch"] = df["n_updates_per_epoch"].astype(int)
    df["max_iter"] = df["max_iter"].astype(int)
    df["stress"] = df["stress"].astype(float)
    df["time_solver"] = df["time_solver"].astype(float)
    return df


def cycle_medians(df: pd.DataFrame, max_iter: int) -> Tuple[float, float]:
    """Return ``(cycle_stress_median, cycle_time_median)`` at max_iter."""
    sub = df[(df["phase"] == "base") & (df["max_iter"] == max_iter)]
    if sub.empty:
        return np.nan, np.nan
    return float(sub["stress"].median()), float(sub["time_solver"].median())


def budget_aggregate(df: pd.DataFrame, max_iter: int) -> pd.DataFrame:
    """One row per budget at the given max_iter, with median stress/time
    and IQR bands."""
    sub = df[(df["phase"] == "budget") & (df["max_iter"] == max_iter)]
    if sub.empty:
        return sub
    g = sub.groupby("n_updates_per_epoch")
    out = pd.DataFrame({
        "budget": g.size().index,
        "stress_q50": g["stress"].median().values,
        "stress_q25": g["stress"].quantile(0.25).values,
        "stress_q75": g["stress"].quantile(0.75).values,
        "time_q50":   g["time_solver"].median().values,
        "time_q25":   g["time_solver"].quantile(0.25).values,
        "time_q75":   g["time_solver"].quantile(0.75).values,
    }).sort_values("budget").reset_index(drop=True)
    return out


def remove_old(dataset: Optional[str] = None) -> None:
    """Delete pre-redesign plot artefacts so the directory stays clean."""
    if dataset is not None:
        for stem in OLD_PER_DATASET_STEMS:
            for ext in (".png", ".pdf"):
                p = RESULTS_DIR / dataset / f"{stem}{ext}"
                if p.exists():
                    p.unlink()
    else:
        if not COMBINED_DIR.exists():
            return
        for stem in OLD_COMBINED_STEMS:
            for ext in (".png", ".pdf"):
                p = COMBINED_DIR / f"{stem}{ext}"
                if p.exists():
                    p.unlink()


# ──────────────────────────────────────────────
# Plot A — stress overhead vs budget (per-dataset, normalized)
# ──────────────────────────────────────────────

def plot_stress_overhead_vs_budget(df: pd.DataFrame, dataset: str, ax=None) -> None:
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))

    plotted_any = False
    for max_iter in sorted(df["max_iter"].unique()):
        cyc_s, _ = cycle_medians(df, max_iter)
        if not np.isfinite(cyc_s):
            continue
        agg = budget_aggregate(df, max_iter)
        if agg.empty:
            continue
        plotted_any = True
        x = agg["budget"].values
        y50 = agg["stress_q50"].values / cyc_s
        y25 = agg["stress_q25"].values / cyc_s
        y75 = agg["stress_q75"].values / cyc_s
        c = MAX_ITER_COLOR[max_iter]
        ax.plot(x, y50, MAX_ITER_LS[max_iter], color=c, linewidth=2,
                marker="o", label=f"random — {MAX_ITER_LABEL[max_iter]}")
        ax.fill_between(x, y25, y75, color=c, alpha=0.18)

    ax.axhline(1.0, linestyle=":", color=CYCLE_COLOR, linewidth=1.8,
               label="SGD-cycle (= 1.0)")

    N = int(df["n_samples"].iloc[0])
    D = int(df["n_features"].iloc[0])
    ax.set_xscale("log")
    ax.set_xlabel("n_updates_per_epoch (log)")
    ax.set_ylabel("stress / cycle stress")
    ax.set_title(f"Stress overhead — {dataset} (N={N:,}, D={D})")
    if plotted_any:
        ymin = max(0.95, ax.get_ylim()[0])
        ymax = max(1.4, ax.get_ylim()[1])
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="best")

    if own:
        out = RESULTS_DIR / dataset / "stress_overhead_vs_budget"
        fig.tight_layout()
        fig.savefig(str(out) + ".png")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# Plot B — speedup vs budget (per-dataset, normalized)
# ──────────────────────────────────────────────

def plot_speedup_vs_budget(df: pd.DataFrame, dataset: str, ax=None) -> None:
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))

    for max_iter in sorted(df["max_iter"].unique()):
        _, cyc_t = cycle_medians(df, max_iter)
        if not np.isfinite(cyc_t):
            continue
        agg = budget_aggregate(df, max_iter)
        if agg.empty:
            continue
        x = agg["budget"].values
        y50 = cyc_t / agg["time_q50"].values
        # Asymmetric band: speedup is cycle_t / random_t; widen using time IQR.
        y_lo = cyc_t / agg["time_q75"].values
        y_hi = cyc_t / agg["time_q25"].values
        c = MAX_ITER_COLOR[max_iter]
        ax.plot(x, y50, MAX_ITER_LS[max_iter], color=c, linewidth=2,
                marker="o", label=f"random — {MAX_ITER_LABEL[max_iter]}")
        ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.18)

    ax.axhline(1.0, linestyle=":", color=CYCLE_COLOR, linewidth=1.8,
               label="SGD-cycle (= 1.0)")

    N = int(df["n_samples"].iloc[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_updates_per_epoch (log)")
    ax.set_ylabel("speedup  (cycle time / random time)")
    ax.set_title(f"Wall-time speedup — {dataset} (N={N:,})")
    ax.legend(loc="best")

    if own:
        out = RESULTS_DIR / dataset / "speedup_vs_budget"
        fig.tight_layout()
        fig.savefig(str(out) + ".png")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# Plot C — normalized Pareto, all datasets on one axes
# ──────────────────────────────────────────────

def plot_pareto_normalized_all(datasets: List[str],
                               xlim=(0.15, 3.0),
                               ylim=(0.97, 2.4)) -> None:
    """One figure, all datasets together, axes normalized to cycle.

    Zoomed to the interesting region: x ∈ [0.15, 3.0] (cycle = 1.0),
    y ∈ [0.97, 2.4]. Points outside that range (full-cycle-budget
    sanity points blow x out to ~16 on some datasets) are dropped
    from the *line*'s endpoint to keep the curve in-frame.
    """
    if not datasets:
        return

    curves = []  # list of (dataset, N, max_iter, x_ratio, y_ratio)
    for ds in datasets:
        df = load(ds)
        N = int(df["n_samples"].iloc[0])
        for mi in sorted(df["max_iter"].unique()):
            cyc_s, cyc_t = cycle_medians(df, mi)
            if not (np.isfinite(cyc_s) and np.isfinite(cyc_t)):
                continue
            agg = budget_aggregate(df, mi)
            if agg.empty:
                continue
            x = agg["time_q50"].values / cyc_t
            y = agg["stress_q50"].values / cyc_s
            order = np.argsort(x)
            curves.append((ds, N, mi, x[order], y[order]))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Win-quadrant shading (faster AND lower stress than cycle).
    # Use fill_between with axis ranges directly; works on log-x.
    ax.fill_between([xlim[0], 1.0], ylim[0], 1.0,
                    color="green", alpha=0.07, zorder=0,
                    label="win quadrant (faster + lower stress)")

    for ds, N, mi, x, y in curves:
        color = DATASET_PALETTE.get(ds, "#666666")
        ls = MAX_ITER_LS[mi]
        # Label only the max_iter=30 curve to avoid duplicates.
        label = f"{ds} (N={N:,})" if mi == 30 else None
        ax.plot(x, y, ls, color=color, linewidth=2.2, marker="o",
                markersize=5.5, label=label, alpha=0.95)

    # Cycle reference lines + anchor star, drawn LAST so they stay on top.
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(1.0, linestyle=":", color="black", linewidth=1.0, alpha=0.6)
    ax.plot(1.0, 1.0, marker="*", markersize=24, color=CYCLE_COLOR,
            markeredgecolor="black", markeredgewidth=1.5,
            linestyle="none", zorder=10,
            label="SGD-cycle  (1, 1)")

    ax.set_xscale("log")
    ax.set_xlabel("wall time / cycle wall time   (1.0 = cycle speed; left = faster)")
    ax.set_ylabel("stress / cycle stress   (1.0 = cycle stress; lower = better)")
    ax.set_title("Pareto: stress vs wall time, normalized to SGD-cycle\n"
                 "(solid = max_iter=30, dashed = max_iter=15)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / "pareto_normalized_all"
    fig.tight_layout()
    fig.savefig(str(out) + ".png")
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# Plot D — speedup at quality threshold vs N
# ──────────────────────────────────────────────

def plot_scaling_vs_N(datasets: List[str],
                      k_values: List[int] = FIXED_K_VALUES,
                      max_iter: int = FIXED_MAX_ITER) -> None:
    """The scaling story — two side-by-side panels at fixed k.

    Left:  speedup (cycle_time / random_time) vs N (log x).
    Right: stress overhead (random_stress / cycle_stress) vs N (log x).

    Each line is one k value. As N grows, cycle work grows ~N² while
    random work grows ~k·N, so the speedup should grow ~linearly with
    N / (2k). The right panel checks that this speedup is NOT bought
    at the price of stress quality — overhead should stay bounded
    around 1.0–1.2 across N.
    """
    if not datasets:
        return

    rows = []
    for ds in datasets:
        df = load(ds)
        N = int(df["n_samples"].iloc[0])
        cyc_s, cyc_t = cycle_medians(df, max_iter)
        if not (np.isfinite(cyc_s) and np.isfinite(cyc_t)):
            continue
        agg = budget_aggregate(df, max_iter)
        if agg.empty:
            continue
        for k in k_values:
            budget = k * N
            # Match exactly; the runner uses budget = k * N for these.
            hit = agg[agg["budget"] == budget]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append({
                "dataset": ds, "N": N, "k": k,
                "speedup": cyc_t / float(r["time_q50"]),
                "stress_ratio": float(r["stress_q50"]) / cyc_s,
            })

    if not rows:
        print(f"  no matching (k * N) budgets found; skipping scaling plot")
        return

    pts = pd.DataFrame(rows).sort_values("N")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    k_colors = {50: "#1f77b4", 100: "#ff7f0e", 25: "#2ca02c", 200: "#9467bd"}

    for k in k_values:
        sub = pts[pts["k"] == k].sort_values("N")
        if sub.empty:
            continue
        c = k_colors.get(k, "#666666")
        axL.plot(sub["N"], sub["speedup"], "-o", color=c, linewidth=2.2,
                 markersize=9, label=f"k = {k}")
        axR.plot(sub["N"], sub["stress_ratio"], "-o", color=c, linewidth=2.2,
                 markersize=9, label=f"k = {k}")

    # Annotate dataset names on the left panel only (less clutter).
    for ds, sub in pts.groupby("dataset"):
        # Use the k=50 point (or the smallest k available) for the label.
        s = sub.sort_values("k").iloc[0]
        axL.annotate(ds, (s["N"], s["speedup"]),
                     textcoords="offset points", xytext=(6, -14),
                     fontsize=9, color="dimgrey")

    for ax in (axL, axR):
        ax.axhline(1.0, linestyle=":", color=CYCLE_COLOR, linewidth=1.8,
                   label="SGD-cycle (= 1.0)")
        ax.set_xscale("log")
        ax.set_xlabel("N (number of samples, log)")

    axL.set_ylabel("speedup  (cycle time / random time)")
    axL.set_title(f"Scaling: wall-time speedup at fixed k  (max_iter={max_iter})")
    axL.legend(loc="best")

    axR.set_ylabel("stress overhead  (random stress / cycle stress)")
    axR.set_title(f"Quality cost at fixed k  (max_iter={max_iter})")
    axR.legend(loc="best")

    fig.suptitle(
        f"As N grows, speedup grows linearly while stress overhead stays bounded   "
        f"(k = {', '.join(str(k) for k in k_values)})",
        y=1.02, fontsize=12,
    )

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / "scaling_vs_N"
    fig.tight_layout()
    fig.savefig(str(out) + ".png", bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# combined (3-column) grids for per-dataset plots A and B
# ──────────────────────────────────────────────

def _grid(datasets, plot_fn, file_stem):
    n = len(datasets)
    if n == 0:
        return
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(6.5 * cols, 4.8 * rows),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        try:
            plot_fn(load(ds), ds, ax=ax)
        except Exception as e:
            ax.set_title(f"{ds} — error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / file_stem
    fig.tight_layout()
    fig.savefig(str(out) + ".png")
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def process(dataset: str) -> None:
    print(f"\n=== plotting {dataset} ===")
    remove_old(dataset)
    df = load(dataset)
    plot_stress_overhead_vs_budget(df, dataset)
    plot_speedup_vs_budget(df, dataset)


def main() -> None:
    targets = sys.argv[1:] or get_all_datasets()
    if not targets:
        print("No datasets with results found. Run run_experiment_i.py first.")
        return

    for ds in targets:
        process(ds)

    print("\n=== combined grids (per-dataset) ===")
    remove_old(None)
    _grid(targets, plot_stress_overhead_vs_budget,
          "combined_stress_overhead_vs_budget")
    _grid(targets, plot_speedup_vs_budget,
          "combined_speedup_vs_budget")

    print("\n=== cross-dataset summaries ===")
    plot_pareto_normalized_all(targets)
    plot_scaling_vs_N(targets)


if __name__ == "__main__":
    main()
