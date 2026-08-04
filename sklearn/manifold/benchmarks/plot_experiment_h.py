"""
Experiment H plotting: visualise the diagnostic results that distinguish
H1 (structural / coverage) from H2 (mechanical / learning-rate).

Reads:
  results/experiment_h/<dataset>/results.csv

Per dataset:
  1. stress_baseline_vs_k.{pdf,png}
        SGD-cycle band + PivotMDS-maxmin + PivotMDS-pca10 across k.
  2. stress_vs_lr.{pdf,png}
        SGD-pivot (raw MaxMin and PCA-10 MaxMin) stress vs learning_rate
        at K_FOR_LR_AND_HYBRID, with the cycle band. Diagnostic for H2: a
        U-shape or sharp drop ==> LR is the bottleneck.
  3. stress_vs_alpha.{pdf,png}
        SGD-hybrid (raw MaxMin and PCA-10 MaxMin) stress vs alpha in [0, 1]
        at K_FOR_LR_AND_HYBRID. Diagnostic for H1: a drop as alpha goes
        1->0.5 ==> coverage matters.

Combined (3-column grids), saved to results/experiment_h/_plots_combined/:
  combined_stress_baseline_ratio.{pdf,png}
  combined_stress_vs_lr.{pdf,png}
  combined_stress_vs_alpha.{pdf,png}

Decisive per-point histogram (Phase 2 decisive plot):
  per_point_stress_histogram.{pdf,png}
  Generated separately by `python plot_experiment_h.py --hist <dataset>`
  because it requires running each variant once and computing per-point
  stress against the full dissimilarity matrix.

Usage:
    python plot_experiment_h.py                       # all datasets with results
    python plot_experiment_h.py fashion_mnist hiva    # subset
    python plot_experiment_h.py --hist coil20         # per-point histogram only
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_h"
COMBINED_DIR = RESULTS_DIR / "_plots_combined"


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

BASELINE_COLOR = "#2ca02c"
PIVOT_MDS_MAXMIN_COLOR = "#ff7f0e"
PIVOT_MDS_PCA10_COLOR  = "#1f77b4"

# Pivot strategy palette — reused across baseline / lr / hybrid plots so
# the same strategy keeps the same colour in every figure.
STRATEGY_COLOR = {
    "maxmin": "#ff7f0e",   # raw MaxMin
    "pca10":  "#1f77b4",   # PCA-10 MaxMin
}
STRATEGY_LABEL = {
    "maxmin": "MaxMin (raw)",
    "pca10":  "MaxMin in PCA-10",
}

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
    df["k"] = df["k"].astype(int)
    df["stress"] = df["stress"].astype(float)
    df["time_solver"] = df["time_solver"].astype(float)
    return df


def _band(ax, x, y_q25, y_q50, y_q75, *, color, label, dashed=False):
    style = "--" if dashed else "-"
    ax.plot(x, y_q50, style, color=color, label=label, linewidth=2)
    ax.fill_between(x, y_q25, y_q75, color=color, alpha=0.18)


def _cycle_band(ax, df):
    cyc = df[df["algo"] == "SGD-cycle"]
    if cyc.empty:
        return None
    median = cyc["stress"].median()
    q25 = cyc["stress"].quantile(0.25)
    q75 = cyc["stress"].quantile(0.75)
    # axhspan spans the full current xlim and follows xscale automatically,
    # so it stays correct when the caller switches to log scale afterwards.
    ax.axhspan(q25, q75, color=BASELINE_COLOR, alpha=0.12)
    ax.axhline(median, linestyle="--", color=BASELINE_COLOR,
               linewidth=1.8, label=f"SGD-cycle (median={median:.2g})")
    return median


# ──────────────────────────────────────────────
# per-dataset plots
# ──────────────────────────────────────────────

def plot_baseline_vs_k(df: pd.DataFrame, dataset: str, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))

    base = df[df["phase"] == "base"]
    for strat in ("maxmin", "pca10"):
        sub = base[base["algo"] == f"PivotMDS-{strat}"]
        if sub.empty:
            continue
        agg = sub.groupby("k")["stress"].agg(
            q25=lambda v: v.quantile(0.25),
            q50="median",
            q75=lambda v: v.quantile(0.75),
        ).reset_index().sort_values("k")
        _band(ax, agg["k"], agg["q25"], agg["q50"], agg["q75"],
              color=STRATEGY_COLOR[strat],
              label=f"PivotMDS — {STRATEGY_LABEL[strat]}")

    _cycle_band(ax, base)

    N = int(df["n_samples"].iloc[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (number of pivots)")
    ax.set_ylabel("raw stress")
    ax.set_title(f"PivotMDS baselines vs SGD-cycle — {dataset} (N={N:,})")
    ax.legend(loc="best")

    if own:
        out = RESULTS_DIR / dataset / "stress_baseline_vs_k"
        fig.tight_layout()
        fig.savefig(str(out) + ".png")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.{{png,pdf}}")


def plot_stress_vs_lr(df: pd.DataFrame, dataset: str, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))

    lr_df = df[df["phase"] == "lr"]
    if lr_df.empty:
        if own:
            plt.close(fig)
        return

    for strat in ("maxmin", "pca10"):
        sub = lr_df[lr_df["pivot_strategy"] == strat]
        if sub.empty:
            continue
        agg = sub.groupby("lr")["stress"].agg(
            q25=lambda v: v.quantile(0.25),
            q50="median",
            q75=lambda v: v.quantile(0.75),
        ).reset_index().sort_values("lr")
        _band(ax, agg["lr"], agg["q25"], agg["q50"], agg["q75"],
              color=STRATEGY_COLOR[strat],
              label=f"SGD-pivot — {STRATEGY_LABEL[strat]}")

    _cycle_band(ax, df[df["phase"] == "base"])

    k_lr = int(lr_df["k"].iloc[0])
    N = int(df["n_samples"].iloc[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning_rate_init")
    ax.set_ylabel("raw stress")
    ax.set_title(f"LR sweep at k={k_lr} — {dataset} (N={N:,})")
    ax.legend(loc="best")

    if own:
        out = RESULTS_DIR / dataset / "stress_vs_lr"
        fig.tight_layout()
        fig.savefig(str(out) + ".png")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.{{png,pdf}}")


def plot_stress_vs_alpha(df: pd.DataFrame, dataset: str, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))

    hy = df[df["phase"] == "hyb"]
    if hy.empty:
        if own:
            plt.close(fig)
        return

    for strat in ("maxmin", "pca10"):
        sub = hy[hy["pivot_strategy"] == strat]
        if sub.empty:
            continue
        agg = sub.groupby("alpha")["stress"].agg(
            q25=lambda v: v.quantile(0.25),
            q50="median",
            q75=lambda v: v.quantile(0.75),
        ).reset_index().sort_values("alpha")
        _band(ax, agg["alpha"], agg["q25"], agg["q50"], agg["q75"],
              color=STRATEGY_COLOR[strat],
              label=f"SGD-hybrid — {STRATEGY_LABEL[strat]}")

    _cycle_band(ax, df[df["phase"] == "base"])

    k_hy = int(hy["k"].iloc[0])
    N = int(df["n_samples"].iloc[0])
    ax.set_yscale("log")
    ax.set_xlabel(r"hybrid_alpha (fraction of pivot updates)")
    ax.set_ylabel("raw stress")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"Hybrid sweep at k={k_hy} — {dataset} (N={N:,})")
    ax.legend(loc="best")

    if own:
        out = RESULTS_DIR / dataset / "stress_vs_alpha"
        fig.tight_layout()
        fig.savefig(str(out) + ".png")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# combined (3-column) grids
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
# decisive per-point histogram (separate entrypoint)
# ──────────────────────────────────────────────

def plot_per_point_histogram(dataset: str, k: int = 50, seed: int = 42,
                             strategy: str = "pca10"):
    """Generate the H1 decision-grade plot for one dataset.

    Runs SGD-cycle, SGD-pivot, and SGD-hybrid (alpha=0.5) at k pivots and
    seed, then plots three side-by-side histograms of per-point stress
    (sum_j (d_ij(emb) - delta_ij)^2), coloured by pivot vs non-pivot.

    strategy: 'pca10' (PCA-10 MaxMin) or 'maxmin' (raw MaxMin).
    """
    from bench_utils import (
        load_local_dataset,
        run_sgd_benchmark,
        run_pivot_benchmark,
        run_hybrid_benchmark,
        per_point_stress,
    )
    from sklearn.metrics import euclidean_distances

    if strategy == "pca10":
        ps_kw = dict(pivot_strategy="maxmin_pca", pivot_pca_dim=10)
    elif strategy == "maxmin":
        ps_kw = dict(pivot_strategy="maxmin", pivot_pca_dim=30)
    else:
        raise ValueError(f"strategy must be 'pca10' or 'maxmin' (got {strategy!r})")

    print(f"\n=== Per-point stress histogram: {dataset} k={k} "
          f"strategy={strategy} seed={seed} ===")
    X, _ = load_local_dataset(dataset)
    D_full = euclidean_distances(X, X)

    cycle = run_sgd_benchmark(X, dissimilarity="euclidean", random_state=seed,
                              max_iter=30, switch_ratio=0.5, lr=0.5, epsilon=1e-3)
    pivot = run_pivot_benchmark(X, dissimilarity="lazy", n_pivots=k,
                                random_state=seed, max_iter=30,
                                switch_ratio=0.5, lr=0.5, epsilon=1e-3,
                                **ps_kw)
    hybrid = run_hybrid_benchmark(X, n_pivots=k, hybrid_alpha=0.5,
                                  random_state=seed, max_iter=30,
                                  switch_ratio=0.5, lr=0.5, epsilon=1e-3,
                                  **ps_kw)

    # Re-derive the same pivot set the benchmark wrappers used (they don't
    # round-trip pivot_indices, but pivot selection is deterministic given
    # X, k, seed, strategy).
    from sklearn.manifold._sgd_mds import (
        _resolve_n_pivots, _select_pivots_maxmin, _maybe_pca_project,
    )
    from sklearn.utils import check_random_state
    rng = check_random_state(seed)
    k_resolved = _resolve_n_pivots(k, 2, X.shape[0])
    if strategy == "pca10":
        feats = _maybe_pca_project(X.astype(np.float64), 10, rng)
    else:
        feats = X.astype(np.float64)
    pivot_indices = set(int(i) for i in _select_pivots_maxmin(
        feats, k_resolved, rng, is_lazy=True))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    labels = [
        (cycle,  "SGD-cycle"),
        (pivot,  f"SGD-pivot-{strategy}"),
        (hybrid, f"SGD-hybrid-{strategy} (alpha=0.5)"),
    ]
    n_samples = X.shape[0]
    is_pivot_mask = np.array(
        [i in pivot_indices for i in range(n_samples)], dtype=bool
    )

    # log-spaced bins, shared across the three panels.
    pps_all = np.concatenate([
        per_point_stress(r["embedding"], D_full) for r, _ in labels
    ])
    eps = max(pps_all[pps_all > 0].min(), 1e-6) if (pps_all > 0).any() else 1e-6
    bins = np.logspace(np.log10(eps), np.log10(pps_all.max()), 50)

    for ax, (res, name) in zip(axes, labels):
        pps = per_point_stress(res["embedding"], D_full)
        ax.hist(pps[~is_pivot_mask], bins=bins, alpha=0.6,
                color="#1f77b4", label=f"non-pivot ({(~is_pivot_mask).sum()})")
        ax.hist(pps[is_pivot_mask], bins=bins, alpha=0.7,
                color="#ff7f0e", label=f"pivot ({is_pivot_mask.sum()})")
        ax.set_xscale("log")
        ax.set_xlabel(r"per-point stress  $\sum_j (d_{ij} - \delta_{ij})^2$")
        ax.set_title(f"{name}\ntotal stress = {res['stress']:.2g}")
        ax.legend(loc="best", fontsize=9)
    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Per-point stress distribution — {dataset}, k={k}, "
        f"strategy={strategy}, seed={seed}",
        fontsize=13,
    )
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"per_point_stress_histogram_{strategy}_k{k}_seed{seed}"
    fig.tight_layout()
    fig.savefig(str(out) + ".png")
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# driver
# ──────────────────────────────────────────────

def run(datasets: List[str]):
    for ds in datasets:
        print(f"\n--- {ds} ---")
        df = load(ds)
        plot_baseline_vs_k(df, ds)
        plot_stress_vs_lr(df, ds)
        plot_stress_vs_alpha(df, ds)

    print(f"\n--- combined grids ---")
    _grid(datasets, plot_baseline_vs_k, "combined_stress_baseline_vs_k")
    _grid(datasets, plot_stress_vs_lr,  "combined_stress_vs_lr")
    _grid(datasets, plot_stress_vs_alpha, "combined_stress_vs_alpha")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--hist":
        if len(args) < 2:
            sys.exit(
                "usage: plot_experiment_h.py --hist <dataset> "
                "[--strategy maxmin|pca10] [k] [seed]"
            )
        ds = args[1]
        rest = args[2:]
        strategy = "pca10"
        if rest and rest[0] == "--strategy":
            strategy = rest[1]
            rest = rest[2:]
        k = int(rest[0]) if len(rest) >= 1 else 50
        seed = int(rest[1]) if len(rest) >= 2 else 42
        plot_per_point_histogram(ds, k=k, seed=seed, strategy=strategy)
    else:
        targets = args or get_all_datasets()
        if not targets:
            sys.exit("No experiment_h results found. Run run_experiment_h.py first.")
        run(targets)