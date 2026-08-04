"""
Plots for Experiment I-scaling (run_experiment_i_scaling.py).

Reads results/experiment_i_scaling/<source>/results.csv.

Per source:
  A. overhead_vs_k.{pdf,png}
        x = k (log), y = stress overhead (random/cycle, linear, anchored
        at 1.0). One line per N. Shows the overhead curve shifting up/right
        as N grows at fixed k.
  B. kstar_vs_N.{pdf,png}
        x = N (log), y = k* needed to reach a target overhead (log).
        One line per target. The slope of the log-log fit is the scaling
        exponent p in k* ~ N^p (annotated). p~0 => constant k; p~1 =>
        linear (budget ~ N^2).

Combined across sources:
  combined_kstar_vs_N.{pdf,png} — Plot B for every source side by side,
  so the exponent can be compared across intrinsic dimensionalities.

Usage:
    python plot_experiment_i_scaling.py            # all sources with results
    python plot_experiment_i_scaling.py hiva       # subset
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_i_scaling"
COMBINED_DIR = RESULTS_DIR / "_plots_combined"

TARGETS = [1.05, 1.10, 1.15]
TARGET_COLOR = {1.05: "#d62728", 1.10: "#1f77b4", 1.15: "#2ca02c"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "font.size": 13,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})


def get_all_sources() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "results.csv").exists()
    )


def load(source: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / source / "results.csv")
    df["n_samples"] = df["n_samples"].astype(int)
    df["stress"] = df["stress"].astype(float)
    return df


def overhead_table(df: pd.DataFrame):
    """Return {N: (ks, overheads)} medians over seeds."""
    out = {}
    for N in sorted(df["n_samples"].unique()):
        cyc = df[(df["phase"] == "cycle") & (df["n_samples"] == N)]["stress"].median()
        bud = df[(df["phase"] == "budget") & (df["n_samples"] == N)]
        ks, ov = [], []
        for k, grp in bud.groupby("k"):
            ks.append(int(k)); ov.append(grp["stress"].median() / cyc)
        order = np.argsort(ks)
        out[int(N)] = (list(np.array(ks)[order]), list(np.array(ov)[order]))
    return out


def k_star(ks, overheads, target):
    ks = np.asarray(ks, float)
    excess = np.asarray(overheads, float) - 1.0
    tgt = target - 1.0
    order = np.argsort(ks); ks, excess = ks[order], excess[order]
    if excess.min() > tgt:
        return None
    if excess.max() <= tgt:
        return float(ks[0])
    lk = np.log(ks); lexc = np.log(np.clip(excess, 1e-9, None))
    return float(np.exp(np.interp(np.log(tgt), lexc[::-1], lk[::-1])))


def plot_overhead_vs_k(df, source, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    tbl = overhead_table(df)
    Ns = sorted(tbl)
    cmap = cm.get_cmap("viridis")
    all_ks = set()
    for i, N in enumerate(Ns):
        ks, ov = tbl[N]
        all_ks.update(ks)
        c = cmap(i / max(len(Ns) - 1, 1))
        ax.plot(ks, ov, "-o", color=c, linewidth=2, label=f"N={N}")
    ax.axhline(1.0, ls=":", color="black", lw=1.2, label="cycle (=1.0)")
    ax.set_xscale("log")
    # Label the x-axis with the exact k values used, not log decades.
    xticks = sorted(all_ks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(k) for k in xticks])
    ax.minorticks_off()
    ax.set_xlabel("k  (n_updates_per_epoch = k·N, log)")
    ax.set_ylabel("stress overhead  (random / cycle)")
    D = int(df["n_features"].iloc[0])
    ax.set_title(f"Overhead vs budget by N — {source} (D={D})")
    ax.legend(loc="best", ncol=2)
    if own:
        out = RESULTS_DIR / source / "overhead_vs_k"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def plot_kstar_vs_N(df, source, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    tbl = overhead_table(df)
    Ns_all = sorted(tbl)
    D = int(df["n_features"].iloc[0])
    for t in TARGETS:
        Ns, ks = [], []
        for N in Ns_all:
            kk = k_star(*tbl[N], t)
            if kk is not None:
                Ns.append(N); ks.append(kk)
        if len(Ns) < 2:
            continue
        c = TARGET_COLOR[t]
        ax.plot(Ns, ks, "o", color=c, markersize=9)
        label = f"overhead ≤ {t:.2f}×"
        if len(Ns) >= 3:
            p, b = np.polyfit(np.log(Ns), np.log(ks), 1)
            xs = np.array([min(Ns), max(Ns)], float)
            ax.plot(xs, np.exp(b) * xs ** p, "-", color=c, lw=2,
                    label=f"{label}:  p={p:.2f}")
        else:
            ax.plot(Ns, ks, "-", color=c, lw=2, label=label)
    # Reference slopes p=0.5 and p=1.0 anchored at the first plotted point.
    if Ns_all:
        N0 = Ns_all[0]
        k0 = k_star(*tbl[N0], 1.10) or 25
        xs = np.array([min(Ns_all), max(Ns_all)], float)
        ax.plot(xs, k0 * (xs / N0) ** 1.0, "--", color="grey", lw=1,
                alpha=0.7, label="p=1 (linear ref)")
        ax.plot(xs, k0 * (xs / N0) ** 0.5, ":", color="grey", lw=1,
                alpha=0.7, label="p=0.5 (√N ref)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (subsample size, log)")
    ax.set_ylabel("k*  (budget to hold overhead, log)")
    ax.set_title(f"Budget scaling k*(N) — {source} (D={D})")
    ax.legend(loc="best")
    if own:
        out = RESULTS_DIR / source / "kstar_vs_N"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def _grid(sources, plot_fn, stem):
    n = len(sources)
    if n == 0:
        return
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.8 * rows),
                             squeeze=False)
    for i, s in enumerate(sources):
        ax = axes[i // cols][i % cols]
        try:
            plot_fn(load(s), s, ax=ax)
        except Exception as e:
            ax.set_title(f"{s} — error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / stem
    fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
    plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def main():
    sources = sys.argv[1:] or get_all_sources()
    if not sources:
        print("No results. Run run_experiment_i_scaling.py first.")
        return
    for s in sources:
        print(f"\n=== plotting {s} ===")
        df = load(s)
        plot_overhead_vs_k(df, s)
        plot_kstar_vs_N(df, s)
    print("\n=== combined ===")
    _grid(sources, plot_kstar_vs_N, "combined_kstar_vs_N")
    _grid(sources, plot_overhead_vs_k, "combined_overhead_vs_k")


if __name__ == "__main__":
    main()
