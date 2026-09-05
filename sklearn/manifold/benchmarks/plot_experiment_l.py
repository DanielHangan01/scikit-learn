"""
Plots for Experiment L (run_experiment_l.py) -- Pivot MDS spectral baseline.

The result Experiment L establishes is a Pareto trade-off, so the figure is a
cost/quality scatter rather than a curve: for each dataset, the three methods
(full-cycle SGD, budgeted SGD, Pivot MDS) are placed by wall time and by stress
relative to the full sweep, and joined so the per-dataset trade-off is visible.

IMPORTANT -- configuration pairing. Pivot MDS is scored at its *best* quality
configuration over (selection strategy x n_pivots), and that same configuration
supplies its wall time. Quoting best-of-all quality against one fixed strategy's
time mixes operating points: `maxmin_pca10` is both better and much cheaper than
`maxmin` on the high-D sources (cifar10_raw at k=200: 1.41x/1.1s vs 1.42x/29.6s),
so the mixed pairing understates Pivot MDS's speed by up to ~13x.

Usage:
    python plot_experiment_l.py                # both suites
    python plot_experiment_l.py --suite k8
"""

from __future__ import annotations

import sys
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).parent / "results"

# Each L suite is only comparable against the SGD experiment it reused the
# subsamples from; see EXPERIMENT_L_RESULTS.md section 1.
SUITES = {
    "j18": ("experiment_l_j18", "experiment_j"),
    "k8": ("experiment_l_k8", "experiment_k_optimized"),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "font.size": 13,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})

C_CYCLE, C_BUDGET, C_PIVOT = "#4d4d4d", "#1f77b4", "#ff7f0e"


def collect(suite: str):
    """One row per dataset: cycle / budgeted-SGD / Pivot-MDS cost and quality."""
    l_dir, sgd_dir = SUITES[suite]
    rows = []
    for f in sorted(glob.glob(str(BASE / l_dir / "*" / "results.csv"))):
        ds = os.path.basename(os.path.dirname(f))
        sgd_path = BASE / sgd_dir / ds / "results.csv"
        if not sgd_path.exists():
            continue
        L = pd.read_csv(f)
        S = pd.read_csv(sgd_path)
        if "max_iter" in S.columns:
            S = S[S["max_iter"] == 30]

        # Compare at the largest N both experiments share.
        N = int(min(L["n_samples"].max(), S["n_samples"].max()))
        L, S = L[L["n_samples"] == N], S[S["n_samples"] == N]
        cyc = S[S["phase"] == "cycle"]
        bud = S[(S["phase"] == "budget") & (S["k"] == 200)]
        if cyc.empty or L.empty:
            continue
        cs, ct = cyc["stress"].median(), cyc["time_solver"].median()

        # Pivot MDS at its best-quality config, timed as that same config.
        q = L.groupby(["pivot_strategy", "n_pivots"])["stress"].median() / cs
        strat, npiv = q.idxmin()
        sub = L[(L["pivot_strategy"] == strat) & (L["n_pivots"] == npiv)]

        rows.append(dict(
            dataset=ds, D=int(L["n_features"].iloc[0]), N=N,
            cycle_t=ct, cycle_q=1.0,
            budget_t=bud["time_solver"].median() if not bud.empty else np.nan,
            budget_q=bud["stress"].median() / cs if not bud.empty else np.nan,
            pivot_t=sub["time_solver"].median(), pivot_q=q.min(),
            pivot_cfg=f"{strat}, k={npiv}",
            pivot_sel_frac=sub["time_pivot_selection"].median() / sub["time_solver"].median(),
        ))
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, suite: str, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for _, r in df.iterrows():
        xs = [r.pivot_t, r.budget_t, r.cycle_t]
        ys = [r.pivot_q, r.budget_q, r.cycle_q]
        ax.plot(xs, ys, "-", color="grey", lw=0.7, alpha=0.45, zorder=1)
    ax.scatter(df.pivot_t, df.pivot_q, s=95, c=C_PIVOT, edgecolors="black",
               zorder=3, label="Pivot MDS (best configuration)")
    ok = df.budget_q == df.budget_q
    ax.scatter(df.budget_t[ok], df.budget_q[ok], s=95, c=C_BUDGET,
               edgecolors="black", zorder=3, label="budgeted SGD ($k=200$)")
    ax.scatter(df.cycle_t, df.cycle_q, s=95, c=C_CYCLE, edgecolors="black",
               marker="s", zorder=3, label="full sweep (reference)")
    ax.axhline(1.0, ls=":", color="black", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"])
    ax.set_xlabel("wall time to produce the embedding (s, log)")
    ax.set_ylabel("stress relative to the full sweep (log)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out) + ".png")
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.{{png,pdf}}")


def main():
    argv = sys.argv[1:]
    suites = list(SUITES)
    if "--suite" in argv:
        suites = [argv[argv.index("--suite") + 1]]
    for suite in suites:
        df = collect(suite)
        if df.empty:
            print(f"no results for suite {suite}")
            continue
        out_dir = BASE / SUITES[suite][0] / "_plots_combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== suite {suite} ({len(df)} datasets) ===")
        sp = df.budget_t / df.pivot_t
        print(f"  Pivot MDS quality  {df.pivot_q.min():.2f}-{df.pivot_q.max():.2f}"
              f"  median {df.pivot_q.median():.2f}")
        print(f"  budgeted SGD       {df.budget_q.min():.2f}-{df.budget_q.max():.2f}"
              f"  median {df.budget_q.median():.2f}")
        print(f"  speedup vs SGD     {sp.min():.1f}-{sp.max():.1f}x"
              f"  median {sp.median():.1f}x")
        print(f"  Pivot MDS quality wins: {(df.pivot_q < df.budget_q).sum()}/{ok_n(df)}")
        print(f"  selection share of Pivot MDS time: "
              f"{100*df.pivot_sel_frac.min():.0f}-{100*df.pivot_sel_frac.max():.0f}%")
        df.to_csv(out_dir / "summary.csv", index=False)
        plot_pareto(df, suite, out_dir / f"pareto_{suite}")


def ok_n(df):
    return int((df.budget_q == df.budget_q).sum())


if __name__ == "__main__":
    main()
