"""
Plots for Experiment K (run_experiment_k.py) -- large-N budget scaling.

Reads results/experiment_k/<dataset>/results.csv (+ meta.json for the
intrinsic-dim annotation). All analysis uses max_iter=REF_MAX_ITER.

Per dataset:
  A. overhead_vs_k.{pdf,png}
        x = k (log, explicit tick per tested budget), y = stress overhead
        (random/cycle, linear, anchored at 1.0). One line per N. Shows the
        overhead curve as N grows at fixed k.
  B. kstar_vs_N.{pdf,png}
        x = N (log), y = k* to reach a target overhead (log). Slope of the
        log-log fit is the scaling exponent p in k* ~ N^p (annotated).
  C. overhead_vs_N_at_fixed_k.{pdf,png}  -- the large-N "money" plot
        x = N (log), y = overhead, one line per fixed k. A flat line means a
        fixed budget k holds quality as N grows (the constant-k win); an
        upward line means the budget must grow with N.

Combined across datasets (_plots_combined/):
  combined_kstar_vs_N, combined_overhead_vs_k, combined_overhead_vs_N_fixed_k.

Usage:
    python plot_experiment_k.py            # all datasets with results
    python plot_experiment_k.py fashion_mnist_full
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_k"
COMBINED_DIR = RESULTS_DIR / "_plots_combined"

REF_MAX_ITER = 30
TARGETS = [1.05, 1.10, 1.15]
TARGET_COLOR = {1.05: "#d62728", 1.10: "#1f77b4", 1.15: "#2ca02c"}
# Fixed-k palette for plot C (shared, distinct per budget).
K_COLOR = {50: "#d62728", 100: "#ff7f0e", 200: "#1f77b4", 400: "#2ca02c"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "font.size": 13,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})


def get_all_datasets() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "results.csv").exists()
    )


def load(dataset: str):
    df = pd.read_csv(RESULTS_DIR / dataset / "results.csv")
    df["n_samples"] = df["n_samples"].astype(int)
    df["stress"] = df["stress"].astype(float)
    if "max_iter" in df.columns:
        df = df[df["max_iter"] == REF_MAX_ITER]
    meta = {}
    mpath = RESULTS_DIR / dataset / "meta.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text())
    return df, meta


def _title(dataset, meta, df):
    D = int(df["n_features"].iloc[0])
    s = meta.get("strain_top2_frac")
    tag = f"D={D}"
    if s is not None:
        tag += f", strain_top2={s:.2f}"
    return f"{dataset} ({tag})"


def overhead_table(df: pd.DataFrame):
    """Return {N: (ks, overheads)} medians over seeds at REF_MAX_ITER."""
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


def plot_overhead_vs_k(df, dataset, meta=None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    if meta is None:
        _, meta = load(dataset)
    tbl = overhead_table(df)
    Ns = sorted(tbl)
    cmap = cm.get_cmap("viridis")
    all_ks = set()
    for i, N in enumerate(Ns):
        ks, ov = tbl[N]
        all_ks.update(ks)
        c = cmap(i / max(len(Ns) - 1, 1))
        ax.plot(ks, ov, "-o", color=c, linewidth=2, label=f"N={N:,}")
    ax.axhline(1.0, ls=":", color="black", lw=1.2, label="cycle (=1.0)")
    ax.set_xscale("log")
    xticks = sorted(all_ks)
    ax.set_xticks(xticks); ax.set_xticklabels([str(k) for k in xticks])
    ax.minorticks_off()
    ax.set_xlabel("k  (n_updates_per_epoch = k·N, log)")
    ax.set_ylabel("stress overhead  (random / cycle)")
    ax.set_title(f"Overhead vs budget — {_title(dataset, meta, df)}")
    ax.legend(loc="best", ncol=2)
    if own:
        _save(fig, RESULTS_DIR / dataset / "overhead_vs_k")


def plot_overhead_vs_N_fixed_k(df, dataset, meta=None, ax=None):
    """The large-N money plot: fixed k, does overhead stay flat as N grows?"""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    if meta is None:
        _, meta = load(dataset)
    tbl = overhead_table(df)
    Ns = sorted(tbl)
    # invert to {k: [(N, overhead)]}
    series = {}
    for N in Ns:
        for k, ov in zip(*tbl[N]):
            series.setdefault(k, []).append((N, ov))
    for k in sorted(series):
        pts = sorted(series[k])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        c = K_COLOR.get(k, None)
        ax.plot(xs, ys, "-o", color=c, lw=2, label=f"k={k}")
    ax.axhline(1.0, ls=":", color="black", lw=1.2, label="cycle (=1.0)")
    ax.set_xscale("log")
    xticks = Ns
    ax.set_xticks(xticks); ax.set_xticklabels([f"{n:,}" for n in xticks],
                                              rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("N (subsample size, log)")
    ax.set_ylabel("stress overhead  (random / cycle)")
    ax.set_title(f"Fixed-k overhead vs N — {_title(dataset, meta, df)}")
    ax.legend(loc="best")
    if own:
        _save(fig, RESULTS_DIR / dataset / "overhead_vs_N_at_fixed_k")


def plot_kstar_vs_N(df, dataset, meta=None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    if meta is None:
        _, meta = load(dataset)
    tbl = overhead_table(df)
    Ns_all = sorted(tbl)
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
    if Ns_all:
        N0 = Ns_all[0]
        k0 = k_star(*tbl[N0], 1.10) or 100
        xs = np.array([min(Ns_all), max(Ns_all)], float)
        ax.plot(xs, k0 * (xs / N0) ** 1.0, "--", color="grey", lw=1,
                alpha=0.7, label="p=1 (linear ref)")
        ax.plot(xs, k0 * (xs / N0) ** 0.5, ":", color="grey", lw=1,
                alpha=0.7, label="p=0.5 (√N ref)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (subsample size, log)")
    ax.set_ylabel("k*  (budget to hold overhead, log)")
    ax.set_title(f"Budget scaling k*(N) — {_title(dataset, meta, df)}")
    ax.legend(loc="best")
    if own:
        _save(fig, RESULTS_DIR / dataset / "kstar_vs_N")


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
    plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def _grid(datasets, plot_fn, stem):
    n = len(datasets)
    if n == 0:
        return
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.8 * rows),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        try:
            df, meta = load(ds)
            plot_fn(df, ds, meta=meta, ax=ax)
        except Exception as e:
            ax.set_title(f"{ds} — error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / stem)


def _speedup_at(df, N, k_ref=50):
    """(measured, deployable) speedup at rung N for budget k_ref.

    measured   = cycle_time / budget_time (includes the O(N^2) exact-stress
                 scoring pass that a deployment would skip).
    deployable = cycle_time / (b*k_ref), where budget_time = a + b*k is fit
                 across k; the k-independent term a (the O(N^2) scoring cost)
                 is removed so only the SGD work is charged to the fast mode.
    """
    cyc_t = df[(df.phase == "cycle") & (df.n_samples == N)]["time_solver"].median()
    ks = sorted(df[(df.phase == "budget") & (df.n_samples == N)]["k"].dropna().unique())
    ts = [df[(df.phase == "budget") & (df.n_samples == N) & (df.k == k)]["time_solver"].median()
          for k in ks]
    meas = float("nan")
    for k, t in zip(ks, ts):
        if int(k) == k_ref and t > 0:
            meas = cyc_t / t
    depl = float("nan")
    if len(ks) >= 2:
        b, a = np.polyfit(np.array(ks, float), np.array(ts, float), 1)
        if b > 0:
            depl = cyc_t / (b * k_ref)
    return meas, depl


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_speedup_vs_D(datasets, k_ref=50):
    """Cross-dataset: speedup vs feature dimension D (the H-K3 money plot).

    Quality is governed by strain_top2; SPEED is governed by D (the lazy
    O(D) per-update cost). Points coloured by regime; measured (hollow) and
    deployable (filled) shown together to expose the O(N^2)-scoring gap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    Ds, meas_s, depl_s, cols, names = [], [], [], [], []
    for ds in datasets:
        df, meta = load(ds)
        N = int(df["n_samples"].max())
        D = int(meta.get("n_features", df["n_features"].iloc[0]))
        st = meta.get("strain_top2_frac", float("nan"))
        m, d = _speedup_at(df, N, k_ref)
        Ds.append(D); meas_s.append(m); depl_s.append(d); names.append(ds)
        cols.append("#1f77b4" if (st == st and st >= 0.22) else "#ff7f0e")
    Ds = np.array(Ds, float)
    ax.scatter(Ds, depl_s, s=110, c=cols, edgecolors="black", zorder=3,
               label="deployable (scoring removed)")
    ax.scatter(Ds, meas_s, s=70, facecolors="none", edgecolors=cols, zorder=3,
               label="measured (incl. O(N²) scoring)")
    for D, d, nm in zip(Ds, depl_s, names):
        if d == d:
            ax.annotate(nm.replace("_", " "), (D, d), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, ls=":", color="black", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("feature dimension D (log)")
    ax.set_ylabel(f"speedup vs cycle at k={k_ref}, N={N:,} (log)")
    rho = _spearman(Ds, depl_s)
    # blue = responsive, orange = hard-middle
    ax.scatter([], [], c="#1f77b4", label="responsive")
    ax.scatter([], [], c="#ff7f0e", label="hard-middle")
    ax.set_title(f"Speed tracks D, not quality — deployable speedup vs D  "
                 f"(Spearman ρ={rho:+.2f})")
    ax.legend(loc="best", fontsize=9)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / "speedup_vs_D")


def plot_kstar_vs_strain(datasets, target=1.05):
    """Cross-dataset: budget needed for a quality target vs 2D-embeddability.

    The cleanest K result: strain_top2 sets the budget you need (independent of
    D), while D only sets what each unit of budget costs. Datasets that never
    reach the target are drawn at the top edge as open markers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys, cols, names, misses = [], [], [], [], []
    for ds in datasets:
        df, meta = load(ds)
        N = int(df["n_samples"].max())
        tbl = overhead_table(df)
        st = meta.get("strain_top2_frac", float("nan"))
        kk = k_star(*tbl[N], target)
        c = "#1f77b4" if (st == st and st >= 0.22) else "#ff7f0e"
        if kk is None:
            misses.append((st, ds, c))
        else:
            xs.append(st); ys.append(kk); cols.append(c); names.append(ds)
    ax.scatter(xs, ys, s=120, c=cols, edgecolors="black", zorder=3)
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm.replace("_", " "), (x, y), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    if ys:
        top = max(ys) * 1.6
        for st, ds, c in misses:
            ax.scatter([st], [top], s=120, facecolors="none", edgecolors=c,
                       zorder=3)
            ax.annotate(f"{ds.replace('_',' ')}\n(never reaches)", (st, top),
                        fontsize=8, xytext=(5, -2), textcoords="offset points")
    rho = _spearman(xs, ys) if len(xs) >= 3 else float("nan")
    ax.set_yscale("log")
    ax.set_xlabel("strain_top2  (2D-embeddability; higher = more responsive)")
    ax.set_ylabel(f"k* — budget needed for ≤ {target:.2f}× overhead (log)")
    ax.scatter([], [], c="#1f77b4", label="responsive")
    ax.scatter([], [], c="#ff7f0e", label="hard-middle")
    ax.scatter([], [], facecolors="none", edgecolors="grey",
               label="never reaches target")
    ax.set_title(f"Strain sets the budget you need — k* vs strain  "
                 f"(Spearman ρ={rho:+.2f}, N=10,000)")
    ax.legend(loc="best", fontsize=9)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / "kstar_vs_strain")


def main():
    datasets = sys.argv[1:] or get_all_datasets()
    if not datasets:
        print("No results. Run run_experiment_k.py first.")
        return
    for ds in datasets:
        print(f"\n=== plotting {ds} ===")
        df, meta = load(ds)
        plot_overhead_vs_k(df, ds, meta=meta)
        plot_overhead_vs_N_fixed_k(df, ds, meta=meta)
        plot_kstar_vs_N(df, ds, meta=meta)
    print("\n=== combined ===")
    _grid(datasets, plot_kstar_vs_N, "combined_kstar_vs_N")
    _grid(datasets, plot_overhead_vs_k, "combined_overhead_vs_k")
    _grid(datasets, plot_overhead_vs_N_fixed_k, "combined_overhead_vs_N_fixed_k")
    if len(datasets) >= 3:
        print("\n=== cross-dataset: the two axes ===")
        plot_speedup_vs_D(datasets)      # D sets what each unit of budget costs
        plot_kstar_vs_strain(datasets)   # strain sets how much budget you need


if __name__ == "__main__":
    main()
