"""
Experiment J analysis + plots: budget-scaling behaviour of k across all 18
datasets, and its mechanistic link to intrinsic dimensionality.

Reads results/experiment_j/<dataset>/{results.csv, meta.json}.

Per dataset derives (at max_iter=30 unless overridden):
  - overhead(N, k) = median_seed(random_stress / cycle_stress)
  - k*(N)          interpolated budget hitting target overhead {1.05,1.10,1.15}
  - p              exponent of k* ~ N^p (log-log slope; needs >=3 N rungs)
  - s              steepness of excess ~ k^-s  (fit at a common N and at full N)
  - floor          min overhead reached at the largest tested k (full N)

Per-dataset plots (combined grids in _plots_combined/):
  - overhead_vs_k        overhead vs k, one line per N (k ticks labelled)
  - kstar_vs_N           k*(N) log-log with fitted p

Cross-dataset payoff (single figures in _plots_combined/):
  - master_table.csv             every measured scalar per dataset
  - p_vs_intrinsic_dim           scaling exponent vs strain_top2_frac / PR
  - steepness_vs_intrinsic_dim   curve steepness s vs intrinsic dim
  - p_vs_steepness               p vs s (expect inverse relation)
  - regime_map                   datasets in (intrinsic dim, floor) space
  - collapse_test                does overhead collapse on coverage f=2k/N
                                 (linear-k law) or on absolute k (const-k law)?

Usage:
    python plot_experiment_j.py            # all datasets with results
    python plot_experiment_j.py coil20     # per-dataset plots for a subset
    python plot_experiment_j.py --table    # print master table only
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_j"
COMBINED_DIR = RESULTS_DIR / "_plots_combined"

TARGETS = [1.05, 1.10, 1.15]
TARGET_COLOR = {1.05: "#d62728", 1.10: "#1f77b4", 1.15: "#2ca02c"}
REF_MAX_ITER = 30          # which max_iter to analyze
COMMON_N = 900             # controlled rung for cross-dataset steepness

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "font.size": 12,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})


# ──────────────────────────────────────────────
# loading
# ──────────────────────────────────────────────

def get_all_datasets() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "results.csv").exists() and (d / "meta.json").exists()
    )


def load(ds: str):
    """Load the scaling-reference (max_iter=REF_MAX_ITER) rows + meta."""
    df = load_raw(ds)
    df = df[df["max_iter"] == REF_MAX_ITER].copy()
    meta = json.loads((RESULTS_DIR / ds / "meta.json").read_text())
    return df, meta


def load_raw(ds: str):
    """All rows (every max_iter)."""
    df = pd.read_csv(RESULTS_DIR / ds / "results.csv")
    df["n_samples"] = df["n_samples"].astype(int)
    df["max_iter"] = df["max_iter"].astype(int)
    df["stress"] = df["stress"].astype(float)
    return df


def iter_penalty_table(ds: str):
    """{N: (ks, ratios)} where ratio = stress(15)/stress(30) for budget runs.
    Returns ({}, cycle_ratio_at_fullN or nan) if both max_iter not present."""
    df = load_raw(ds)
    mis = sorted(df["max_iter"].unique())
    if not (15 in mis and 30 in mis):
        return {}, float("nan")
    out = {}
    for N in sorted(df["n_samples"].unique()):
        bud = df[(df["phase"] == "budget") & (df["n_samples"] == N)]
        ks, ratios = [], []
        for k, grp in bud.groupby("k"):
            s15 = grp[grp["max_iter"] == 15]["stress"].median()
            s30 = grp[grp["max_iter"] == 30]["stress"].median()
            if np.isfinite(s15) and np.isfinite(s30) and s30 > 0:
                ks.append(int(k)); ratios.append(s15 / s30)
        if ks:
            order = np.argsort(ks)
            out[int(N)] = (list(np.array(ks)[order]), list(np.array(ratios)[order]))
    # cycle reference ratio at full N
    Nmax = max(df["n_samples"].unique())
    cyc = df[(df["phase"] == "cycle") & (df["n_samples"] == Nmax)]
    c15 = cyc[cyc["max_iter"] == 15]["stress"].median()
    c30 = cyc[cyc["max_iter"] == 30]["stress"].median()
    cyc_ratio = c15 / c30 if (np.isfinite(c15) and np.isfinite(c30) and c30 > 0) else float("nan")
    return out, cyc_ratio


# ──────────────────────────────────────────────
# derived quantities
# ──────────────────────────────────────────────

def overhead_table(df) -> Dict[int, Tuple[List[int], List[float]]]:
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


def k_star(ks, ov, target) -> Optional[float]:
    ks = np.asarray(ks, float); excess = np.asarray(ov, float) - 1.0
    tgt = target - 1.0
    order = np.argsort(ks); ks, excess = ks[order], excess[order]
    if excess.min() > tgt:
        return None
    if excess.max() <= tgt:
        return float(ks[0])
    lk = np.log(ks); lexc = np.log(np.clip(excess, 1e-9, None))
    return float(np.exp(np.interp(np.log(tgt), lexc[::-1], lk[::-1])))


def fit_exponent(tbl, target) -> Optional[float]:
    Ns, ks = [], []
    for N in sorted(tbl):
        kk = k_star(*tbl[N], target)
        if kk is not None:
            Ns.append(N); ks.append(kk)
    if len(Ns) < 3:
        return None
    return float(np.polyfit(np.log(Ns), np.log(ks), 1)[0])


def fit_steepness(ks, ov) -> Optional[float]:
    """excess ~ k^-s. Returns s (>0). Fit over points with excess>1e-3."""
    ks = np.asarray(ks, float); excess = np.asarray(ov, float) - 1.0
    m = excess > 1e-3
    if m.sum() < 2:
        return None
    sl = np.polyfit(np.log(ks[m]), np.log(excess[m]), 1)[0]
    return float(-sl)


def collapse_r2(rows) -> Tuple[float, float]:
    """R^2 of log(excess) vs log(k) (const-k law) and vs log(f=2k/N)
    (linear-k law), pooled across all (N,k) for one dataset."""
    K = np.array([r[0] for r in rows], float)
    N = np.array([r[1] for r in rows], float)
    E = np.array([r[2] for r in rows], float) - 1.0
    m = E > 1e-3
    if m.sum() < 4:
        return float("nan"), float("nan")
    K, N, E = K[m], N[m], E[m]
    f = 2 * K / (N - 1)

    def r2(x):
        x = np.log(x); y = np.log(E)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yh = A @ coef
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

    return float(r2(K)), float(r2(f))


def classify(p, s, floor) -> str:
    if s is not None and s > 0.8 and (floor is None or floor < 1.04):
        return "constant-k"
    if (s is not None and s < 0.3) or (floor is not None and floor > 1.05):
        return "saturating"
    return "intermediate"


def measure(ds: str) -> Dict:
    df, meta = load(ds)
    tbl = overhead_table(df)
    Ns = sorted(tbl)
    Nmax = Ns[-1]
    # steepness at full N and at the common controlled rung if present
    s_full = fit_steepness(*tbl[Nmax])
    s_common = fit_steepness(*tbl[COMMON_N]) if COMMON_N in tbl else None
    # floor: best (min) overhead reached at full N
    floor = float(min(tbl[Nmax][1])) if tbl[Nmax][1] else None
    rec = {
        "dataset": ds, "N": meta["n_samples"], "D": meta["n_features"],
        "pca_PR": meta.get("pca_participation_ratio"),
        "pca_n90": meta.get("pca_n90"),
        "strain_top2": meta.get("strain_top2_frac"),
        "strain_PR": meta.get("strain_participation_ratio"),
        "n_rungs": len(Ns),
        "s_fullN": s_full, "s_commonN": s_common, "floor_fullN": floor,
    }
    for t in TARGETS:
        rec[f"p_{t:.2f}"] = fit_exponent(tbl, t)
    # collapse test
    pooled = []
    for N in Ns:
        ks, ov = tbl[N]
        for k, o in zip(ks, ov):
            pooled.append((k, N, o))
    r2k, r2f = collapse_r2(pooled)
    rec["collapse_R2_k"] = r2k
    rec["collapse_R2_f"] = r2f
    # iteration penalty: peak (worst) mid-k stress(15)/stress(30) at full N,
    # and the cycle reference ratio.
    ip_tbl, cyc_ratio = iter_penalty_table(ds)
    rec["iter_penalty_cycle"] = cyc_ratio
    if Nmax in ip_tbl:
        rec["iter_penalty_peak"] = float(max(ip_tbl[Nmax][1]))
        # ratio at k=50 specifically (the recommended operating point)
        ks50, r50 = ip_tbl[Nmax]
        rec["iter_penalty_k50"] = float(dict(zip(ks50, r50)).get(50, float("nan")))
    else:
        rec["iter_penalty_peak"] = float("nan")
        rec["iter_penalty_k50"] = float("nan")
    rec["regime"] = classify(rec.get("p_1.10"), s_full, floor)
    return rec


# ──────────────────────────────────────────────
# per-dataset plots
# ──────────────────────────────────────────────

def plot_overhead_vs_k(ds, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    df, meta = load(ds)
    tbl = overhead_table(df)
    Ns = sorted(tbl)
    cmap = cm.get_cmap("viridis")
    all_ks = set()
    for i, N in enumerate(Ns):
        ks, ov = tbl[N]; all_ks.update(ks)
        ax.plot(ks, ov, "-o", color=cmap(i / max(len(Ns) - 1, 1)),
                lw=2, label=f"N={N}")
    ax.axhline(1.0, ls=":", color="black", lw=1.2)
    ax.set_xscale("log")
    xticks = sorted(all_ks)
    ax.set_xticks(xticks); ax.set_xticklabels([str(k) for k in xticks])
    ax.minorticks_off()
    ax.set_xlabel("k  (n_updates_per_epoch = k·N)")
    ax.set_ylabel("overhead (random / cycle)")
    ax.set_title(f"{ds} (N={meta['n_samples']}, D={meta['n_features']}, "
                 f"strain₂={meta.get('strain_top2_frac', float('nan')):.2f})")
    ax.legend(loc="best", ncol=2, fontsize=8)
    if own:
        out = RESULTS_DIR / ds / "overhead_vs_k"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def plot_kstar_vs_N(ds, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    df, meta = load(ds)
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
        ax.plot(Ns, ks, "o", color=c, ms=8)
        if len(Ns) >= 3:
            p, b = np.polyfit(np.log(Ns), np.log(ks), 1)
            xs = np.array([min(Ns), max(Ns)], float)
            ax.plot(xs, np.exp(b) * xs ** p, "-", color=c, lw=2,
                    label=f"≤{t:.2f}×: p={p:.2f}")
        else:
            ax.plot(Ns, ks, "-", color=c, lw=2, label=f"≤{t:.2f}×")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (subsample size)")
    ax.set_ylabel("k* (budget to hold overhead)")
    ax.set_title(f"{ds} — k*(N)")
    ax.legend(loc="best")
    if own:
        out = RESULTS_DIR / ds / "kstar_vs_N"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def plot_iter_penalty_vs_k(ds, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 5))
    ip_tbl, cyc_ratio = iter_penalty_table(ds)
    meta = json.loads((RESULTS_DIR / ds / "meta.json").read_text())
    if not ip_tbl:
        ax.text(0.5, 0.5, "no 15-vs-30 data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{ds}")
        if own:
            plt.close()
        return
    Ns = sorted(ip_tbl)
    cmap = cm.get_cmap("plasma")
    all_ks = set()
    for i, N in enumerate(Ns):
        ks, r = ip_tbl[N]; all_ks.update(ks)
        ax.plot(ks, r, "-o", color=cmap(i / max(len(Ns) - 1, 1)), lw=2,
                label=f"N={N}")
    if np.isfinite(cyc_ratio):
        ax.axhline(cyc_ratio, ls="--", color="green", lw=1.5,
                   label=f"cycle ({cyc_ratio:.3f})")
    ax.axhline(1.0, ls=":", color="black", lw=1.0)
    ax.set_xscale("log")
    xticks = sorted(all_ks)
    ax.set_xticks(xticks); ax.set_xticklabels([str(k) for k in xticks])
    ax.minorticks_off()
    ax.set_xlabel("k  (n_updates_per_epoch = k·N)")
    ax.set_ylabel("stress(15) / stress(30)")
    ax.set_title(f"{ds} — iteration penalty "
                 f"(strain₂={meta.get('strain_top2_frac', float('nan')):.2f})")
    ax.legend(loc="best", ncol=2, fontsize=8)
    if own:
        out = RESULTS_DIR / ds / "iter_penalty_vs_k"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def _grid(datasets, plot_fn, stem):
    n = len(datasets)
    if n == 0:
        return
    cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.6 * rows),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        try:
            plot_fn(ds, ax=ax)
        except Exception as e:
            ax.set_title(f"{ds} — error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / stem
    fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
    plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


# ──────────────────────────────────────────────
# cross-dataset scatters
# ──────────────────────────────────────────────

def _scatter(M, xcol, ycol, xlabel, ylabel, title, stem, *,
             xlog=False, ylog=False, annotate=True):
    sub = M.dropna(subset=[xcol, ycol])
    if sub.empty:
        print(f"  (skip {stem}: no data)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub[xcol], sub[ycol], s=60, c="#1f77b4", zorder=3)
    if annotate:
        for _, r in sub.iterrows():
            ax.annotate(r["dataset"], (r[xcol], r[ycol]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    # Spearman correlation (rank, robust to nonlinearity)
    try:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(sub[xcol], sub[ycol])
        title = f"{title}   (Spearman ρ={rho:.2f}, p={pv:.3f}, n={len(sub)})"
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / stem
    fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
    plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def cross_dataset(datasets):
    M = pd.DataFrame([measure(ds) for ds in datasets])
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    M.to_csv(COMBINED_DIR / "master_table.csv", index=False)

    cols = ["dataset", "N", "D", "strain_top2", "pca_PR", "n_rungs",
            "p_1.05", "p_1.10", "p_1.15", "s_fullN", "s_commonN",
            "floor_fullN", "collapse_R2_k", "collapse_R2_f",
            "iter_penalty_k50", "iter_penalty_peak", "iter_penalty_cycle",
            "regime"]
    print("\n=== master table ===")
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(M[cols].to_string(index=False))

    # Headline mechanistic scatters.
    _scatter(M, "strain_top2", "p_1.10",
             "strain_top2_frac  (2D-embeddability; high = easy)",
             "scaling exponent p  (k* ~ N^p, target 1.10×)",
             "Does the scaling exponent track 2D-embeddability?",
             "p_vs_intrinsic_dim")
    _scatter(M, "strain_top2", "s_fullN",
             "strain_top2_frac (high = easy)",
             "overhead-vs-k steepness s  (excess ~ k^-s)",
             "Curve steepness vs 2D-embeddability",
             "steepness_vs_intrinsic_dim")
    _scatter(M, "pca_PR", "p_1.10",
             "PCA participation ratio (effective feature dim)",
             "scaling exponent p (target 1.10×)",
             "Scaling exponent vs PCA effective dim",
             "p_vs_pca_dim", xlog=True)
    _scatter(M, "s_fullN", "p_1.10",
             "steepness s", "scaling exponent p (target 1.10×)",
             "p vs steepness (expect inverse)",
             "p_vs_steepness")
    _scatter(M, "strain_top2", "floor_fullN",
             "strain_top2_frac (high = easy)",
             "overhead floor at full N (min over k)",
             "Saturation floor vs 2D-embeddability",
             "regime_map")
    # Iteration-penalty interaction (H-J5): is the mid-k 15-vs-30 hump
    # largest exactly where the budget knob has leverage (steep curve /
    # high 2D-embeddability)?
    _scatter(M, "strain_top2", "iter_penalty_peak",
             "strain_top2_frac (high = easy / steep curve)",
             "peak mid-k penalty  stress(15)/stress(30) at full N",
             "Iteration penalty vs 2D-embeddability",
             "iter_penalty_vs_intrinsic_dim")
    _scatter(M, "s_fullN", "iter_penalty_peak",
             "overhead-vs-k steepness s",
             "peak mid-k penalty  stress(15)/stress(30)",
             "Iteration penalty vs curve steepness",
             "iter_penalty_vs_steepness")

    # Collapse test bar: per dataset, which law fits better.
    sub = M.dropna(subset=["collapse_R2_k", "collapse_R2_f"]).sort_values("strain_top2")
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(sub)); w = 0.4
        ax.bar(x - w / 2, sub["collapse_R2_k"], w, label="R² vs k (const-k law)",
               color="#1f77b4")
        ax.bar(x + w / 2, sub["collapse_R2_f"], w, label="R² vs f=2k/N (linear-k law)",
               color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(sub["dataset"], rotation=60, ha="right")
        ax.set_ylabel("R² of log(excess) collapse")
        ax.set_title("Which invariant controls overhead? (datasets sorted by strain_top2)")
        ax.legend()
        out = COMBINED_DIR / "collapse_test"
        fig.tight_layout(); fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
        plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")

    return M


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    table_only = "--table" in args
    args = [a for a in args if not a.startswith("--")]
    datasets = args or get_all_datasets()
    if not datasets:
        print("No results. Run run_experiment_j.py first.")
        return

    if not table_only:
        for ds in datasets:
            print(f"\n=== plotting {ds} ===")
            plot_overhead_vs_k(ds)
            plot_kstar_vs_N(ds)
            plot_iter_penalty_vs_k(ds)
        print("\n=== combined per-dataset grids ===")
        _grid(datasets, plot_overhead_vs_k, "combined_overhead_vs_k")
        _grid(datasets, plot_kstar_vs_N, "combined_kstar_vs_N")
        _grid(datasets, plot_iter_penalty_vs_k, "combined_iter_penalty_vs_k")

    print("\n=== cross-dataset analysis ===")
    cross_dataset(datasets)


if __name__ == "__main__":
    main()
