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
    python plot_experiment_k.py --exp experiment_k_optimized

The --exp flag selects which results/<dirname> tree to read. Pointing it at
`experiment_k_optimized` also switches the speed plots into OPTIMIZED mode:
that protocol fits with compute_stress=False and scores outside the timed
region, so time_solver is pure fit time and measured speedup IS deployable
speedup -- the `a + b*k` decomposition used to strip plain K's in-fit O(N^2)
scoring cost is retired and must not be applied on top.
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

EXPERIMENT = "experiment_k"
RESULTS_DIR = Path(__file__).parent / "results" / EXPERIMENT
COMBINED_DIR = RESULTS_DIR / "_plots_combined"

# Plain K timed its fits with in-fit exact stress evaluation inside the timed
# region; the optimized rerun does not. See module docstring.
OPTIMIZED = False


def _budget_series(df, k_ref):
    """(Ns, overheads, speedups) at fixed k_ref across a dataset's ladder."""
    tbl = overhead_table(df)
    Ns, ov, sp = [], [], []
    for N in sorted(tbl):
        ks, overheads = tbl[N]
        hit = [o for k, o in zip(ks, overheads) if int(k) == k_ref]
        m, _ = _speedup_at(df, N, k_ref)
        if hit and m == m:
            Ns.append(N); ov.append(hit[0]); sp.append(m)
    return Ns, ov, sp


def _select_experiment(name: str) -> None:
    """Rebind the module-level result paths to results/<name>."""
    global EXPERIMENT, RESULTS_DIR, COMBINED_DIR, OPTIMIZED
    EXPERIMENT = name
    RESULTS_DIR = Path(__file__).parent / "results" / name
    COMBINED_DIR = RESULTS_DIR / "_plots_combined"
    OPTIMIZED = "optimized" in name

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

    measured   = cycle_time / budget_time.
    deployable = cycle_time / (b*k_ref), where budget_time = a + b*k is fit
                 across k; the k-independent term a (plain K's in-fit O(N^2)
                 scoring cost) is removed so only the SGD work is charged to
                 the fast mode.

    Under the OPTIMIZED protocol time_solver already excludes stress scoring,
    so there is no k-independent contaminant to strip: measured IS deployable
    and the fit is not run (returning it again would double-count).
    """
    cyc_t = df[(df.phase == "cycle") & (df.n_samples == N)]["time_solver"].median()
    ks = sorted(df[(df.phase == "budget") & (df.n_samples == N)]["k"].dropna().unique())
    ts = [df[(df.phase == "budget") & (df.n_samples == N) & (df.k == k)]["time_solver"].median()
          for k in ks]
    meas = float("nan")
    for k, t in zip(ks, ts):
        if int(k) == k_ref and t > 0:
            meas = cyc_t / t
    if OPTIMIZED:
        return meas, meas
    depl = float("nan")
    if len(ks) >= 2:
        b, a = np.polyfit(np.array(ks, float), np.array(ts, float), 1)
        if b > 0:
            depl = cyc_t / (b * k_ref)
    return meas, depl


def _annotate_spread(ax, xs, ys, labels, dy=13):
    """Annotate points, nudging labels apart when they would collide.

    Several sources share a feature dimension (512, 784) or a strain value,
    so the default fixed offset stacks their labels on top of each other.
    Walk the points in display coordinates and push each label up by a
    multiple of dy whenever it lands on one already placed.
    """
    ax.figure.canvas.draw()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    order = np.argsort(pts[:, 0])
    placed = []  # (display_x, display_y) of labels already emitted
    for i in order:
        px, py = pts[i]
        step = 0
        while any(abs(px - qx) < 90 and abs(py + step * dy - qy) < dy
                  for qx, qy in placed):
            step += 1
        placed.append((px, py + step * dy))
        ax.annotate(str(labels[i]).replace("_", " "), (xs[i], ys[i]),
                    fontsize=8, xytext=(5, 3 + step * dy),
                    textcoords="offset points")


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
    Ds, meas_s, depl_s, cols, names, Ns = [], [], [], [], [], []
    for ds in datasets:
        df, meta = load(ds)
        N = int(df["n_samples"].max())
        D = int(meta.get("n_features", df["n_features"].iloc[0]))
        st = meta.get("strain_top2_frac", float("nan"))
        m, d = _speedup_at(df, N, k_ref)
        Ds.append(D); meas_s.append(m); depl_s.append(d); names.append(ds)
        Ns.append(N)
        cols.append("#1f77b4" if (st == st and st >= 0.22) else "#ff7f0e")
    Ds = np.array(Ds, float)
    # Speedup grows with N, so points sitting at a lower top rung are not
    # comparable down the column; flag them rather than silently mixing them in.
    top_rung = max(Ns)
    names = [f"{nm} (N={n:,})" if n < top_rung else nm
             for nm, n in zip(names, Ns)]
    if OPTIMIZED:
        # measured == deployable; single series. Use a neutral proxy handle --
        # passing the per-point colour list would tint the legend key with one
        # regime's colour and collide with the regime keys below.
        ax.scatter(Ds, meas_s, s=110, c=cols, edgecolors="black", zorder=3)
        ax.scatter([], [], s=110, facecolors="white", edgecolors="black",
                   label="measured (= deployable)")
    else:
        ax.scatter(Ds, depl_s, s=110, c=cols, edgecolors="black", zorder=3,
                   label="deployable (scoring removed)")
        ax.scatter(Ds, meas_s, s=70, facecolors="none", edgecolors=cols,
                   zorder=3, label="measured (incl. O(N²) scoring)")
    ref_s = np.asarray(meas_s if OPTIMIZED else depl_s, float)
    ok = ref_s == ref_s
    ax.set_xscale("log"); ax.set_yscale("log")
    _annotate_spread(ax, Ds[ok], ref_s[ok],
                     [nm for nm, k in zip(names, ok) if k])
    ax.axhline(1.0, ls=":", color="black", lw=1.0)
    ax.set_xlabel("feature dimension D (log)")
    ax.set_ylabel(f"speedup vs cycle at k={k_ref} (log)")
    rho = _spearman(Ds[ok], ref_s[ok])
    # blue = responsive, orange = hard-middle
    ax.scatter([], [], c="#1f77b4", label="responsive")
    ax.scatter([], [], c="#ff7f0e", label="hard-middle")
    kind = "measured" if OPTIMIZED else "deployable"
    ax.set_title(f"Speed tracks D, not quality — {kind} speedup vs D  "
                 f"(Spearman ρ={rho:+.2f})")
    ax.legend(loc="best", fontsize=9)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / "speedup_vs_D")


def plot_budget_summary(datasets, k_ref=200, stacked=False):
    """Quality and speed together at one budget -- the paper figure.

    The two standalone plots each need a legend naming eight sources, which
    dominates them on its own. Pairing them lets one legend serve both and
    states the whole result in one place: at fixed k the overhead is flat in
    N while the speedup climbs linearly.

    `stacked` puts the panels one above the other for a single-column figure;
    the default places them side by side for a full-width one.
    """
    if stacked:
        fig, (ax_q, ax_s) = plt.subplots(2, 1, figsize=(6.4, 7.4))
    else:
        fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    rows = []
    for ds in datasets:
        df, meta = load(ds)
        st = meta.get("strain_top2_frac", float("nan"))
        D = int(meta.get("n_features", df["n_features"].iloc[0]))
        rows.append((st, ds, df, D))
    rows.sort(key=lambda r: -r[0])
    palette = plt.get_cmap("tab10").colors
    handles, labels = [], []
    top_rung = max(int(df["n_samples"].max()) for _, _, df, _ in rows)
    for i, (st, ds, df, D) in enumerate(rows):
        Ns, ov, sp = _budget_series(df, k_ref)
        if len(Ns) < 2:
            continue
        responsive = st == st and st >= 0.22
        c = palette[i % len(palette)]
        ls = "-" if responsive else "--"
        ax_q.plot(Ns, ov, "o", ls=ls, lw=2, color=c)
        ln, = ax_s.plot(Ns, sp, "o", ls=ls, lw=2, color=c)
        short = max(Ns) < top_rung
        handles.append(ln)
        labels.append(f"{ds.replace('_', ' ')} (D={D}, strain={st:.2f})"
                      + (f", N≤{max(Ns):,}" if short else ""))

    ax_q.axhline(1.0, ls=":", color="black", lw=1.2)
    ax_q.set_ylabel("stress overhead" if stacked
                    else "stress overhead  (budget / full sweep)")
    # Neutral descriptors only -- the interpretation belongs in the caption.
    ax_q.set_title(f"(a) stress overhead at $k={k_ref}$")

    # Reference slope: speedup should track N at fixed budget.
    lines = ax_s.get_lines()
    if lines:
        x0, y0 = lines[0].get_xdata()[0], lines[0].get_ydata()[0]
        xs = np.array([x0, top_rung], float)
        ref, = ax_s.plot(xs, y0 * xs / x0, ":", color="grey", lw=1.3)
        handles.append(ref); labels.append("∝ N (reference slope)")
    ax_s.axhline(1.0, ls=":", color="black", lw=1.0)
    ax_s.set_yscale("log")
    ax_s.set_ylabel("speedup (log)" if stacked
                    else "measured speedup vs full sweep (log)")
    ax_s.set_title(f"(b) measured speedup at $k={k_ref}$")

    for ax in (ax_q, ax_s):
        ax.set_xscale("log")
        ax.set_xticks([10000, 20000, 35000, 50000, 75000, 100000])
        ax.set_xticklabels(["10k", "20k", "35k", "50k"])
        ax.minorticks_off()
        ax.set_xlabel("N (subsample size, log)")

    # Linestyle key goes in the legend rather than a suptitle, so the figure
    # carries no assertion the caption does not make.
    key_r, = ax_q.plot([], [], "-", color="black", lw=2)
    key_h, = ax_q.plot([], [], "--", color="black", lw=2)
    handles += [key_r, key_h]
    labels += ["responsive ($\\mathrm{strain}_2 \\geq 0.22$)", "hard middle"]
    ncol = 1 if stacked else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               fontsize=8 if stacked else 9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.24 if stacked else 0.19, 1, 1.0))
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMBINED_DIR / (f"budget_summary_k{k_ref}"
                          + ("_stacked" if stacked else ""))
    fig.savefig(str(out) + ".png"); fig.savefig(str(out) + ".pdf")
    plt.close(fig); print(f"  wrote {out}.{{png,pdf}}")


def plot_overhead_vs_N_single_k(datasets, k_ref=200):
    """Cross-dataset constant-k plot: one budget, every source, one axis.

    The per-dataset grid shows all four budgets across eight panels; this
    collapses it to the single claim the paper makes -- at a fixed k the
    overhead is flat in N -- so responsive sources (flat and near 1.0) and
    hard-middle sources (flat but pinned high) can be read against each other.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    rows = []
    for ds in datasets:
        df, meta = load(ds)
        st = meta.get("strain_top2_frac", float("nan"))
        rows.append((st, ds, df, meta))
    rows.sort(key=lambda r: -r[0])
    # Regime is carried by linestyle, so colour is free to identify the source
    # -- with eight near-coincident lines a single colour per regime makes the
    # legend unreadable.
    palette = plt.get_cmap("tab10").colors
    for i, (st, ds, df, meta) in enumerate(rows):
        tbl = overhead_table(df)
        xs, ys = [], []
        for N in sorted(tbl):
            ks, ov = tbl[N]
            for k, o in zip(ks, ov):
                if int(k) == k_ref:
                    xs.append(N); ys.append(o)
        if not xs:
            continue
        responsive = st == st and st >= 0.22
        ax.plot(xs, ys, "-o", lw=2, color=palette[i % len(palette)],
                ls="-" if responsive else "--",
                label=f"{ds.replace('_', ' ')} (strain={st:.2f})")
    ax.axhline(1.0, ls=":", color="black", lw=1.2, label="full cycle (=1.0)")
    ax.set_xscale("log")
    ax.set_xticks([10000, 20000, 35000, 50000, 75000, 100000])
    ax.set_xticklabels(["10,000", "20,000", "35,000", "50,000"])
    ax.minorticks_off()
    ax.set_xlabel("N (subsample size, log)")
    ax.set_ylabel(f"stress overhead at k={k_ref}  (budget / full cycle)")
    ax.set_title(f"A fixed budget holds quality as N grows (k={k_ref})\n"
                 "solid = responsive (strain ≥ 0.22), dashed = hard middle")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / f"overhead_vs_N_k{k_ref}")


def plot_speedup_vs_N(datasets, k_ref=200):
    """Cross-dataset: measured speedup vs N at a fixed budget (the paper plot).

    Overhead is flat in N at fixed k while cycle cost grows as N^2, so the
    speedup grows ~linearly in N. D shifts each curve vertically (it sets the
    per-update cost) but does not flatten it: high D delays the win rather
    than capping it. Sources whose top rung is not the shared 50,000 (i.e.
    california_housing, run at its true full N=20,640) are drawn dashed so
    their shorter ladder is not read as a different slope.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cmap = cm.get_cmap("viridis")
    rows = []
    for ds in datasets:
        df, meta = load(ds)
        D = int(meta.get("n_features", df["n_features"].iloc[0]))
        rows.append((D, ds, df))
    rows.sort(key=lambda r: r[0])
    top_rung = max(int(df["n_samples"].max()) for _, _, df in rows)
    for i, (D, ds, df) in enumerate(rows):
        Ns = sorted(int(n) for n in df["n_samples"].unique())
        xs, ys = [], []
        for N in Ns:
            m, _ = _speedup_at(df, N, k_ref)
            if m == m:
                xs.append(N); ys.append(m)
        if len(xs) < 2:
            continue
        short = max(xs) < top_rung
        c = cmap(i / max(len(rows) - 1, 1))
        ax.plot(xs, ys, "--o" if short else "-o", color=c, lw=2,
                label=f"{ds.replace('_', ' ')} (D={D})"
                      + (f", N≤{max(xs):,}" if short else ""))
    # Linear-in-N reference anchored at the lowest-D curve's first point.
    lines = ax.get_lines()
    if lines:
        x0, y0 = lines[0].get_xdata()[0], lines[0].get_ydata()[0]
        xs = np.array([x0, top_rung], float)
        ax.plot(xs, y0 * xs / x0, ":", color="grey", lw=1.2, alpha=0.8,
                label="∝ N (reference slope)")
    ax.axhline(1.0, ls=":", color="black", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10000, 20000, 35000, 50000, 75000, 100000])
    ax.set_xticklabels(["10,000", "20,000", "35,000", "50,000"])
    ax.minorticks_off()
    ax.set_xlabel("N (subsample size, log)")
    ax.set_ylabel(f"measured speedup vs full cycle at k={k_ref} (log)")
    ax.set_title(f"Speedup grows ∝ N at fixed budget (k={k_ref})")
    # Outside the axes: eight curves plus a reference line cover the data
    # wherever matplotlib would otherwise place the box.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / "speedup_vs_N")


def plot_kstar_vs_strain(datasets, target=1.05):
    """Cross-dataset: budget needed for a quality target vs 2D-embeddability.

    The cleanest K result: strain_top2 sets the budget you need (independent of
    D), while D only sets what each unit of budget costs. Datasets that never
    reach the target are drawn at the top edge as open markers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys, cols, names, misses = [], [], [], [], []
    top_rungs = []
    for ds in datasets:
        df, meta = load(ds)
        N = int(df["n_samples"].max())
        top_rungs.append(N)
        tbl = overhead_table(df)
        st = meta.get("strain_top2_frac", float("nan"))
        kk = k_star(*tbl[N], target)
        c = "#1f77b4" if (st == st and st >= 0.22) else "#ff7f0e"
        if kk is None:
            misses.append((st, ds, c))
        else:
            xs.append(st); ys.append(kk); cols.append(c); names.append(ds)
    ax.scatter(xs, ys, s=120, c=cols, edgecolors="black", zorder=3)
    ax.set_yscale("log")
    # Sources that never reach the target have no k*; park them above the
    # highest real point, stacked so their labels stay legible.
    miss_xs, miss_ys, miss_names = [], [], []
    if ys:
        top = max(ys) * 1.6
        for j, (st, ds, c) in enumerate(sorted(misses)):
            y = top * (1.0 + 0.22 * j)
            ax.scatter([st], [y], s=120, facecolors="none", edgecolors=c,
                       zorder=3)
            miss_xs.append(st); miss_ys.append(y)
            miss_names.append(f"{ds} (never reaches)")
    _annotate_spread(ax, list(xs) + miss_xs, list(ys) + miss_ys,
                     list(names) + miss_names)
    rho = _spearman(xs, ys) if len(xs) >= 3 else float("nan")
    ax.set_xlabel("strain_top2  (2D-embeddability; higher = more responsive)")
    ax.set_ylabel(f"k* — budget needed for ≤ {target:.2f}× overhead (log)")
    ax.scatter([], [], c="#1f77b4", label="responsive")
    ax.scatter([], [], c="#ff7f0e", label="hard-middle")
    ax.scatter([], [], facecolors="none", edgecolors="grey",
               label="never reaches target")
    # Each point is at its own source's top rung; they mostly coincide, so
    # name the common one and flag it when they do not. Kept short -- the
    # long-form caveat belongs in the caption, not in an overflowing title.
    rung = (f"N={max(top_rungs):,}" if len(set(top_rungs)) == 1
            else f"N≤{max(top_rungs):,}")
    ax.set_title(f"$k^*$ vs. $\\mathrm{{strain}}_2$  "
                 f"(Spearman ρ={rho:+.2f}, {rung})")
    ax.legend(loc="best", fontsize=9)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    _save(fig, COMBINED_DIR / "kstar_vs_strain")


def main():
    argv = sys.argv[1:]
    if "--exp" in argv:
        i = argv.index("--exp")
        if i + 1 >= len(argv):
            raise SystemExit("--exp requires a results/<dirname> argument")
        _select_experiment(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    datasets = argv or get_all_datasets()
    if not datasets:
        print(f"No results in {RESULTS_DIR}. Run the experiment first.")
        return
    print(f"reading {RESULTS_DIR}"
          + ("  [optimized protocol: pure fit time]" if OPTIMIZED else ""))
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
        # k=200 is the ~+1% operating point the optimized run reports against;
        # plain K's existing figures are at k=50, so keep that there.
        k_ref = 200 if OPTIMIZED else 50
        plot_speedup_vs_D(datasets, k_ref=k_ref)  # D sets the cost per update
        plot_kstar_vs_strain(datasets)            # strain sets the budget needed
        if OPTIMIZED:
            plot_speedup_vs_N(datasets, k_ref=k_ref)  # and N sets how far it goes
            plot_overhead_vs_N_single_k(datasets, k_ref=k_ref)
            plot_budget_summary(datasets, k_ref=k_ref)  # both, paired
            plot_budget_summary(datasets, k_ref=k_ref, stacked=True)


if __name__ == "__main__":
    main()
