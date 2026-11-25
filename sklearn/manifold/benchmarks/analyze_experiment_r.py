"""Analysis of Experiment R: Enhanced Force-Scheme against O's and P's arms.

R's fits are read next to O's (small suite) and P's (large suite), which share
their inputs and seeds. Every ratio is against full SGDMDS's medians on the
same dataset and N, as in ``analyze_experiment_o_p_q``, whose helpers this
reuses; SMACOF (random init) and blocked SGDMDS at k=200 are shown alongside.

- ``r_small.csv`` / ``r_large.csv``: per dataset (and N), every arm's median
  stress ÷ full, speedup over full, iterations and optimal scale.
- ``r_summary.csv``: per suite and arm, medians over datasets, and on how many
  the arm has lower stress than full or is faster than it.
- ``r_ttt.csv``: time for each EFS fit to reach 1.01x, 1.05x and 1.10x full's
  final stress, from the histories (P's and R's large ones are estimated).
- ``comparison.md``: per dataset, mean stress and time of full SGDMDS against
  full EFS, and of blocked SGDMDS (k=200) against s-EFS, at 1 and 8 threads.

Writes to ``results_paper/analysis_r/``, with ``tables.md``.

Usage:
    python analyze_experiment_r.py
"""

import numpy as np
import pandas as pd

import analyze_experiment_o_p_q as A

A.FOLDERS.update(r_o="experiment_r_o", r_p="experiment_r_p",
                 r_epochs="experiment_r_epochs")
SUITES = {"small": ("o", "r_o"), "large": ("p", "r_p")}
REFERENCE = {"small": ["smacof-random", "blocked-k200-m16"],
             "large": ["blocked-k200-m16"]}
FACTORS = (1.01, 1.05, 1.10)


def is_efs(config):
    return config == "efs" or config.startswith("efs-")


def load(suite, root=A.ROOT):
    """Full SGDMDS, the reference arms and every EFS arm of one suite."""
    base, r = SUITES[suite]
    res = pd.concat([A.load_results(base, root), A.load_results(r, root)],
                    ignore_index=True)
    keep = (res.config == "full") | res.config.isin(REFERENCE[suite]) | (
        res.config.map(is_efs))
    res = A.add_ratios_to_full(res[keep])
    efs_rungs = res.loc[res.config.map(is_efs), A.KEY].drop_duplicates()
    res = res.merge(efs_rungs, on=A.KEY)
    return res[res.full_stress.notna()]


def per_dataset(res):
    """Median stress ratio, speedup, iterations and scale per (dataset, N, arm)."""
    if res.empty:
        return pd.DataFrame()
    s = A.summarize(res, ["stress_ratio", "speedup", "n_iter_run",
                          "scale_alpha"], A.KEY + ["n_features", "config"])
    return s.assign(verdict=A.time_verdict(s.speedup_med))


def summary(table, suite):
    rows = []
    for config, g in table.groupby("config"):
        if config == "full":
            continue
        rows.append(dict(
            suite=suite, config=config, rungs=len(g),
            stress_ratio_med=g.stress_ratio_med.median(),
            stress_ratio_min=g.stress_ratio_med.min(),
            stress_ratio_max=g.stress_ratio_med.max(),
            lower_stress=int((g.stress_ratio_med < 1).sum()),
            speedup_med=g.speedup_med.median(),
            speedup_min=g.speedup_med.min(),
            speedup_max=g.speedup_med.max(),
            faster=int((g.verdict == "faster").sum()),
            slower=int((g.verdict == "slower").sum()),
            n_iter_med=g.n_iter_run_med.median()))
    return pd.DataFrame(rows)


def time_to_full(suite, root=A.ROOT):
    """Per EFS fit, the time to reach each factor times full's final stress."""
    base, r = SUITES[suite]
    targets = A.load_histories(base, root)
    hist = A.load_histories(r, root)
    out = [A.add_time_to_target(hist, A.full_targets(targets, f),
                                "scaled").assign(factor=f) for f in FACTORS]
    ttt = pd.concat(out, ignore_index=True)
    ttt = ttt[ttt.target.notna()]
    if ttt.empty:
        return pd.DataFrame()
    s = A.summarize_ttt(ttt, A.KEY + ["config", "factor"])
    return s.assign(suite=suite)


def render(T):
    def cell(r):
        return f"{r.stress_ratio_med:.3f} · {A.fmt_speed(r.speedup_med)}"

    parts = ["# Experiment R: Enhanced Force-Scheme\n",
             "Cells: stress ÷ full SGDMDS · speedup over full SGDMDS (medians "
             "over seeds). Arms ending in -t8 ran on 8 threads.\n"]
    for suite in ("small", "large"):
        t = T[f"r_{suite}"]
        if t.empty:
            continue
        t = t[t.config != "full"]
        wide = t.assign(cell=[cell(r) for r in t.itertuples()]).pivot_table(
            index=A.KEY + ["n_features"], columns="config", values="cell",
            aggfunc="first").reset_index()
        wide.columns.name = None
        parts += [f"## {suite} suite\n", A.md(wide.rename(columns={
            "n_samples": "N", "n_features": "D"}).fillna("—")), ""]
    s = T["r_summary"].copy()
    for c in s.columns:
        if c.startswith("stress_ratio"):
            s[c] = s[c].map("{:.3f}".format)
        elif c.startswith("speedup"):
            s[c] = s[c].map(A.fmt_x)
    parts += ["## Summary per arm\n", A.md(s), ""]
    tt = T["r_ttt"]
    if not tt.empty:
        g = tt.groupby(["suite", "config", "factor"]).agg(
            reached=("n_reached", "sum"), fits=("n_fits", "sum"),
            ttt_med=("ttt_med", "median")).reset_index()
        g["ttt_med"] = g.ttt_med.map(A.fmt_time)
        parts += ["## Time to reach a factor of full's final stress\n",
                  "Median over datasets of each one's median time; "
                  "reached counts fits.\n", A.md(g), ""]
    return "\n".join(parts)


def fmt_seconds(t):
    if t < 10:
        return f"{t:.2f} s"
    return f"{t:.1f} s" if t < 100 else f"{t:,.0f} s"


_HISTORIES = {}


def histories(exp, root=A.ROOT):
    """``A.load_histories``, read once per experiment."""
    if (exp, root) not in _HISTORIES:
        _HISTORIES[exp, root] = A.load_histories(exp, root)
    return _HISTORIES[exp, root]


def time_ratio_to_higher(ours, efs):
    """Median over seeds of ours' time ÷ EFS's time to reach a shared level.

    ``ours`` and ``efs`` are the histories of two arms on one (dataset, N).
    Fits are paired by seed, and each pair's level is the higher of its two
    final stresses, so both fits reach it.
    """
    efs = efs.set_index("seed")
    ratios = []
    for h in ours.itertuples():
        if h.seed not in efs.index:
            continue
        e = efs.loc[h.seed]
        level = max(h.scaled[-1], e.scaled[-1])
        ratios.append(A.time_to_target(h.time, h.scaled, level)
                      / A.time_to_target(e.time, e.scaled, level))
    return float(np.median(ratios)), len(ratios)


def paired_table(suite, ours, efs, our_name, efs_name, root=A.ROOT):
    """Per (dataset, N): mean stress and time of one of our arms and an EFS arm.

    EFS's stress doesn't depend on its thread count, so it is the 1-thread
    arm's; the times are both arms'. The last column compares the times both
    take (ours on 1 thread, EFS on 8) to reach a level both reach: per seed, the
    higher of the two final stresses (``time_ratio_to_higher``). It reads the
    histories, estimated from P's pair sample on the large suite, for both.
    """
    base, r = SUITES[suite]
    res = pd.concat([A.load_results(base, root), A.load_results(r, root)])
    hist = pd.concat([histories(base, root), histories(r, root)])
    m = res.groupby(A.KEY + ["n_features", "config"]).agg(
        t=("time_fit", "mean"), s=("stress_scaled", "mean")).reset_index()
    rows = []
    for (dataset, n, d), g in m.groupby(["dataset", "n_samples", "n_features"]):
        g = g.set_index("config")
        if not {ours, efs, f"{efs}-t8"} <= set(g.index):
            continue
        o, e, e8 = g.loc[ours], g.loc[efs], g.loc[f"{efs}-t8"]
        h = hist[(hist.dataset == dataset) & (hist.n_samples == n)]
        ratio, _ = time_ratio_to_higher(h[h.config == ours],
                                        h[h.config == f"{efs}-t8"])
        rows.append({
            "dataset": dataset, "N": f"{n:,}", "D": f"{d:,}",
            f"{our_name} stress": f"{o.s:.4g}", f"{efs_name} stress": f"{e.s:.4g}",
            f"{our_name} / {efs_name} stress": f"{o.s / e.s:.3f}",
            f"{our_name} time": fmt_seconds(o.t),
            f"{efs_name} 1t time": fmt_seconds(e.t),
            f"{efs_name} 8t time": fmt_seconds(e8.t),
            f"{our_name} / {efs_name} 8t time": f"{o.t / e8.t:.2f}",
            f"{our_name} / {efs_name} 8t time to the higher stress": f"{ratio:.2f}",
            "_n": n})
    out = pd.DataFrame(rows)
    return out.sort_values(["_n", "dataset"]).drop(columns="_n")


def epochs_table(root=A.ROOT):
    """Small suite: full SGDMDS at each epoch count against full EFS.

    Mean stress and time over seeds; stress as ours ÷ EFS's, so below 1 ours is
    lower. Empty until the ``epochs`` suite of Experiment R has run.
    """
    if not list((root / A.FOLDERS["r_epochs"]).glob("*/results.csv")):
        return pd.DataFrame()
    res = pd.concat([A.load_results("r_epochs", root), A.load_results("r_o", root)])
    m = res.groupby(["dataset", "n_samples", "config"]).agg(
        t=("time_fit", "mean"), s=("stress_scaled", "mean")).reset_index()
    epochs = sorted({int(c.split("-e")[1]) for c in m.config
                     if c.startswith("full-e")})
    rows = []
    for (dataset, n), g in m.groupby(["dataset", "n_samples"]):
        g = g.set_index("config")
        if not {"efs", "efs-t8", *(f"full-e{e}" for e in epochs)} <= set(g.index):
            continue
        row = {"dataset": dataset, "N": f"{n:,}", "EFS stress": f"{g.loc['efs'].s:.4g}"}
        for e in epochs:
            ratio = g.loc[f"full-e{e}"].s / g.loc["efs"].s
            row[f"{e} epochs / EFS stress"] = f"{ratio:.3f}"
        for e in epochs:
            row[f"{e} epochs time"] = fmt_seconds(g.loc[f"full-e{e}"].t)
        row["EFS 1t time"] = fmt_seconds(g.loc["efs"].t)
        row["EFS 8t time"] = fmt_seconds(g.loc["efs-t8"].t)
        row["_n"] = n
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=list(rows[0]))
    return out.sort_values(["_n", "dataset"]).drop(columns="_n")


def comparison_tables(root=A.ROOT):
    full = ("full", "efs", "full SGDMDS", "EFS")
    blocked = ("blocked-k200-m16", "efs-sample", "blocked", "s-EFS")
    tables = {"full_small": paired_table("small", *full, root=root),
              "full_large": paired_table("large", *full, root=root),
              "blocked_small": paired_table("small", *blocked, root=root),
              "blocked_large": paired_table("large", *blocked, root=root)}
    epochs = epochs_table(root)
    if not epochs.empty:
        tables["epochs_small"] = epochs
    return tables


def run(root=A.ROOT):
    out = root / "analysis_r"
    T = {}
    for suite in SUITES:
        if not list((root / A.FOLDERS[SUITES[suite][1]]).glob("*/results.csv")):
            T[f"r_{suite}"] = pd.DataFrame()
            continue
        T[f"r_{suite}"] = per_dataset(load(suite, root))
    if all(T[f"r_{s}"].empty for s in SUITES):
        raise SystemExit(f"No Experiment R results under {root}.")
    out.mkdir(parents=True, exist_ok=True)
    T["r_summary"] = pd.concat([summary(T[f"r_{s}"], s) for s in SUITES
                                if not T[f"r_{s}"].empty], ignore_index=True)
    T["r_ttt"] = pd.concat([time_to_full(s, root) for s in SUITES
                            if not T[f"r_{s}"].empty], ignore_index=True)
    for name, table in T.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "tables.md").write_text(render(T))
    C = comparison_tables(root)
    (out / "comparison.md").write_text("\n\n".join(
        f"<!-- {name} -->\n{A.md(table)}" for name, table in C.items()) + "\n")
    print(f"Wrote {out}")
    return T


if __name__ == "__main__":
    run()
