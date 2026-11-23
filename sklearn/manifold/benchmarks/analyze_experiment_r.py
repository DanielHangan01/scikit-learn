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

Writes to ``results_paper/analysis_r/``, with ``tables.md``.

Usage:
    python analyze_experiment_r.py
"""

import pandas as pd

import analyze_experiment_o_p_q as A

A.FOLDERS.update(r_o="experiment_r_o", r_p="experiment_r_p")
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
    print(f"Wrote {out}")
    return T


if __name__ == "__main__":
    run()
