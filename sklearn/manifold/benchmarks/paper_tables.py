"""Paper tables for the three steps, from ``results_paper/analysis/``.

Step 1 is SMACOF vs full SGDMDS on the small suite (O). Steps 2 and 3 are
budgeted, then blocked, SGDMDS against full on the large suite (P). Every
ratio is against full SGDMDS's median on the same dataset and N. Run
``analyze_experiment_o_p_q.py`` first.

Usage:
    python paper_tables.py            # all tables as markdown
    python paper_tables.py --no-input # skip stress-1 (needs the datasets)
"""

import argparse

import numpy as np
import pandas as pd

import analyze_experiment_o_p_q as A
from dataset_tables import load_input, md, sum_sq_distances

AN = A.ROOT / "analysis"
P_N = (10_000, 20_000, 35_000, 50_000)
K = 200


def read(name):
    return pd.read_csv(AN / f"{name}.csv")


def n_label(n):
    return "20,640" if n == 20_640 else f"{n // 1000}k"


def col_of(n):
    """P's ladder column: california_housing's 20,640 sits in the 20k column."""
    return 20_000 if n == 20_640 else n


def x(v, digits=2):
    return A.fmt_x(v, digits)


def stress_cell(ratio, speedup):
    return f"{ratio:.3f} · {A.fmt_speed(speedup)}"


def step1(with_input=True):
    s = read("step1").sort_values("N")
    ttt = read("step1_ttt")
    ttt = ttt[ttt.config == "full"].set_index("dataset")
    rows = []
    for r in s.itertuples():
        row = {"dataset": r.dataset, "N": f"{r.N:,}", "D": f"{r.D:,}"}
        if with_input:
            total = sum_sq_distances(load_input(r.dataset, r.N, np.nan))
            for name, med in (("SMACOF", r.smacof_stress_scaled_med),
                              ("full", r.full_stress_scaled_med)):
                row[f"stress-1 {name}"] = f"{np.sqrt(med / total):.3f}"
        row["full ÷ SMACOF stress"] = f"{r.stress_ratio:.3f}"
        row["SMACOF iterations"] = (f"{r.smacof_iter_med:.0f} "
                                    f"({r.smacof_at_cap}/{r.n_seeds} at cap)")
        row["SMACOF time"] = A.fmt_time(r.smacof_time_fit_med)
        row["full time"] = A.fmt_time(r.full_time_fit_med)
        row["speedup"] = A.fmt_speed(r.speedup)
        t = ttt.loc[r.dataset]
        row["to SMACOF's stress"] = A.fmt_speed(t.speedup_vs_smacof)
        rows.append(row)
    out = pd.DataFrame(rows)
    summary = {c: "" for c in out.columns}
    summary.update({
        "dataset": "**median**",
        "full ÷ SMACOF stress": f"**{s.stress_ratio.median():.3f}**",
        "speedup": f"**{x(s.speedup.median())}**",
        "to SMACOF's stress": f"**{x(ttt.speedup_vs_smacof.median())}**"})
    return pd.concat([out, pd.DataFrame([summary])], ignore_index=True)


def p_grid(arm, k=K):
    """Sources × N at one k: stress ÷ full · speedup over full."""
    v = read("vs_full_p")
    v = v[(v.arm == arm) & (v.k == k)]
    full = A.load_results("p")
    full = full[full.config == "full"].groupby(A.KEY).time_fit.median()
    rows = []
    for (dataset, d), g in v.groupby(["dataset", "n_features"]):
        row = {"dataset": dataset, "D": f"{d:,}"}
        for n in P_N:
            hit = g[g.n_samples.map(col_of) == n]
            row[n_label(n)] = ("—" if hit.empty else
                               stress_cell(hit.stress_ratio_med.iloc[0],
                                           hit.speedup_med.iloc[0]))
        top = g.n_samples.max()
        row["full time at top N"] = A.fmt_time(full[(dataset, top)])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("D", key=lambda c: c.str.replace(",", "")
                                          .astype(int))


def p_per_k(arms=("budget", "blocked")):
    """Median (range) over sources of stress ÷ full and speedup, per k and N."""
    v = read("vs_full_p")
    v = v.assign(col=v.n_samples.map(col_of))
    rows = []
    for arm in arms:
        for k in sorted(v.k.dropna().unique()):
            row = {"arm": arm, "k": f"{k:.0f}"}
            for n in P_N:
                g = v[(v.arm == arm) & (v.k == k) & (v.col == n)]
                row[n_label(n)] = (
                    f"{g.stress_ratio_med.median():.3f} · "
                    f"{x(g.speedup_med.median(), 1)} "
                    f"({x(g.speedup_med.min(), 1)}–{x(g.speedup_med.max(), 1)})")
            rows.append(row)
    return pd.DataFrame(rows)


def o_by_k():
    """Small suite: budgeted and blocked ÷ full at every k (stress · speedup)."""
    v = read("vs_full_o")
    full = read("step1").set_index("dataset")
    rows = []
    for dataset in full.sort_values("N").index:
        f = full.loc[dataset]
        for arm in ("budget", "blocked"):
            g = v[(v.dataset == dataset) & (v.arm == arm)].set_index("k")
            row = {"dataset": dataset, "N": f"{f.N:,}", "D": f"{f.D:,}",
                   "full time": A.fmt_time(f.full_time_fit_med),
                   "arm": {"budget": "budgeted", "blocked": "blocked"}[arm]}
            for k in (50, 100, 200, 400):
                row[f"k={k}"] = ("—" if k not in g.index else
                                 stress_cell(g.loc[k].stress_ratio_med,
                                             g.loc[k].speedup_med))
            rows.append(row)
    return pd.DataFrame(rows)


def step3_top(with_squad=True):
    """Each source at its largest N: full, budgeted and blocked at k=200."""
    v = read("vs_full_p")
    b = read("blocked_p")
    sq = read("squad_best_p")
    full = A.load_results("p")
    full = full[full.config == "full"].groupby(A.KEY).time_fit.median()
    rows = []
    for dataset, g in v.groupby("dataset"):
        n = g.n_samples.max()
        g = g[(g.n_samples == n) & (g.k == K)].set_index("arm")
        bb = b[(b.dataset == dataset) & (b.n_samples == n) & (b.k == K)].iloc[0]
        row = {"dataset": dataset, "N": n_label(n),
               "D": f"{int(g.n_features.iloc[0]):,}",
               "full time": A.fmt_time(full[(dataset, n)]),
               "budgeted": stress_cell(g.loc["budget"].stress_ratio_med,
                                       g.loc["budget"].speedup_med),
               "blocked": stress_cell(g.loc["blocked"].stress_ratio_med,
                                      g.loc["blocked"].speedup_med),
               "blocked ÷ budgeted stress": f"{bb.stress_ratio_med:.3f}",
               "blocked over budgeted": A.fmt_speed(bb.speedup_med)}
        if with_squad:
            s = sq[(sq.dataset == dataset) & (sq.n_samples == n)].iloc[0]
            row["SQuaD-MDS best"] = stress_cell(
                s.stress_ratio_med, full[(dataset, n)] / s.time_fit_med)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("D", key=lambda c: c.str.replace(",", "").astype(int))


def step3_targets(factors=(1.01, 1.05)):
    """Each source at its largest N: speedup over the full run to reach a target."""
    t = read("ttt_p")
    rows = []
    for dataset, g in t.groupby("dataset"):
        n = g.n_samples.max()
        g = g[g.n_samples == n]
        row = {"dataset": dataset, "N": n_label(n)}
        for f in factors:
            for arm in ("full", "budget", "blocked", "squad"):
                h = g[(g.factor == f) & (g.arm == arm)]
                if h.empty or not np.isfinite(h.ttt_med.iloc[0]):
                    cell = "—"
                else:
                    h = h.iloc[0]
                    k = ("" if arm in ("full", "squad")
                         else f" (k{h.config.split('-')[1][1:]})")
                    cell = x(h.speedup_vs_full) + k
                row[f"{f:.2f}× {arm}"] = cell
        rows.append(row)
    d = A.load_results("p").groupby("dataset").n_features.first()
    out = pd.DataFrame(rows)
    return (out.assign(D=out.dataset.map(d)).sort_values("D").drop(columns="D"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-input", action="store_true")
    args = parser.parse_args()
    tables = {
        "step1": step1(not args.no_input),
        "step2_grid": p_grid("budget"),
        "step2_per_k": p_per_k(("budget",)),
        "step3_grid": p_grid("blocked"),
        "step3_per_k": p_per_k(("blocked",)),
        "step3_top": step3_top(),
        "step3_targets": step3_targets(),
        "o_by_k": o_by_k(),
    }
    for name, table in tables.items():
        print(f"<!-- {name} -->\n{md(table, left=('dataset', 'arm'))}\n")


if __name__ == "__main__":
    main()
