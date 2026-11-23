"""
Analysis of Experiments O, P and Q: the tables of EXPERIMENT_O_P_Q_RESULTS.md and
tidy data for the figures.

Reads ``experiment_{o,p,q}/<dataset>/{results.csv,histories.jsonl}`` under
``SGDMDS_RESULTS_ROOT`` (default ``benchmarks/results_paper``) and writes CSV
tables plus ``tables.md`` (the doc's tables) to ``<root>/analysis/``, or to
``--out``. It never writes into the experiment folders. Every table is also a
function, so plotters can import this module and work from its DataFrames.

Conventions (EXPERIMENT_O_P_Q_RESULTS.md, section 0):

- Seeds are summarised by median and IQR (linear quantiles).
- Tables use scale-optimal stress (``stress_scaled``). Step 1's time to target
  uses raw stress, because SMACOF's own history is raw; alpha is within 0.002
  of 1 for SMACOF and SGDMDS, so the two agree.
- Ratios to full SGDMDS divide each fit by the full arm's median at the same
  dataset and N. P's full arm has one seed above N=10k (three at 10k).
- A time ratio within ``COMPARABLE`` of 1 is "comparable" (the campaign's 15%
  rule), not a win for either side.
- Time to target is the first crossing in a fit's history, interpolated
  linearly between the two snapshots that bracket it; ``inf`` if never
  reached. A config's median counts only if more than half its seeds reach
  the target. P's histories are pair estimates, so P's targets come from the
  full arm's estimated final stress on the same pairs.
- SQuaD-MDS's setting is chosen on ``SELECT_SEEDS`` and reported on
  ``REPORT_SEEDS`` in O. P has 5 seeds and 12 settings, so it is chosen and
  reported on the same seeds. SGDMDS arms use every seed.

Usage:
    python analyze_experiment_o_p_q.py
    python analyze_experiment_o_p_q.py --out /tmp/analysis
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

BENCH_DIR = Path(__file__).parent
ROOT = Path(os.environ.get("SGDMDS_RESULTS_ROOT", BENCH_DIR / "results_paper"))
FOLDERS = {"o": "experiment_o", "p": "experiment_p", "q": "experiment_q"}
KEY = ["dataset", "n_samples"]
COMPARABLE = 1.15
TARGET_FACTORS = (1.01, 1.02, 1.05, 1.10)
SMACOF_CAP = 300
SELECT_SEEDS = list(range(42, 47))
REPORT_SEEDS = list(range(47, 52))
SQUAD_TOP = {"o": 8000, "p": 4000}
SQUAD_DEFAULT = {"o": "squad-i1000-lr550-exa1-pca-capnone",
                 "p": "squad-i1000-lr550-exa1-pca-cap4"}
SGD_ARMS = ("full", "budget", "blocked")
ARMS = SGD_ARMS + ("squad",)


def arm_of(config):
    """Arm of a config id: full, budget, blocked, squad or the SMACOF config."""
    for arm in ("budget", "blocked", "squad"):
        if config.startswith(arm + "-"):
            return arm
    return config


def squad_n_iter(config):
    """``n_iter`` of a SQuaD-MDS config id such as ``squad-i8000-lr150-...``."""
    return int(config.split("-")[1][1:])


def load_results(exp, root=ROOT):
    """All fits of one experiment, one row each, with ``arm`` and ``exp``."""
    paths = sorted((root / FOLDERS[exp]).glob("*/results.csv"))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.assign(arm=df.config.map(arm_of), exp=exp)
    return df.reset_index(drop=True)


def load_histories(exp, root=ROOT):
    """One row per fit: ``time``, ``raw`` and ``scaled`` stress as arrays."""
    rows = []
    for path in sorted((root / FOLDERS[exp]).glob("*/histories.jsonl")):
        with open(path) as f:
            for line in f:
                h = json.loads(line)
                scaled = h["stress_scaled"]
                rows.append(dict(
                    dataset=path.parent.name, n_samples=h["n_samples"],
                    config=h["config"], seed=h["seed"],
                    time=np.asarray(h["time"], dtype=float),
                    raw=np.asarray(h["stress_raw"], dtype=float),
                    scaled=None if scaled is None else np.asarray(scaled, float),
                    estimated=bool(h.get("estimated"))))
    hist = pd.DataFrame(rows)
    return hist.assign(arm=hist.config.map(arm_of))


def quantile(values, q):
    """Linear quantile that stays ``inf`` when a bracketing value is ``inf``."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan
    pos = q * (v.size - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if np.isinf(v[hi]) and pos > lo:
        return v[hi]
    if np.isinf(v[lo]):
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(df, cols, by):
    """Median, 25th and 75th percentile of ``cols`` per group, and the fits."""
    out = []
    for keys, g in df.groupby(by, sort=True):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for c in cols:
            row[f"{c}_med"] = quantile(g[c], 0.5)
            row[f"{c}_q25"] = quantile(g[c], 0.25)
            row[f"{c}_q75"] = quantile(g[c], 0.75)
        row["n_fits"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


def time_verdict(speedup):
    """'faster', 'comparable' or 'slower' for a speedup under the 15% rule."""
    speedup = np.asarray(speedup, dtype=float)
    return np.where(speedup >= COMPARABLE, "faster",
                    np.where(speedup <= 1 / COMPARABLE, "slower", "comparable"))


def lookup(series, frame):
    """``series`` (indexed by KEY) aligned to the rows of ``frame``."""
    return series.reindex(pd.MultiIndex.from_frame(frame[KEY])).to_numpy()


def add_ratios_to_full(df, stress="stress_scaled"):
    """Per-fit stress ratio and speedup against full SGDMDS's medians."""
    full = (df[df.arm == "full"].groupby(KEY)
            .agg(full_stress=(stress, "median"), full_time=("time_fit", "median"),
                 full_seeds=("seed", "nunique")).reset_index())
    df = df.drop(columns=[c for c in full.columns if c not in KEY], errors="ignore")
    df = df.merge(full, on=KEY, how="left")
    return df.assign(stress_ratio=df[stress] / df.full_stress,
                     speedup=df.full_time / df.time_fit)


def time_to_target(time, stress, target, interpolate=True):
    """Time of the first crossing of ``target`` (``inf`` if never reached)."""
    below = np.flatnonzero(stress <= target)
    if below.size == 0:
        return np.inf
    i = below[0]
    if i == 0 or not interpolate:
        return float(time[i])
    s0, s1, t0, t1 = stress[i - 1], stress[i], time[i - 1], time[i]
    return float(t0 + (t1 - t0) * (s0 - target) / (s0 - s1))


def add_time_to_target(hist, targets, column):
    """Per history row: ``ttt`` (interpolated), ``ttt_snapshot`` and ``target``.

    ``targets`` is a Series indexed by KEY. ``ttt_snapshot`` is the time of
    the first snapshot at or below the target, an upper bound on ``ttt``.
    """
    tgt = lookup(targets, hist)
    ttt = [time_to_target(t, s, g) for t, s, g in zip(hist.time, hist[column], tgt)]
    snap = [time_to_target(t, s, g, interpolate=False)
            for t, s, g in zip(hist.time, hist[column], tgt)]
    return hist.assign(ttt=ttt, ttt_snapshot=snap, target=tgt)


def summarize_ttt(hist, by):
    """Median and IQR of the time to target, and how many fits reached it."""
    out = summarize(hist, ["ttt"], by)
    reached = hist.assign(r=np.isfinite(hist.ttt)).groupby(by).r.sum()
    return out.merge(reached.rename("n_reached").reset_index(), on=by)


def step1_table(o):
    """SMACOF (random init) vs full SGDMDS per dataset of O."""
    rows = []
    for name, g in o.groupby("dataset"):
        sm = g[g.config == "smacof-random"]
        fu = g[g.config == "full"]
        sm_med = sm.stress_scaled.median()
        row = dict(dataset=name, N=g.n_samples.iloc[0], D=g.n_features.iloc[0])
        for tag, part in (("smacof", sm), ("full", fu)):
            for c in ("stress_scaled", "time_fit"):
                for q, suffix in ((0.5, "med"), (0.25, "q25"), (0.75, "q75")):
                    row[f"{tag}_{c}_{suffix}"] = quantile(part[c], q)
        row["stress_ratio"] = fu.stress_scaled.median() / sm_med
        row["stress_ratio_q25"] = fu.stress_scaled.quantile(0.25) / sm_med
        row["stress_ratio_q75"] = fu.stress_scaled.quantile(0.75) / sm_med
        row["smacof_iqr_rel"] = (sm.stress_scaled.quantile(0.75)
                                 - sm.stress_scaled.quantile(0.25)) / sm_med
        row["stress_p"] = mannwhitneyu(fu.stress_scaled, sm.stress_scaled).pvalue
        row["full_lower_all_seeds"] = fu.stress_scaled.max() < sm.stress_scaled.min()
        row["speedup"] = row["smacof_time_fit_med"] / row["full_time_fit_med"]
        row["smacof_iter_med"] = sm.n_iter_run.median()
        row["smacof_at_cap"] = int((sm.n_iter_run >= SMACOF_CAP).sum())
        row["n_seeds"] = len(sm)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    return out.assign(verdict=time_verdict(out.speedup))


def step1_tests(table):
    """Wilcoxon signed-rank across datasets of the per-dataset medians."""
    return dict(n=len(table), stress_p=wilcoxon(np.log(table.stress_ratio)).pvalue,
                time_p=wilcoxon(np.log(table.speedup)).pvalue,
                stress_ratio_median=table.stress_ratio.median(),
                speedup_median=table.speedup.median())


def step1_time_to_smacof(df, hist, configs=("full", "budget-k200",
                                            "blocked-k200-m16")):
    """How soon SGDMDS reaches SMACOF's final raw stress, and the converse.

    Returns the summary per (dataset, N, config), the converse (SMACOF's time
    to full SGDMDS's median final stress) and the per-fit rows.
    """
    sm = df[df.config == "smacof-random"]
    target = sm.groupby(KEY).stress_raw.median()
    h = add_time_to_target(hist[hist.config.isin(configs)], target, "raw")
    out = summarize_ttt(h, KEY + ["config"])
    out["smacof_time"] = lookup(sm.groupby(KEY).time_fit.median(), out)
    ok = out.n_reached * 2 > out.n_fits
    out["speedup_vs_smacof"] = np.where(ok, out.smacof_time / out.ttt_med, np.nan)
    full_target = df[df.config == "full"].groupby(KEY).stress_raw.median()
    hs = add_time_to_target(hist[hist.config == "smacof-random"], full_target,
                            "raw")
    return out, summarize_ttt(hs, KEY + ["config"]), h


def classical_smacof_table(o):
    """Classical-init SMACOF against full SGDMDS and random-init SMACOF."""
    rows = []
    for name, g in o.groupby("dataset"):
        cl = g[g.config == "smacof-classical_mds"]
        fu = g[g.config == "full"]
        sm = g[g.config == "smacof-random"]
        c = cl.stress_scaled.median()
        rows.append(dict(
            dataset=name, N=g.n_samples.iloc[0], D=g.n_features.iloc[0],
            classical_stress=c,
            classical_stress_spread=cl.stress_scaled.max() / cl.stress_scaled.min(),
            vs_full=c / fu.stress_scaled.median(),
            full_q25=fu.stress_scaled.quantile(0.25) / c,
            full_q75=fu.stress_scaled.quantile(0.75) / c,
            beats_all_full_seeds=c < fu.stress_scaled.min(),
            vs_smacof_random=c / sm.stress_scaled.median(),
            classical_time=cl.time_fit.median(),
            full_time=fu.time_fit.median(),
            speedup_full=cl.time_fit.median() / fu.time_fit.median(),
            classical_iter=cl.n_iter_run.median(),
            classical_at_cap=int((cl.n_iter_run >= SMACOF_CAP).sum())))
    out = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    return out.assign(verdict=time_verdict(out.speedup_full))


def vs_full_table(df, arms=("budget", "blocked")):
    """Stress overhead and speedup against full SGDMDS per (dataset, N, config)."""
    df = add_ratios_to_full(df)
    sub = df[df.arm.isin(arms)]
    out = summarize(sub, ["stress_ratio", "speedup", "time_fit"],
                    KEY + ["n_features", "arm", "k", "config"])
    out["full_seeds"] = lookup(df.groupby(KEY).full_seeds.first(), out)
    return out.assign(verdict=time_verdict(out.speedup_med))


def q_table(q):
    """Q's time and stress per (dataset, N, config), against full SGDMDS."""
    q = add_ratios_to_full(q)
    out = summarize(q, ["time_fit", "stress_ratio", "speedup", "n_iter_run"],
                    KEY + ["n_features", "config"])
    return out.assign(verdict=time_verdict(out.speedup_med))


def q_slopes(qt, n_min=1000):
    """Log-log slope of median time against N per (dataset, config)."""
    rows = []
    for (name, cfg), g in qt[qt.n_samples >= n_min].groupby(["dataset", "config"]):
        slope = np.polyfit(np.log(g.n_samples), np.log(g.time_fit_med), 1)[0]
        rows.append(dict(dataset=name, config=cfg, slope=slope))
    return pd.DataFrame(rows)


def q_break_even(qt):
    """Smallest N from which each arm is faster than full SGDMDS at every N.

    For ``smacof-random`` that is where SMACOF overtakes full SGDMDS.
    """
    rows = []
    for (name, cfg), g in qt[qt.config != "full"].groupby(["dataset", "config"]):
        g = g.sort_values("n_samples")
        faster = (g.verdict == "faster").to_numpy()
        n_from = np.nan
        for i in range(len(g)):
            if faster[i:].all():
                n_from = g.n_samples.iloc[i]
                break
        rows.append(dict(dataset=name, D=g.n_features.iloc[0], config=cfg,
                         faster_from_N=n_from,
                         comparable_at=list(g.n_samples[g.verdict == "comparable"])))
    return pd.DataFrame(rows)


def cost_per_pair(results, hists):
    """Per SGDMDS fit: setup time, time per epoch and ns per pair per epoch.

    Setup is everything before the first snapshot (for the full sweep, the
    distance matrix and the pair arrays). SMACOF's cost is per pair per
    iteration over the whole fit.
    """
    rows = []
    for exp, df in results.items():
        h = hists[exp]
        h = h[h.arm.isin(SGD_ARMS)]
        cols = KEY + ["n_features", "config", "seed", "k", "time_fit"]
        h = h.merge(df[cols], on=KEY + ["config", "seed"])
        n = h.n_samples.to_numpy(dtype=float)
        pairs = np.where(h.arm == "full", n * (n - 1) / 2, h.k * n)
        epoch = np.array([(t[-1] - t[0]) / (len(t) - 1) for t in h.time])
        rows.append(pd.DataFrame(dict(
            exp=exp, dataset=h.dataset, n_samples=h.n_samples,
            n_features=h.n_features, arm=h.arm, config=h.config, seed=h.seed,
            setup=[t[0] for t in h.time], epoch=epoch, ns_per_pair=1e9 * epoch / pairs,
            time_fit=h.time_fit)))
        sm = df[df.config == "smacof-random"]
        if len(sm):
            n = sm.n_samples.to_numpy(dtype=float)
            rows.append(pd.DataFrame(dict(
                exp=exp, dataset=sm.dataset, n_samples=sm.n_samples,
                n_features=sm.n_features, arm="smacof-random",
                config=sm.config, seed=sm.seed, setup=np.nan,
                epoch=sm.time_fit / sm.n_iter_run,
                ns_per_pair=1e9 * sm.time_fit / (sm.n_iter_run * n * (n - 1) / 2),
                time_fit=sm.time_fit)))
    fits = pd.concat(rows, ignore_index=True)
    return fits, summarize(fits, ["ns_per_pair", "setup", "epoch"],
                           ["exp", "dataset", "n_samples", "n_features", "arm"])


def blocked_vs_budget(df):
    """Blocked against budgeted SGDMDS at the same k, paired by seed."""
    sgd = df[df.arm.isin(["budget", "blocked"])]
    wide = sgd.pivot_table(index=KEY + ["n_features", "k", "seed"], columns="arm",
                           values=["stress_scaled", "time_fit"]).dropna()
    pairs = pd.DataFrame({
        "stress_ratio": (wide["stress_scaled"]["blocked"]
                         / wide["stress_scaled"]["budget"]),
        "speedup": wide["time_fit"]["budget"] / wide["time_fit"]["blocked"],
    }).reset_index()
    out = summarize(pairs, ["stress_ratio", "speedup"], KEY + ["n_features", "k"])
    return out.assign(verdict=time_verdict(out.speedup_med)), pairs


def full_targets(hist, factor):
    """``factor`` times the full arm's median final stress, from the histories."""
    full = hist[hist.arm == "full"]
    final = full.assign(final=[s[-1] for s in full.scaled])
    return factor * final.groupby(KEY).final.median()


def time_to_full(hist, factors=TARGET_FACTORS):
    """Time for every fit to reach each factor times full SGDMDS's stress."""
    h = hist[hist.arm.isin(ARMS)]
    out = [add_time_to_target(h, full_targets(hist, f), "scaled").assign(factor=f)
           for f in factors]
    return pd.concat(out, ignore_index=True)


def best_by_ttt(ttt, by=KEY + ["factor", "arm"]):
    """Per group, the config with the lowest median time to target.

    Only configs where more than half the seeds reach the target qualify.
    """
    s = summarize_ttt(ttt, by + ["config"])
    s = s[s.n_reached * 2 > s.n_fits]
    if s.empty:
        return s
    return s.loc[s.groupby(by).ttt_med.idxmin()].reset_index(drop=True)


def time_to_full_table(df, ttt, exp):
    """Fastest config per arm to reach each target, against full's run time.

    SQuaD-MDS's config is chosen on SELECT_SEEDS and reported on REPORT_SEEDS
    in O, so its median can be ``inf`` if the chosen config misses the target
    on the reported seeds.
    """
    sgd = best_by_ttt(ttt[ttt.arm != "squad"])
    sq = ttt[ttt.arm == "squad"]
    if exp == "o":
        chosen = best_by_ttt(sq[sq.seed.isin(SELECT_SEEDS)])
        rep = summarize_ttt(sq[sq.seed.isin(REPORT_SEEDS)],
                            KEY + ["factor", "arm", "config"])
        sq_best = rep.merge(chosen[KEY + ["factor", "config"]],
                            on=KEY + ["factor", "config"])
    else:
        sq_best = best_by_ttt(sq)
    out = pd.concat([sgd, sq_best], ignore_index=True)
    out["full_time"] = lookup(df[df.arm == "full"].groupby(KEY).time_fit.median(),
                              out)
    out["speedup_vs_full"] = out.full_time / out.ttt_med
    return out.sort_values(KEY + ["factor", "arm"]).reset_index(drop=True)


def fastest_arm(ttt_table):
    """Per (dataset, N, target), the arm that reaches it first, and how many do."""
    rows = []
    for keys, g in ttt_table.groupby(KEY + ["factor"]):
        g = g[np.isfinite(g.ttt_med)]
        best = g.loc[g.ttt_med.idxmin()] if len(g) else None
        rows.append(dict(zip(KEY + ["factor"], keys),
                         fastest=None if best is None else best.arm,
                         fastest_config=None if best is None else best.config,
                         arms_reaching=",".join(sorted(g.arm))))
    return pd.DataFrame(rows)


def squad_selection(df, exp):
    """SQuaD-MDS's best setting per (dataset, N), its ladder and its default.

    O: chosen on SELECT_SEEDS by median scaled stress, reported on
    REPORT_SEEDS. P: chosen and reported on all its seeds. ``ladder`` holds the
    best setting at each ``n_iter``; ``gain_last_rung`` is the stress at the
    rung below the top divided by the stress at the top.
    """
    df = add_ratios_to_full(df)
    sq = df[df.arm == "squad"].assign(n_iter=lambda d: d.config.map(squad_n_iter))
    select = SELECT_SEEDS if exp == "o" else sorted(sq.seed.unique())
    report = REPORT_SEEDS if exp == "o" else select
    med = (sq[sq.seed.isin(select)].groupby(KEY + ["n_iter", "config"])
           .stress_scaled.median().rename("select_stress").reset_index())
    rep = summarize(sq[sq.seed.isin(report)],
                    ["stress_ratio", "stress_scaled", "time_fit", "scale_alpha"],
                    KEY + ["config"])

    def pick(group_cols):
        idx = med.groupby(group_cols).select_stress.idxmin()
        return med.loc[idx].merge(rep, on=KEY + ["config"])

    full_med = df[df.arm == "full"].groupby(KEY).stress_scaled.median()
    best = pick(KEY)
    best["at_top"] = best.n_iter == SQUAD_TOP[exp]
    best["select_ratio"] = best.select_stress / lookup(full_med, best)
    ladder = pick(KEY + ["n_iter"])
    rungs = sorted(ladder.n_iter.unique())
    top = ladder[ladder.n_iter == rungs[-1]].set_index(KEY).stress_ratio_med
    below = ladder[ladder.n_iter == rungs[-2]].set_index(KEY).stress_ratio_med
    best["gain_last_rung"] = lookup(below / top, best)
    default = summarize(sq[sq.config == SQUAD_DEFAULT[exp]],
                        ["stress_ratio", "time_fit"], KEY)
    best["default_ratio"] = lookup(default.set_index(KEY).stress_ratio_med, best)
    return best, ladder, default


def squad_vs_sgd(df, exp, k=200):
    """SQuaD-MDS's best setting against budgeted and blocked SGDMDS at ``k``."""
    best, _, _ = squad_selection(df, exp)
    v = vs_full_table(df)
    v = v[v.k == k].pivot_table(index=KEY, columns="arm",
                                values=["stress_ratio_med", "time_fit_med"])
    v.columns = [f"{arm}_{m.replace('_med', '')}" for m, arm in v.columns]
    out = best[KEY + ["config", "n_iter", "stress_ratio_med", "time_fit_med"]]
    out = out.rename(columns={"stress_ratio_med": "squad_stress_ratio",
                              "time_fit_med": "squad_time"})
    out = out.merge(v.reset_index(), on=KEY)
    out["blocked_vs_squad_stress"] = out.blocked_stress_ratio / out.squad_stress_ratio
    out["blocked_vs_squad_time"] = out.blocked_time_fit / out.squad_time
    return out


def tradeoff_points(df, exp):
    """One (time, stress ratio) point per SGD config and SQuaD-MDS n_iter rung."""
    sgd = vs_full_table(df, arms=SGD_ARMS)
    sgd = sgd.assign(rung=sgd.k)
    _, ladder, _ = squad_selection(df, exp)
    sq = ladder.assign(arm="squad", rung=ladder.n_iter)
    cols = KEY + ["arm", "config", "rung", "time_fit_med", "time_fit_q25",
                  "time_fit_q75", "stress_ratio_med", "stress_ratio_q25",
                  "stress_ratio_q75"]
    return pd.concat([sgd[[c for c in cols if c in sgd]],
                      sq[[c for c in cols if c in sq]]], ignore_index=True)


def data_quality(results):
    """Counts, divergences, scale factors, estimate error and canary spread."""
    rows = []
    for exp, df in results.items():
        sgd = df[df.arm.isin(SGD_ARMS + ("smacof-random",))]
        sq = df[df.arm == "squad"]
        rows.append(dict(
            exp=exp, fits=len(df), datasets=df.dataset.nunique(),
            rungs=df.groupby(KEY).ngroups, diverged=int(df.diverged.sum()),
            alpha_dev_sgd_smacof=float((sgd.scale_alpha - 1).abs().max()),
            alpha_squad_min=sq.scale_alpha.min() if len(sq) else np.nan,
            alpha_squad_max=sq.scale_alpha.max() if len(sq) else np.nan,
            est_err_med=df.est_rel_err.median(), est_err_max=df.est_rel_err.max(),
            clip_frac_max=sq.clip_frac.max() if len(sq) else np.nan,
            canary_d8_min=df.canary_d8.min(), canary_d8_max=df.canary_d8.max(),
            canary_d784_min=df.canary_d784.min(),
            canary_d784_max=df.canary_d784.max()))
    return pd.DataFrame(rows)


def run(root=ROOT, out=None):
    """Compute every table, write the CSVs and ``tables.md``; return the tables."""
    out = Path(out) if out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    res = {e: load_results(e, root) for e in FOLDERS}
    hist = {e: load_histories(e, root) for e in FOLDERS}
    T = {"quality": data_quality(res), "step1": step1_table(res["o"])}
    T["step1_tests"] = pd.DataFrame([step1_tests(T["step1"])])
    T["step1_ttt"], T["step1_converse"], _ = step1_time_to_smacof(res["o"],
                                                                  hist["o"])
    T["step1_q_ttt"], T["step1_q_converse"], _ = step1_time_to_smacof(res["q"],
                                                                      hist["q"])
    T["classical"] = classical_smacof_table(res["o"])
    T["q"] = q_table(res["q"])
    T["q_slopes"] = q_slopes(T["q"])
    T["q_break_even"] = q_break_even(T["q"])
    T["cost_fits"], T["cost"] = cost_per_pair(res, hist)
    for e in ("o", "p", "q"):
        T[f"vs_full_{e}"] = vs_full_table(res[e])
    for e in ("o", "p"):
        T[f"blocked_{e}"], _ = blocked_vs_budget(res[e])
        ttt = time_to_full(hist[e])
        T[f"ttt_fits_{e}"] = ttt.drop(columns=["time", "raw", "scaled"])
        T[f"ttt_{e}"] = time_to_full_table(res[e], ttt, e)
        T[f"fastest_{e}"] = fastest_arm(T[f"ttt_{e}"])
        T[f"squad_best_{e}"], T[f"squad_ladder_{e}"], T[f"squad_default_{e}"] = (
            squad_selection(res[e], e))
        T[f"squad_vs_sgd_{e}"] = squad_vs_sgd(res[e], e)
        T[f"tradeoff_{e}"] = tradeoff_points(res[e], e)
    for name, table in T.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "tables.md").write_text(render(T))
    return T


def fmt_time(t):
    if not np.isfinite(t):
        return "—"
    if t < 1:
        return f"{t:.2g} s"
    return f"{t:.3g} s" if t < 100 else f"{t:,.0f} s"


def fmt_x(x, digits=2):
    if not np.isfinite(x):
        return "—"
    return f"{x:,.0f}×" if x >= 100 else f"{x:.{digits}f}×"


def fmt_speed(x):
    """A speedup with its 15% verdict: '≈' comparable, '(slower)' slower."""
    if not np.isfinite(x):
        return "—"
    v = time_verdict([x])[0]
    return {"faster": fmt_x(x), "comparable": f"≈{fmt_x(x)}",
            "slower": f"{fmt_x(x)} (slower)"}[v]


def fmt_iqr(med, q25, q75, digits=3):
    return f"{med:.{digits}f} [{q25:.{digits}f}–{q75:.{digits}f}]"


def fmt_time_iqr(med, q25, q75):
    return f"{fmt_time(med)} [{fmt_time(q25).rstrip(' s')}–{fmt_time(q75)}]"


def md(df):
    """A DataFrame as a GitHub markdown table; text columns left-aligned."""
    cols = list(df.columns)
    align = ["---" if c in ("dataset", "setting", "fastest") else "--:"
             for c in cols]
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


def short_squad(config):
    """``squad-i8000-lr150-exa0-pca-cap1`` -> ``i8000 lr150 pca cap1``."""
    parts = config.split("-")[1:]
    parts = [p for p in parts if p != "exa0"]
    return " ".join("exa" if p == "exa1" else p for p in parts)


TITLES = {
    "T1": "Step 1, O: SMACOF (random init) vs full SGDMDS",
    "T2": "Step 1, O: time to reach SMACOF's final stress",
    "T3": "Step 1, Q: SMACOF ÷ full time · SMACOF iterations · SMACOF ÷ full stress",
    "T3s": "Step 1, Q: log-log slope of median time against N (N ≥ 1,000)",
    "T4": "Cost per pair: full sweep (per epoch) and SMACOF (per iteration), by N",
    "T5": "Step 2, O: budgeted ÷ full (stress overhead · speedup)",
    "T5s": "Step 2, O: budgeted and blocked ÷ full, summary per k",
    "T6": "Step 2, P: budgeted ÷ full (stress overhead · speedup)",
    "T6s": "Step 2, P: budgeted and blocked ÷ full, summary per k",
    "T7": "Step 2, Q: speedup over full SGDMDS at k=200",
    "T8": "Cost per pair: budgeted (m=2) and blocked (m=16), by D",
    "T9": "Step 3: blocked vs budgeted at k=200, paired by seed",
    "T10": "Step 3, O: SQuaD-MDS's best setting",
    "T11": "Step 3, P: SQuaD-MDS's best setting (N=10k and each source's top N)",
    "T12": "Step 3, O: time to reach 1.01× and 1.05× full stress, as speedup over "
           "the full run",
    "T13": "Step 3, P: time to reach 1.01× and 1.05× full stress (N=10k and each "
           "source's top N)",
    "T14": "Step 3: which arm reaches the target first",
    "T15": "Step 3: SQuaD-MDS's best vs blocked SGDMDS at k=200",
    "T16": "Classical-init SMACOF, O (for the later study)",
}
T8_ROWS = {"seismic", "spambase", "hatespeech", "epileptic", "coil20",
           "fashion_mnist", "cifar10", "fmd", "hiva", "swiss_roll", "torus"}


def ladder_ends(df):
    """Rows at N=10k and at each dataset's largest N (P's condensed tables)."""
    top = df.groupby("dataset").n_samples.transform("max")
    return df[(df.n_samples == 10_000) | (df.n_samples == top)]


def per_k_summary(v):
    """Median and range of overhead and speedup per (arm, k), with verdicts."""
    rows = []
    for (arm, k), g in v.groupby(["arm", "k"]):
        c = g.verdict.value_counts()
        rows.append({
            "arm": arm, "k": int(k), "rungs": len(g),
            "overhead: median (range)": f"{g.stress_ratio_med.median():.3f} "
            f"({g.stress_ratio_med.min():.3f}–{g.stress_ratio_med.max():.3f})",
            "speedup: median (range)": f"{fmt_x(g.speedup_med.median())} "
            f"({fmt_x(g.speedup_med.min())}–{fmt_x(g.speedup_med.max())})",
            "faster / ≈ / slower": f"{c.get('faster', 0)} / "
            f"{c.get('comparable', 0)} / {c.get('slower', 0)}"})
    return pd.DataFrame(rows)


def render_tables(T):
    """The doc's tables as ``{name: markdown}``, keyed as in ``TITLES``."""
    R = {}
    s1 = T["step1"]
    R["T1"] = md(pd.DataFrame({
        "dataset": s1.dataset, "N": s1.N, "D": s1.D,
        "SMACOF iters (at cap)": [f"{int(i)} ({c}/10)" for i, c in
                                  zip(s1.smacof_iter_med, s1.smacof_at_cap)],
        "SMACOF stress IQR": (100 * s1.smacof_iqr_rel).map("{:.1f}%".format),
        "full ÷ SMACOF stress": [fmt_iqr(m, a, b) for m, a, b in zip(
            s1.stress_ratio, s1.stress_ratio_q25, s1.stress_ratio_q75)],
        "SMACOF time": [fmt_time_iqr(m, a, b) for m, a, b in zip(
            s1.smacof_time_fit_med, s1.smacof_time_fit_q25,
            s1.smacof_time_fit_q75)],
        "full time": [fmt_time_iqr(m, a, b) for m, a, b in zip(
            s1.full_time_fit_med, s1.full_time_fit_q25, s1.full_time_fit_q75)],
        "SMACOF ÷ full time": s1.speedup.map(fmt_speed),
    }))

    tt = T["step1_ttt"].set_index(["dataset", "config"])
    rows = []
    for r in s1.itertuples():
        f = tt.loc[(r.dataset, "full")]
        row = {"dataset": r.dataset, "N": r.N, "SMACOF time": fmt_time(f.smacof_time),
               "full: time to SMACOF's stress": fmt_time_iqr(f.ttt_med, f.ttt_q25,
                                                             f.ttt_q75),
               "seeds reaching": f"{int(f.n_reached)}/10",
               "SMACOF time ÷ that": fmt_x(f.speedup_vs_smacof)}
        for cfg, tag in (("budget-k200", "same for budget k200"),
                         ("blocked-k200-m16", "same for blocked k200")):
            b = tt.loc[(r.dataset, cfg)]
            row[tag] = f"{fmt_x(b.speedup_vs_smacof)} ({int(b.n_reached)}/10)"
        rows.append(row)
    R["T2"] = md(pd.DataFrame(rows))

    q = T["q"].set_index(["dataset", "n_samples", "config"])
    rows = []
    for n in sorted(T["q"].n_samples.unique()):
        row = {"N": n}
        for name, d in (("s_curve", 100), ("swiss_roll", 300), ("torus", 500)):
            sm, fu = q.loc[(name, n, "smacof-random")], q.loc[(name, n, "full")]
            row[f"{name} (D={d})"] = (
                f"{fmt_speed(sm.time_fit_med / fu.time_fit_med)} · "
                f"{sm.n_iter_run_med:.0f} · {sm.stress_ratio_med:.3f}")
        rows.append(row)
    R["T3"] = md(pd.DataFrame(rows))
    sl = T["q_slopes"].pivot(index="dataset", columns="config", values="slope")
    R["T3s"] = md(sl.map("{:.2f}".format).reset_index()[
        ["dataset", "smacof-random", "full", "budget-k200", "blocked-k200-m16"]])

    c = T["cost"]
    bins = [0, 1100, 2100, 2800, 3300, 5000, 6000, 10_000, 20_000, 35_000, 50_000]
    labels = ["≤1,100", "1,400–2,100", "2,500–2,750", "3,000–3,250",
              "3,500–5,000", "5,750", "10k", "20k", "35k", "50k"]
    c = c.assign(band=pd.cut(c.n_samples, bins=bins, labels=labels))
    rows = []
    for band, g in c[c.arm == "full"].groupby("band", observed=True):
        s = c[(c.arm == "smacof-random") & (c.band == band)]
        rows.append({"N": band, "fits": int(g.n_fits.sum()),
                     "full sweep, ns per pair per epoch":
                     f"{g.ns_per_pair_med.min():.0f}–{g.ns_per_pair_med.max():.0f}",
                     "SMACOF, ns per pair per iteration": "—" if s.empty else
                     f"{s.ns_per_pair_med.min():.0f}–{s.ns_per_pair_med.max():.0f}"})
    R["T4"] = md(pd.DataFrame(rows))

    sgd = c[c.arm.isin(["budget", "blocked"])]
    top = sgd.loc[sgd.groupby(["exp", "dataset", "arm"]).n_samples.idxmax()]
    w = top.pivot_table(index=["exp", "dataset", "n_samples", "n_features"],
                        columns="arm", values="ns_per_pair_med").reset_index()
    w = w[(w.exp == "p") | w.dataset.isin(T8_ROWS)]
    w = w.sort_values(["n_features", "n_samples"])
    R["T8"] = md(pd.DataFrame({
        "dataset": w.dataset + " (" + w.exp.str.upper() + ")",
        "N": w.n_samples, "D": w.n_features,
        "budget ns/pair": w.budget.map("{:.0f}".format),
        "blocked ns/pair": w.blocked.map("{:.0f}".format),
        "budget ÷ blocked": (w.budget / w.blocked).map("{:.2f}".format)}))

    for e, name in (("o", "T5"), ("p", "T6")):
        v = T[f"vs_full_{e}"]
        R[f"{name}s"] = md(per_k_summary(v))
        rows = []
        for (ds, n), g in v[v.arm == "budget"].groupby(KEY):
            row = {"dataset": ds, "N": n, "D": g.n_features.iloc[0]}
            for r in g.sort_values("k").itertuples():
                row[f"k={r.k:g}"] = (f"{r.stress_ratio_med:.3f} · "
                                     f"{fmt_speed(r.speedup_med)}")
            rows.append(row)
        order = ["N", "dataset"] if e == "o" else ["dataset", "N"]
        R[name] = md(pd.DataFrame(rows).sort_values(order).fillna("—"))

    q = T["q"].set_index(["dataset", "n_samples", "config"])
    rows = []
    for n in sorted(T["q"].n_samples.unique()):
        row = {"N": n}
        for ds, d in (("s_curve", 100), ("swiss_roll", 300), ("torus", 500)):
            for cfg, tag in (("budget-k200", "budget"),
                             ("blocked-k200-m16", "blocked")):
                row[f"D={d} {tag}"] = fmt_speed(q.loc[(ds, n, cfg)].speedup_med)
        rows.append(row)
    R["T7"] = md(pd.DataFrame(rows))

    rows = []
    for e in ("o", "p"):
        b = T[f"blocked_{e}"]
        b = b[b.k == 200]
        b = ladder_ends(b) if e == "p" else b
        for r in b.sort_values(["n_features", "n_samples"]).itertuples():
            rows.append({"dataset": f"{r.dataset} ({e.upper()})", "N": r.n_samples,
                         "D": r.n_features,
                         "blocked ÷ budget stress": fmt_iqr(
                             r.stress_ratio_med, r.stress_ratio_q25,
                             r.stress_ratio_q75),
                         "budget ÷ blocked time": fmt_speed(r.speedup_med)})
    R["T9"] = md(pd.DataFrame(rows))

    for e, name in (("o", "T10"), ("p", "T11")):
        b = T[f"squad_best_{e}"].sort_values(KEY)
        b = ladder_ends(b) if e == "p" else b
        t = pd.DataFrame({
            "dataset": b.dataset, "N": b.n_samples,
            "setting": b.config.map(short_squad),
            "stress ÷ full": [fmt_iqr(m, a, z) for m, a, z in zip(
                b.stress_ratio_med, b.stress_ratio_q25, b.stress_ratio_q75)],
            "on selection seeds": b.select_ratio.map("{:.3f}".format),
            "last rung's gain": b.gain_last_rung.map("{:.3f}×".format),
            "time": b.time_fit_med.map(fmt_time),
            "authors' default ÷ full": b.default_ratio.map("{:.3f}".format)})
        R[name] = md(t.drop(columns="N" if e == "o" else "on selection seeds"))

    for e, name in (("o", "T12"), ("p", "T13")):
        tt = T[f"ttt_{e}"]
        tt = tt[tt.factor.isin([1.01, 1.05])]
        tt = ladder_ends(tt) if e == "p" else tt
        rows = []
        for (ds, n), g in tt.groupby(KEY):
            full = g[(g.arm == "full") & (g.factor == 1.01)]
            row = {"dataset": ds, "N": n, "full run": fmt_time(g.full_time.iloc[0]),
                   "full reaches 1.01×": "—" if full.empty else fmt_time(
                       full.ttt_med.iloc[0])}
            for f in (1.01, 1.05):
                for arm in ("budget", "blocked", "squad"):
                    r = g[(g.arm == arm) & (g.factor == f)]
                    if r.empty or not np.isfinite(r.ttt_med.iloc[0]):
                        row[f"{f:.2f}× {arm}"] = "—"
                        continue
                    r = r.iloc[0]
                    tag = (f"i{squad_n_iter(r.config)}" if arm == "squad"
                           else f"k{r.config.split('-')[1][1:]}")
                    row[f"{f:.2f}× {arm}"] = f"{fmt_x(r.speedup_vs_full)} ({tag})"
            rows.append(row)
        R[name] = md(pd.DataFrame(rows))

    rows = []
    for e in ("o", "p"):
        for factor, g in T[f"fastest_{e}"].groupby("factor"):
            counts = g.fastest.fillna("none").value_counts()
            reached = g.arms_reaching.fillna("").str.contains("squad").sum()
            rows.append({"suite": e.upper(), "target": f"{factor:.2f}×",
                         "rungs": len(g),
                         **{arm: int(counts.get(arm, 0)) for arm in ARMS},
                         "SQuaD-MDS reaches it": int(reached)})
    R["T14"] = md(pd.DataFrame(rows))

    rows = []
    for e in ("o", "p"):
        s = T[f"squad_vs_sgd_{e}"]
        s = ladder_ends(s) if e == "p" else s
        for r in s.sort_values(KEY).itertuples():
            rows.append({"dataset": f"{r.dataset} ({e.upper()})", "N": r.n_samples,
                         "SQuaD ÷ full": f"{r.squad_stress_ratio:.3f}",
                         "SQuaD time": fmt_time(r.squad_time),
                         "blocked k200 ÷ full": f"{r.blocked_stress_ratio:.3f}",
                         "blocked time": fmt_time(r.blocked_time_fit),
                         "blocked ÷ SQuaD stress": f"{r.blocked_vs_squad_stress:.3f}",
                         "blocked ÷ SQuaD time": fmt_x(r.blocked_vs_squad_time)})
    R["T15"] = md(pd.DataFrame(rows))

    cl = T["classical"]
    R["T16"] = md(pd.DataFrame({
        "dataset": cl.dataset, "N": cl.N, "D": cl.D,
        "iters (at cap)": [f"{int(i)} ({c}/10)" for i, c in
                           zip(cl.classical_iter, cl.classical_at_cap)],
        "classical ÷ full stress": cl.vs_full.map("{:.3f}".format),
        "full's IQR ÷ classical": [f"{a:.3f}–{b:.3f}" for a, b in
                                   zip(cl.full_q25, cl.full_q75)],
        "classical ÷ random-init SMACOF": cl.vs_smacof_random.map("{:.3f}".format),
        "classical time": cl.classical_time.map(fmt_time),
        "full ÷ classical speed": cl.speedup_full.map(fmt_speed)}))
    return R


def render(T):
    """All of the doc's tables as one markdown file, in the doc's order."""
    R = render_tables(T)
    return "".join(f"\n## {k}. {TITLES[k]}\n\n{R[k]}\n" for k in TITLES)


def main():
    args = sys.argv[1:]
    out = args[args.index("--out") + 1] if "--out" in args else None
    T = run(out=out)
    print(f"Wrote {len(T)} tables to "
          f"{(Path(out) if out else ROOT / 'analysis').resolve()}")
    print(T["quality"].to_string())


if __name__ == "__main__":
    main()
