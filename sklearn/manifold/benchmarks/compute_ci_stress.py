import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel  # pip already present in most envs; if not, I can give a pure-numpy version.

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_a" / "sgd_30_epochs"
N_INIT_EXPECTED = 20
ALGO_SMACOF = "SMACOF"
ALGO_SGD = "SGD-MDS"

def mean_ci95(x: np.ndarray):
    """t-based 95% CI for the mean."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return m, s, (m, m)
    se = s / math.sqrt(n)
    tcrit = float(t.ppf(0.975, df=n - 1))
    lo = m - tcrit * se
    hi = m + tcrit * se
    return m, s, (lo, hi)

def load_dataset_results(ds_dir: Path):
    res_path = ds_dir / "results.json"
    if not res_path.exists():
        return None
    with open(res_path, "r") as f:
        data = json.load(f)

    runs = data["runs"]
    # Build seed->stress dict per algo (so we can pair properly)
    sm = {}
    sg = {}

    sm_time = {}
    sg_time = {}

    for r in runs:
        algo = r["algo"]
        seed = int(r["seed"])
        stress = float(r["stress"])
        time = float(r.get("time", np.nan))
        if algo == ALGO_SMACOF:
            sm[seed] = stress
            sm_time[seed] = time
        elif algo == ALGO_SGD:
            sg[seed] = stress
            sg_time[seed] = time

    # Intersect seeds to ensure pairing
    common_seeds = sorted(set(sm.keys()) & set(sg.keys()))
    if len(common_seeds) == 0:
        return None

    sm_stress = np.array([sm[s] for s in common_seeds], dtype=float)
    sg_stress = np.array([sg[s] for s in common_seeds], dtype=float)

    sm_t = np.array([sm_time[s] for s in common_seeds], dtype=float)
    sg_t = np.array([sg_time[s] for s in common_seeds], dtype=float)

    return common_seeds, sm_stress, sg_stress, sm_t, sg_t

def main():
    rows = []

    for ds_dir in sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()]):
        out = load_dataset_results(ds_dir)
        if out is None:
            continue
        seeds, sm_stress, sg_stress, sm_t, sg_t = out
        n = len(seeds)

        sm_mean, sm_std, (sm_lo, sm_hi) = mean_ci95(sm_stress)
        sg_mean, sg_std, (sg_lo, sg_hi) = mean_ci95(sg_stress)

        # “Mean outside other’s CI?” checks (heuristic)
        sg_mean_outside_sm_ci = not (sm_lo <= sg_mean <= sm_hi)
        sm_mean_outside_sg_ci = not (sg_lo <= sm_mean <= sg_hi)

        # Paired test on differences
        diffs = sg_stress - sm_stress
        d_mean, d_std, (d_lo, d_hi) = mean_ci95(diffs)
        tstat, pval = ttest_rel(sg_stress, sm_stress)  # H0: mean(diff)=0

        # Speedup stats (paired)
        # If you want "solver time" only, ensure your logged "time" matches that.
        speedups = sm_t / sg_t
        sp_mean = float(np.nanmean(speedups))
        sp_med = float(np.nanmedian(speedups))

        # Stress ratio stats
        ratios = sg_stress / sm_stress
        ratio_mean = float(np.mean(ratios))
        ratio_med = float(np.median(ratios))

        rows.append({
            "dataset": ds_dir.name,
            "n": n,
            "sm_mean": sm_mean,
            "sm_ci_lo": sm_lo,
            "sm_ci_hi": sm_hi,
            "sg_mean": sg_mean,
            "sg_ci_lo": sg_lo,
            "sg_ci_hi": sg_hi,
            "sg_mean_outside_sm_ci": sg_mean_outside_sm_ci,
            "sm_mean_outside_sg_ci": sm_mean_outside_sg_ci,
            "diff_mean(sg-sm)": d_mean,
            "diff_ci_lo": d_lo,
            "diff_ci_hi": d_hi,
            "paired_t_pval": float(pval),
            "speedup_mean": sp_mean,
            "speedup_median": sp_med,
            "stress_ratio_mean": ratio_mean,
            "stress_ratio_median": ratio_med,
        })

    df = pd.DataFrame(rows).sort_values("paired_t_pval")
    print("\n=== Summary (sorted by paired t-test p-value) ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[[
            "dataset", "n",
            "sm_mean", "sm_ci_lo", "sm_ci_hi",
            "sg_mean", "sg_mean_outside_sm_ci",
            "diff_mean(sg-sm)", "diff_ci_lo", "diff_ci_hi",
            "paired_t_pval",
            "speedup_median", "stress_ratio_median"
        ]])

    # “Good example of the trend”: large speedup + SGD not worse stress
    candidates = df[(df["stress_ratio_median"] <= 1.0)].copy()
    candidates = candidates.sort_values(["speedup_median"], ascending=False)
    print("\n=== Strong ‘SGD faster and not worse stress’ examples (top 5) ===")
    print(candidates[["dataset", "speedup_median", "stress_ratio_median", "paired_t_pval"]].head(5))

    # “Outliers”: extreme speedups or extreme stress ratios
    # (you can redefine outliers however you like)
    sp_q1, sp_q3 = df["speedup_median"].quantile([0.25, 0.75])
    sp_iqr = sp_q3 - sp_q1
    sp_hi = sp_q3 + 1.5 * sp_iqr

    r_q1, r_q3 = df["stress_ratio_median"].quantile([0.25, 0.75])
    r_iqr = r_q3 - r_q1
    r_hi = r_q3 + 1.5 * r_iqr
    r_lo = r_q1 - 1.5 * r_iqr

    outliers = df[(df["speedup_median"] > sp_hi) | (df["stress_ratio_median"] > r_hi) | (df["stress_ratio_median"] < r_lo)]
    print("\n=== Outlier datasets (IQR rule on speedup_median or stress_ratio_median) ===")
    print(outliers[["dataset", "speedup_median", "stress_ratio_median", "paired_t_pval"]])

if __name__ == "__main__":
    main()
