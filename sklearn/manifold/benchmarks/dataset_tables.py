"""Per-dataset tables for Experiments O and P, one row per algorithm setting.

``dataset_table`` covers one (experiment, dataset, N). Every ratio is against
full SGDMDS's median on the same dataset and N, as in
``analyze_experiment_o_p_q``:

- stress: median scale-optimal stress, its ratio to full [IQR over seeds], and
  Kruskal's stress-1, sqrt(stress / sum of squared input distances);
- time: median wall time [IQR] and the speedup over full, under the 15% rule;
- time to reach 1.01x and 1.05x full's final stress, from the histories;
- kNN@10: the share of each point's 10 nearest input neighbours kept in the
  embedding, for seed 42 only (the only saved embeddings).

SQuaD-MDS in O lists the best setting per ``n_iter`` (chosen on seeds 42-46,
reported on 47-51) and the authors' default; in P it lists all 12 settings.

``cheapest_k_table`` lists, per O dataset, the fastest k whose median stress
stays within each tolerance of full's.

Usage:
    python dataset_tables.py o fashion_mnist
    python dataset_tables.py p fashion_mnist_full 50000
    python dataset_tables.py k
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import analyze_experiment_o_p_q as A
from bench_utils import load_local_dataset

FACTORS = (1.01, 1.05)
TOLERANCES = (1.01, 1.02, 1.05)
KNN = 10
ARM_NAMES = {"smacof": "SMACOF", "full": "SGDMDS full", "budget": "SGDMDS budgeted",
             "blocked": "SGDMDS blocked", "squad": "SQuaD-MDS"}


def load_input(dataset, n, subsample_seed):
    """The standardised X the fits saw, with P's fixed subsample if any."""
    X = np.ascontiguousarray(load_local_dataset(dataset)[0], dtype=np.float64)
    if pd.notna(subsample_seed) and n < X.shape[0]:
        idx = np.random.RandomState(int(subsample_seed)).choice(
            X.shape[0], size=n, replace=False)
        X = np.ascontiguousarray(X[idx])
    return X


def sum_sq_distances(X):
    """Sum over i<j of squared input distances: N times the total variance."""
    return X.shape[0] * float(((X - X.mean(axis=0)) ** 2).sum())


def knn_indices(Z, k=KNN):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    return nn.kneighbors(Z, return_distance=False)[:, 1:]


def knn_recall(nx, Y, k=KNN):
    """Mean share of the input neighbours ``nx`` that stay neighbours in Y."""
    ny = knn_indices(Y, k)
    return float((nx[:, :, None] == ny[:, None, :]).any(axis=2).mean())


def time_to_targets(hist, factors=FACTORS):
    """Per fit, the time to reach each factor times full's median final stress.

    SMACOF records raw stress only, which is within 1e-8 of its scaled stress.
    """
    full = hist[hist.arm == "full"]
    final = np.median([s[-1] for s in full.scaled])
    rows = []
    for h in hist.itertuples():
        stress = h.raw if h.scaled is None else h.scaled
        row = dict(config=h.config, seed=h.seed)
        for f in factors:
            row[f] = A.time_to_target(h.time, stress, f * final)
        rows.append(row)
    return pd.DataFrame(rows)


def squad_rows(fits, exp):
    """SQuaD-MDS configs to list, the seeds to report them on, and the best."""
    best, ladder, _ = A.squad_selection(fits, exp)
    if exp == "o":
        configs = list(ladder.sort_values("n_iter").config)
        seeds = A.REPORT_SEEDS
    else:
        sq = fits[fits.arm == "squad"]
        configs = sorted(sq.config.unique(),
                         key=lambda c: (A.squad_n_iter(c), c))
        seeds = sorted(sq.seed.unique())
    if A.SQUAD_DEFAULT[exp] not in configs:
        configs.append(A.SQUAD_DEFAULT[exp])
    return configs, seeds, best.config.iloc[0]


def kind(config):
    """``smacof``, ``full``, ``budget``, ``blocked`` or ``squad``."""
    return "smacof" if config.startswith("smacof-") else A.arm_of(config)


def order_key(config):
    arm = kind(config)
    rank = ["smacof", "full", "budget", "blocked", "squad"].index(arm)
    if arm in ("budget", "blocked"):
        return rank, int(config.split("-")[1][1:])
    if arm == "squad":
        return rank, A.squad_n_iter(config)
    return rank, config


def setting(config, arm, best, exp):
    if arm == "smacof":
        return config.split("-", 1)[1].replace("_mds", "") + " init"
    if arm == "full":
        return "all pairs"
    if arm in ("budget", "blocked"):
        return "k=" + config.split("-")[1][1:]
    label = A.short_squad(config)
    if config == best:
        label = "★ " + label
    if config == A.SQUAD_DEFAULT[exp]:
        label += " (authors' default)"
    return label


def iterations(g, arm):
    if arm == "smacof":
        capped = int((g.n_iter_run >= A.SMACOF_CAP).sum())
        return f"{g.n_iter_run.median():.0f} ({capped}/{len(g)} at cap)"
    if arm == "squad":
        return f"{int(g.n_iter.iloc[0])}"
    return f"{g.n_iter_run.median():.0f} epochs"


def fmt_ttt(values):
    reached = int(np.isfinite(values).sum())
    med = A.quantile(values, 0.5) if reached * 2 > len(values) else np.inf
    return f"{A.fmt_time(med)} ({reached}/{len(values)})"


def dataset_table(exp, dataset, n=None, with_input=True):
    """One row per algorithm setting on one (dataset, N), as a DataFrame."""
    fits = A.load_results(exp)
    fits = fits[fits.dataset == dataset]
    n = int(n or fits.n_samples.max())
    fits = A.add_ratios_to_full(fits[fits.n_samples == n])
    hist = A.load_histories(exp)
    hist = hist[(hist.dataset == dataset) & (hist.n_samples == n)]
    ttt = time_to_targets(hist)

    sq_configs, sq_seeds, best = squad_rows(fits, exp)
    keep = (fits.arm != "squad") | (fits.config.isin(sq_configs)
                                    & fits.seed.isin(sq_seeds))
    fits = fits[keep]
    ttt = ttt.merge(fits[["config", "seed"]], on=["config", "seed"])

    if with_input:
        X = load_input(dataset, n, fits.subsample_seed.iloc[0])
        total = sum_sq_distances(X)
        nx = knn_indices(X)
        emb_dir = A.ROOT / A.FOLDERS[exp] / dataset / "embeddings"

    rows = []
    for config in sorted(fits.config.unique(), key=order_key):
        g = fits[fits.config == config]
        arm = kind(config)
        t = ttt[ttt.config == config]
        k = g.k.iloc[0]
        share = 1.0 if arm == "full" else (
            2 * k / (n - 1) if arm in ("budget", "blocked") else np.nan)
        row = {
            "algorithm": ARM_NAMES[arm],
            "setting": setting(config, arm, best, exp),
            "fits": len(g),
            "pairs per epoch": "—" if np.isnan(share) else f"{100 * share:.3g}%",
            "stress": f"{g.stress_scaled.median():.4g}",
            "÷ full [IQR]": A.fmt_iqr(g.stress_ratio.median(),
                                      g.stress_ratio.quantile(0.25),
                                      g.stress_ratio.quantile(0.75)),
            "α": f"{g.scale_alpha.median():.3f}",
            "time [IQR]": A.fmt_time_iqr(g.time_fit.median(),
                                         g.time_fit.quantile(0.25),
                                         g.time_fit.quantile(0.75)),
            "speedup": A.fmt_speed(g.full_time.iloc[0] / g.time_fit.median()),
        }
        for f in FACTORS:
            row[f"to {f:.2f}× full"] = fmt_ttt(t[f].to_numpy())
        row["iterations"] = iterations(g, arm)
        if with_input:
            row["stress-1"] = f"{np.sqrt(g.stress_scaled.median() / total):.4f}"
            path = emb_dir / f"N{n}-{config}-s42.npy"
            row[f"kNN@{KNN}"] = (f"{knn_recall(nx, np.load(path)):.3f}"
                                 if path.exists() else "—")
        rows.append(row)
    out = pd.DataFrame(rows)
    if with_input:
        cols = list(out.columns)
        cols.insert(cols.index("α"), cols.pop(cols.index("stress-1")))
        out = out[cols]
    return out


def cheapest_k_table(exp="o"):
    """Per dataset, the fastest k within each stress tolerance of full."""
    v = A.vs_full_table(A.add_ratios_to_full(A.load_results(exp)))
    rows = []
    for (dataset, n), g in v.groupby(A.KEY):
        row = {"dataset": dataset, "N": n, "D": int(g.n_features.iloc[0]),
               "k=50 pairs per epoch": f"{100 * 100 / (n - 1):.3g}%",
               "k=200 pairs per epoch": f"{100 * 400 / (n - 1):.3g}%"}
        for arm in ("budget", "blocked"):
            s = g[(g.arm == arm) & (g.k == 200)].speedup_med.iloc[0]
            row[f"{arm} break-even k"] = f"≈{200 * s:,.0f}"
        for tol in TOLERANCES:
            for arm in ("budget", "blocked"):
                ok = g[(g.arm == arm) & (g.stress_ratio_med <= tol)]
                if ok.empty:
                    cell = "none"
                else:
                    b = ok.loc[ok.speedup_med.idxmax()]
                    cell = f"k={b.k:.0f}: {A.fmt_speed(b.speedup_med)}"
                row[f"≤{tol:.2f}× {arm}"] = cell
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N")


def md(df, left=("algorithm", "setting", "dataset")):
    """A DataFrame as a markdown table, with the text columns left-aligned."""
    cols = list(df.columns)
    align = ["---" if c in left else "--:" for c in cols]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("exp", choices=["o", "p", "k"])
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("n", nargs="?", type=int)
    parser.add_argument("--no-input", action="store_true",
                        help="skip stress-1 and kNN, which need the dataset")
    args = parser.parse_args()
    if args.exp == "k":
        table = cheapest_k_table()
    elif args.dataset is None:
        parser.error("a dataset is needed for o and p")
    else:
        table = dataset_table(args.exp, args.dataset, args.n,
                              with_input=not args.no_input)
    print(md(table))


if __name__ == "__main__":
    sys.exit(main())
