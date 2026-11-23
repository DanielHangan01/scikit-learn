"""Dataset suites and SQuaD-MDS grids of the paper's runs (O, P and Q).

These were first defined by earlier experiments, whose runners now live on the
``research/history`` branch: the small suite by J, the large suite, its ladder
and subsample seed by K-optimized, and the SQuaD-MDS grids by M. The values are
unchanged, so O and P reproduce those experiments' inputs.
"""

from typing import List

# Espadoto et al.'s collection minus orl, whose 400 rows hold one value 360 times.
SMALL_SUITE = [
    "bank", "cifar10", "cnae9", "coil20", "epileptic", "fashion_mnist",
    "fmd", "har", "hatespeech", "hiva", "imdb", "secom", "seismic",
    "sentiment", "sms", "spambase", "svhn",
]

LARGE_SUITE = [
    "fashion_mnist_full",  # D=784   strain 0.37  N_full= 70,000
    "feynman_iii_15_12",   # D=3     strain 0.69  N_full=100,000
    "california_housing",  # D=8     strain 0.65  N_full= 20,640
    "patchcamelyon",       # D=512   strain 0.31  N_full=100,000
    "cifar10_raw",         # D=3072  strain 0.38  N_full= 50,000
    "ag_news",             # D=384   strain 0.08  N_full=100,000
    "tiny_imagenet",       # D=512   strain 0.07  N_full=100,000
    "mnist_digits",        # D=784   strain 0.12  N_full= 70,000
]
N_LADDER = [10000, 20000, 35000, 50000, 70000, 100000]
SUBSAMPLE_SEED = 42  # one fixed subsample per (dataset, N)

# SQuaD-MDS settings: the full cross on the small suite, and the settings that
# matter on the large one (only patchcamelyon has many duplicates, so one step
# cap suffices there).
SQUAD_GRID_SMALL = {
    "n_iter": [100, 250, 500, 1000, 2000, 4000, 8000],
    "lr": [150.0, 550.0, 1500.0],
    "exaggerate_d": [False, True],
    "init": ["pca", "random"],
    "max_step_frac": [None, 4.0, 1.0],
}
SQUAD_GRID_LARGE = {
    "n_iter": [100, 250, 500, 1000, 2000, 4000],
    "lr": [550.0],
    "exaggerate_d": [False, True],
    "init": ["pca"],
    "max_step_frac": [4.0],
}


def make_ladder(n_full: int) -> List[int]:
    """Ladder rungs up to ``n_full``, ending at ``n_full`` itself.

    The full N replaces the top rung when within 5% of it (california_housing's
    20,640 replaces 20,000) and is appended otherwise.
    """
    rungs = [n for n in N_LADDER if n <= n_full]
    if not rungs:
        return [n_full]
    if n_full <= N_LADDER[-1] and n_full not in rungs:
        if n_full < 1.05 * rungs[-1]:
            rungs[-1] = n_full
        else:
            rungs.append(n_full)
    return sorted(set(rungs))
