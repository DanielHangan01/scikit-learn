"""
Benchmarking Utilities for SGD-MDS vs SMACOF.
Handles local dataset loading and execution wrappers.
"""
import sys
import os
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent  # sklearn/manifold/
sys.path.insert(0, str(parent_dir.parent.parent))

from sklearn.manifold import MDS as SMACOF_MDS
try:
    from sklearn.manifold._sgd_mds import (
        SGDMDS,
        _resolve_n_pivots,
        _select_pivots_maxmin,
        _maybe_pca_project,
    )
    from sklearn.manifold._pivot_mds import pivot_mds as _pivot_mds_fn
except ImportError:
    print("Could not import SGDMDS from sklearn.manifold._sgd_mds.")
    print("Make sure you are running this from the benchmarks directory.")
    sys.exit(1)

from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, check_random_state

# Path to: sklearn/manifold/benchmarks/datasets/
DATASET_ROOT = current_dir / "datasets"


def stress_and_optimal_scale_chunked(X, embedding, block_size=2048):
    """Exact stress + stress-optimal rescale, in O(N * block_size) memory.

    Classical-MDS-family embeddings (Pivot MDS, ClassicalMDS) fit inner
    products, not distances, so they land at a stress-suboptimal global
    scale. The comparable-to-SGD number requires rescaling by the
    closed-form optimum ``alpha = <D_emb, D> / <D_emb, D_emb>`` first.

    Naively that needs two full N x N matrices (~40 GB at N=50,000). But
    everything follows from three scalars, each accumulable over row
    blocks:

        S_ee = sum D_emb^2 ,  S_ed = sum D_emb * D ,  S_dd = sum D^2

    giving, in closed form,

        alpha         = S_ed / S_ee
        stress_raw    = 0.5 * (S_ee - 2*S_ed + S_dd)
        stress_scaled = 0.5 * (S_dd - S_ed^2 / S_ee)

    so one blocked pass yields both stresses and the scale factor.

    Returns ``(stress_scaled, stress_raw, alpha)``. Both stresses use the
    full-matrix convention (both (i,j) and (j,i) counted), matching
    SGDMDS's ``_full_stress`` and every other stress in these benchmarks.
    """
    from sklearn.metrics import euclidean_distances

    X = np.asarray(X, dtype=np.float64)
    embedding = np.asarray(embedding, dtype=np.float64)
    n = X.shape[0]
    S_ee = S_ed = S_dd = 0.0
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        d_x = euclidean_distances(X[start:stop], X)
        d_e = euclidean_distances(embedding[start:stop], embedding)
        S_ee += float(np.einsum("ij,ij->", d_e, d_e))
        S_ed += float(np.einsum("ij,ij->", d_e, d_x))
        S_dd += float(np.einsum("ij,ij->", d_x, d_x))
    alpha = (S_ed / S_ee) if S_ee > 0.0 else 1.0
    stress_raw = 0.5 * (S_ee - 2.0 * S_ed + S_dd)
    stress_scaled = 0.5 * (S_dd - (S_ed * S_ed / S_ee)) if S_ee > 0.0 else stress_raw
    return stress_scaled, stress_raw, alpha


def stress_and_optimal_scale_chunked_multi(X, embeddings, block_size=2048):
    """Batched version of :func:`stress_and_optimal_scale_chunked`.

    Scores MANY embeddings of the same X in a single blocked pass. The
    input-distance block ``d_x`` costs O(block * N * D) and is identical for
    every embedding, while each embedding's own block costs only
    O(block * N * n_components) -- so computing ``d_x`` once and reusing it
    across m embeddings is ~m times cheaper than m separate calls whenever
    D >> n_components.

    Concretely, scoring 30 Pivot-MDS embeddings of a D=3072 dataset at
    N=50,000 drops from ~26 minutes to ~1 minute.

    Returns a list of ``(stress_scaled, stress_raw, alpha)``, one per input
    embedding, in the order given.
    """
    from sklearn.metrics import euclidean_distances

    X = np.asarray(X, dtype=np.float64)
    embs = [np.asarray(e, dtype=np.float64) for e in embeddings]
    n = X.shape[0]
    m = len(embs)
    S_ee = np.zeros(m)
    S_ed = np.zeros(m)
    S_dd = 0.0
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        d_x = euclidean_distances(X[start:stop], X)
        S_dd += float(np.einsum("ij,ij->", d_x, d_x))
        for t, e in enumerate(embs):
            d_e = euclidean_distances(e[start:stop], e)
            S_ee[t] += float(np.einsum("ij,ij->", d_e, d_e))
            S_ed[t] += float(np.einsum("ij,ij->", d_e, d_x))
    out = []
    for t in range(m):
        if S_ee[t] > 0.0:
            alpha = S_ed[t] / S_ee[t]
            scaled = 0.5 * (S_dd - S_ed[t] * S_ed[t] / S_ee[t])
        else:
            alpha, scaled = 1.0, 0.5 * S_dd
        raw = 0.5 * (S_ee[t] - 2.0 * S_ed[t] + S_dd)
        out.append((scaled, raw, alpha))
    return out


def stress_exact_chunked(X, embedding, block_size=2048):
    """Exact raw stress in O(N * block_size) memory (no full N x N matrix).

    Computes ``0.5 * sum_{i,j} (d_ij(embedding) - delta_ij(X))**2`` -- the
    same full-matrix convention as SGDMDS's internal ``_full_stress`` (both
    (i,j) and (j,i) counted, diagonal zero), so values are directly
    comparable with every previous experiment's ``stress``.

    Row-block streaming: for each block of rows, build only the
    (block x N) slices of the feature-space and embedding-space distance
    matrices and accumulate. Removes the O(N^2) memory wall that capped
    exact scoring at N=20,000 -- exact stress at N=100,000 needs ~2 GB of
    transient buffers instead of two 80 GB matrices. Time is still O(N^2 D)
    but BLAS-parallel and, in the optimized Experiment K protocol, excluded
    from benchmark timings.
    """
    from sklearn.metrics import euclidean_distances

    X = np.asarray(X, dtype=np.float64)
    embedding = np.asarray(embedding, dtype=np.float64)
    n = X.shape[0]
    total = 0.0
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        d_x = euclidean_distances(X[start:stop], X)
        d_e = euclidean_distances(embedding[start:stop], embedding)
        d_e -= d_x
        total += float(np.einsum("ij,ij->", d_e, d_e))
    return 0.5 * total

def generate_torus(n_samples=2000, R=1.0, r=0.4, noise=0.0, random_state=None):
    """
    Generates a synthetic Torus. Notorious for trapping MDS in folded minima.
    """
    rng = check_random_state(random_state)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    X = np.column_stack([x, y, z])
    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)
    return X, None

def load_local_dataset(
    name,
    n_samples=None,
    random_state=42,
    target_dim=None,
    noise_scale=0.1,
    lift_synthetic=True,
):
    """
    Loads a dataset from the local 'datasets' folder or generates it if synthetic.

    Expected structure:
    datasets/
       <name>/
           X.npy (or X.csv)
           y.npy (or y.csv) [Optional]
    """
    name = name.lower()

    def _standardize(X):
        return StandardScaler().fit_transform(X)

    def _maybe_lift(X):
        if (target_dim is not None) and (lift_synthetic is True):
            X = lift_to_high_dim(X, target_dim=target_dim, noise_scale=noise_scale, random_state=random_state)
        return X

    if name == "swiss_roll":
        print(f"Generating synthetic {name}...")
        X, y = make_swiss_roll(n_samples=n_samples or 2000, noise=0.1, random_state=random_state)
        X = _standardize(X)
        X = _maybe_lift(X)
        X = _standardize(X)
        print(f"  Loaded {name}: N={X.shape[0]}, D={X.shape[1]}")
        return X, y

    elif name == "s_curve":
        print(f"Generating synthetic {name}...")
        X, y = make_s_curve(n_samples=n_samples or 2000, noise=0.1, random_state=random_state)
        X = _standardize(X)
        X = _maybe_lift(X)
        X = _standardize(X)
        print(f"  Loaded {name}: N={X.shape[0]}, D={X.shape[1]}")
        return X, y

    elif name == "torus":
        print(f"Generating synthetic {name}...")
        X, y = generate_torus(n_samples=n_samples or 2000, random_state=random_state)
        X = _standardize(X)
        X = _maybe_lift(X)
        X = _standardize(X)
        print(f"  Loaded {name}: N={X.shape[0]}, D={X.shape[1]}")
        return X, y

    target_dir = DATASET_ROOT / name
    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {target_dir}")

    print(f"Loading local dataset: {name} from {target_dir}...")

    if (target_dir / "X.npy").exists():
        X = np.load(target_dir / "X.npy")
    elif (target_dir / "X.csv").exists():
        try:
            X = pd.read_csv(target_dir / "X.csv").values
        except Exception:
            X = pd.read_csv(target_dir / "X.csv", header=None).values
    else:
        raise FileNotFoundError(f"Could not find X.npy or X.csv in {target_dir}")

    y = None
    if (target_dir / "y.npy").exists():
        y = np.load(target_dir / "y.npy")
    elif (target_dir / "y.csv").exists():
        try:
            y = pd.read_csv(target_dir / "y.csv").values.ravel()
        except Exception:
            y = pd.read_csv(target_dir / "y.csv", header=None).values.ravel()

    if n_samples is not None and n_samples < X.shape[0]:
        print(f"  Subsampling from {X.shape[0]} to {n_samples}...")
        if y is not None:
            X, y = resample(X, y, n_samples=n_samples, random_state=random_state, replace=False)
        else:
            X = resample(X, n_samples=n_samples, random_state=random_state, replace=False)

    X = _standardize(X)

    print(f"  Loaded {name}: N={X.shape[0]}, D={X.shape[1]}")
    return X, y

def lift_to_high_dim(X, target_dim, noise_scale=0.1, random_state=42):
    """
    Lift an (N, d0) dataset into (N, target_dim) by appending Gaussian noise dims.
    """
    if target_dim <= X.shape[1]:
        return X

    rng = check_random_state(random_state)
    n, d0 = X.shape
    extra = rng.normal(loc=0.0, scale=noise_scale, size=(n, target_dim - d0))
    return np.hstack([X, extra])

def run_smacof_benchmark(X, dissimilarity="euclidean", random_state=None, max_iter=300):
    """
    Runs SMACOF. Supports 'precomputed' or 'euclidean'.
    If precomputed, X must be the distance matrix.
    """
    print(f"  [SMACOF] Running on N={X.shape[0]} (Mode: {dissimilarity})...")
    
    mds = SMACOF_MDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=max_iter,
        eps=1e-12,
        random_state=random_state,
        dissimilarity=dissimilarity, # Passes 'precomputed' or 'euclidean'
        n_jobs=1,
        normalized_stress=False 
    )
    
    t0 = time()
    embedding = mds.fit_transform(X)
    total_time = time() - t0
    
    if hasattr(mds, "stress_history_") and mds.stress_history_ is not None:
        history = mds.stress_history_
    else:
        history = [(total_time, mds.stress_)]
        
    return {
        "algo": "SMACOF",
        "embedding": embedding,
        "stress": mds.stress_,
        "history": history,
        "time": total_time,
        "n_iter": mds.n_iter_
    }


def run_sgd_benchmark(X, dissimilarity="precomputed", random_state=None, max_iter=30,
                      switch_ratio=0.5, lr=0.5, epsilon=0.001, compute_stress=True):
    """
    Runs SGDMDS. Supports 'precomputed', 'euclidean', 'lazy'.

    compute_stress=False skips all in-fit stress evaluation (the embedding is
    identical); the caller scores the returned embedding externally (e.g.
    stress_exact_chunked) and the returned "time" is then pure fit time.
    """
    print(f"  [SGD-{dissimilarity}] Running on N={X.shape[0]}...")

    sgd = SGDMDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=max_iter,
        learning_rate_init=lr,
        scheduler="hybrid",
        scheduler_switch_ratio=switch_ratio,
        epsilon=epsilon,
        dissimilarity=dissimilarity,
        compute_stress=compute_stress,
        random_state=random_state,
        n_jobs=1
    )

    t0 = time()
    embedding = sgd.fit_transform(X)
    total_time = time() - t0

    return {
        "algo": f"SGD-{dissimilarity}",
        "embedding": embedding,
        "stress": sgd.stress_,
        "history": sgd.stress_history_,
        "time": total_time,
        "n_iter": sgd.n_iter_
    }


def run_random_budget_benchmark(X, n_updates_per_epoch, random_state=None,
                                max_iter=30, switch_ratio=0.5, lr=0.5,
                                epsilon=0.001, compute_stress=True):
    """
    Runs SGDMDS with sampling_strategy='random' at a fixed per-epoch budget.

    n_updates_per_epoch: int (or 'auto' for the full-cycle default).
    Used by Experiment I to characterise the stress / wall-time Pareto curve
    of budgeted uniform random pair sampling.

    compute_stress=False skips all in-fit stress evaluation -- in lazy mode
    that avoids materializing the full N x N distance matrix just for
    scoring (the O(N*D) fast-mode memory promise then actually holds). The
    caller scores externally; "time" is then pure fit time.
    """
    budget_label = (
        "auto" if n_updates_per_epoch == "auto" else f"b{n_updates_per_epoch}"
    )
    print(f"  [SGD-random-{budget_label}] Running on N={X.shape[0]}, "
          f"max_iter={max_iter}...")

    sgd = SGDMDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=max_iter,
        learning_rate_init=lr,
        scheduler="hybrid",
        scheduler_switch_ratio=switch_ratio,
        epsilon=epsilon,
        dissimilarity="lazy",
        sampling_strategy="random",
        n_updates_per_epoch=n_updates_per_epoch,
        compute_stress=compute_stress,
        random_state=random_state,
        n_jobs=1,
    )

    t0 = time()
    embedding = sgd.fit_transform(X)
    total_time = time() - t0

    return {
        "algo": f"SGD-random-{budget_label}",
        "embedding": embedding,
        "stress": sgd.stress_,
        "history": sgd.stress_history_,
        "time": total_time,
        "n_iter": sgd.n_iter_,
        "n_updates_per_epoch": n_updates_per_epoch,
    }


def run_pivot_benchmark(X, dissimilarity="lazy", n_pivots="auto", random_state=None,
                        max_iter=30, switch_ratio=0.5, lr=0.5, epsilon=0.001,
                        pivot_strategy="maxmin", pivot_pca_dim=30):
    """
    Runs SGDMDS with pivot sampling strategy.
    dissimilarity: 'lazy' or 'euclidean' (precomputed pivot pairs).
    n_pivots: int or 'auto'.
    pivot_strategy: 'maxmin' or 'maxmin_pca'.
    pivot_pca_dim: PCA target dim (only used when pivot_strategy='maxmin_pca').
    """
    k_label = n_pivots if isinstance(n_pivots, str) else f"k{n_pivots}"
    pca_tag = f"-pca{pivot_pca_dim}" if pivot_strategy == "maxmin_pca" else ""
    print(f"  [SGD-pivot-{dissimilarity}{pca_tag}-{k_label}] Running on N={X.shape[0]}...")

    sgd = SGDMDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=max_iter,
        learning_rate_init=lr,
        scheduler="hybrid",
        scheduler_switch_ratio=switch_ratio,
        epsilon=epsilon,
        dissimilarity=dissimilarity,
        sampling_strategy="pivot",
        n_pivots=n_pivots,
        pivot_strategy=pivot_strategy,
        pivot_pca_dim=pivot_pca_dim,
        random_state=random_state,
        n_jobs=1,
    )

    t0 = time()
    embedding = sgd.fit_transform(X)
    total_time = time() - t0

    actual_k = int(sgd.pivot_indices_.shape[0])
    return {
        "algo": f"SGD-pivot-{dissimilarity}-{k_label}",
        "embedding": embedding,
        "stress": sgd.stress_,
        "history": sgd.stress_history_,
        "time": total_time,
        "n_iter": sgd.n_iter_,
        "n_pivots": actual_k,
    }


def run_pivot_mds_benchmark(X, n_pivots="auto", random_state=None,
                            pivot_strategy="maxmin", pivot_pca_dim=30,
                            scoring_block_size=2048, score=True):
    """
    Runs standalone Pivot MDS (Brandes & Pich, 2007) — no SGD.

    Spectral baseline: select k pivots, then compute the embedding from a
    single SVD on the centered N-by-k squared-distance matrix.

    n_pivots: int or 'auto'.
    pivot_strategy: 'maxmin' (raw farthest-point) or 'maxmin_pca' (farthest-
        point in a PCA-reduced feature space).
    pivot_pca_dim: PCA target dim (only used when pivot_strategy='maxmin_pca').
    """
    strategy_tag = (
        "maxmin" if pivot_strategy == "maxmin" else f"pca{pivot_pca_dim}"
    )
    k_label = n_pivots if isinstance(n_pivots, str) else f"k{n_pivots}"
    print(f"  [PivotMDS-{strategy_tag}-{k_label}] Running on N={X.shape[0]}...")

    rng = check_random_state(random_state)
    n_samples = X.shape[0]
    k_resolved = _resolve_n_pivots(n_pivots, 2, n_samples)

    X64 = np.ascontiguousarray(X, dtype=np.float64)

    # NOTE: pivot SELECTION is part of the algorithm's cost and must be
    # timed. Greedy maxmin is O(k*N*D) and single-threaded, so it dominates:
    # at N=50,000, D=784, k=200 it costs ~15 s against ~0.4 s for the
    # (BLAS-parallel) spectral solve. An earlier version of this wrapper
    # started the timer after selection and so under-reported total time by
    # ~40x. `time` is now the full cost of producing an embedding;
    # `time_pivot_selection` / `time_spectral` give the split.
    t_sel0 = time()
    if pivot_strategy == "maxmin_pca":
        features_proj = _maybe_pca_project(X64, pivot_pca_dim, rng)
        pivot_indices = _select_pivots_maxmin(
            features_proj, k_resolved, rng, is_lazy=True,
        )
    else:
        pivot_indices = _select_pivots_maxmin(
            X64, k_resolved, rng, is_lazy=True,
        )
    selection_time = time() - t_sel0

    t0 = time()
    embedding, _ = _pivot_mds_fn(
        X64, pivot_indices, n_components=2, metric="euclidean",
    )
    spectral_time = time() - t0
    total_time = selection_time + spectral_time

    # Full N(N-1)/2 stress, for direct comparison with SGD results.
    #
    # Classical-MDS-family embeddings fit inner products (strain), not
    # distances, so they come out at a stress-suboptimal global scale.
    # SGD optimizes scale implicitly while minimizing stress, so an
    # unscaled comparison overstates the spectral method's stress gap
    # (by up to ~6x on the H datasets). The embedding is therefore
    # rescaled by the closed-form stress-optimal factor
    #     alpha = <D_emb, D> / <D_emb, D_emb>
    # before stress is computed. The unscaled value is kept as
    # "stress_raw" for reference.
    # Computed in one row-blocked pass (O(N * block) memory) so this scales
    # past N=20,000; the previous full-matrix version needed two N x N
    # matrices (~40 GB at N=50,000). Scoring happens outside the timed
    # region above, so `time` is pure solve time.
    #
    # score=False returns the UNSCALED embedding and no stress, for callers
    # that will batch-score many embeddings of the same X together (see
    # stress_and_optimal_scale_chunked_multi) -- far cheaper when D is large,
    # since the input-distance blocks are then computed once, not once per fit.
    if not score:
        return {
            "algo": f"PivotMDS-{strategy_tag}-{k_label}",
            "embedding": embedding,
            "stress": None,
            "stress_raw": None,
            "scale_alpha": None,
            "history": [(total_time, None)],
            "time": total_time,
            "time_pivot_selection": selection_time,
            "time_spectral": spectral_time,
            "time_scoring": 0.0,
            "n_iter": 0,
            "n_pivots": int(pivot_indices.shape[0]),
            "pivot_indices": pivot_indices,
        }

    t_score0 = time()
    stress, stress_raw, alpha = stress_and_optimal_scale_chunked(
        X64, embedding, block_size=scoring_block_size
    )
    scoring_time = time() - t_score0
    embedding = embedding * alpha

    return {
        "algo": f"PivotMDS-{strategy_tag}-{k_label}",
        "embedding": embedding,
        "stress": stress,
        "stress_raw": stress_raw,
        "scale_alpha": alpha,
        "history": [(total_time, stress)],
        "time": total_time,
        "time_pivot_selection": selection_time,
        "time_spectral": spectral_time,
        "time_scoring": scoring_time,
        "n_iter": 0,
        "n_pivots": int(pivot_indices.shape[0]),
        "pivot_indices": pivot_indices,
    }


def run_squad_mds_benchmark(X, n_iter=1000, lr=550.0, exaggerate_d=True,
                            stop_exaggeration=0.6, metric="euclidean",
                            init="pca", momentum=0.0, lr_decay="geometric",
                            max_step_frac=1.0, random_state=None,
                            scoring_block_size=2048, score=True,
                            record_loss=False):
    """
    Runs standalone SQuaD-MDS (Lambert et al., 2022) — no SGD-MDS involved.

    A *stochastic quartet* baseline: each iteration partitions the points
    into random disjoint quartets and follows the gradient of a per-quartet
    cost over sum-normalized ("relative") distances. O(N*D) per iteration,
    no distance matrix.

    THE EMBEDDING HAS NO MEANINGFUL GLOBAL SCALE. The optimized cost
    normalizes both the high-dimensional and the embedding distances of each
    quartet by their own sum, so it is invariant to a global rescale — the
    method imposes no scale constraint by design (paper §2.2). Scoring raw
    stress on the returned coordinates would measure an arbitrary scale, so
    the closed-form stress-optimal factor alpha = <D_emb,D>/<D_emb,D_emb> is
    applied first, exactly as for the classical-MDS family in
    run_pivot_mds_benchmark. `stress` is the rescaled (comparable) number;
    `stress_raw` and `scale_alpha` are also returned.

    Note this rescale is a WEAKER repair here than for Pivot MDS: SQuaD-MDS
    normalizes PER QUARTET, not globally, so a single global alpha need not
    be able to undo it. Watch `scale_alpha`'s spread across seeds.

    max_step_frac is a trust region on the per-iteration point displacement,
    guarding the O(lr / S) gradient blow-up that duplicate points cause (see
    sklearn/manifold/_squad_mds.py). It is swept, not fixed, because it
    changes results on the 7 collection datasets that carry duplicates;
    `clip_frac` in the returned dict reports how often it fired, and is
    0.000 on clean data, where the guard is provably inert.

    score=False returns the UNSCALED embedding and no stress, for callers
    batching many embeddings of the same X through
    stress_and_optimal_scale_chunked_multi.
    """
    from sklearn.manifold._squad_mds import squad_mds

    exa_tag = f"-exa{stop_exaggeration:g}" if exaggerate_d else ""
    label = f"SQuaD-MDS-i{n_iter}-lr{lr:g}{exa_tag}"
    print(f"  [{label}] Running on N={X.shape[0]}...")

    X64 = np.ascontiguousarray(X, dtype=np.float64)

    # PCA initialisation is part of the algorithm as the authors publish it
    # (their main.py does it before the timer, but it is not free at large
    # N*D), so it is inside the timed region. `time_init` gives the split.
    t0 = time()
    embedding, info = squad_mds(
        X64, n_components=2, n_iter=n_iter, lr=lr,
        exaggerate_d=exaggerate_d, stop_exaggeration=stop_exaggeration,
        metric=metric, init=init, momentum=momentum, lr_decay=lr_decay,
        max_step_frac=max_step_frac, random_state=random_state,
        record_loss=record_loss,
    )
    total_time = time() - t0

    result = {
        "algo": label,
        "embedding": embedding,
        "history": [(total_time, None)],
        "time": total_time,
        "n_iter": info["n_iter"],
        "n_quartets": info["n_quartets"],
        "n_clipped": info["n_clipped"],
        "clip_frac": info["n_clipped"] / max(info["n_point_steps"], 1),
        "loss_history": info["loss_history"],
        "stress": None,
        "stress_raw": None,
        "scale_alpha": None,
        "time_scoring": 0.0,
    }
    if not score:
        return result

    t_score0 = time()
    stress, stress_raw, alpha = stress_and_optimal_scale_chunked(
        X64, embedding, block_size=scoring_block_size
    )
    result.update({
        "embedding": embedding * alpha,
        "stress": stress,
        "stress_raw": stress_raw,
        "scale_alpha": alpha,
        "history": [(total_time, stress)],
        "time_scoring": time() - t_score0,
    })
    return result


def run_hybrid_benchmark(X, n_pivots="auto", hybrid_alpha=0.5,
                         random_state=None, max_iter=30, switch_ratio=0.5,
                         lr=0.5, epsilon=0.001, pivot_strategy="maxmin",
                         pivot_pca_dim=30):
    """
    Runs SGDMDS with hybrid (pivot + random) sampling — lazy mode only.

    Per epoch, performs ``hybrid_alpha * k * N`` pivot updates and
    ``(1 - hybrid_alpha) * k * N`` random updates, keeping the per-epoch
    update budget equal to pure-pivot sampling.

    n_pivots: int or 'auto'.
    hybrid_alpha: float in [0, 1] — fraction of pivot updates per epoch.
    pivot_strategy: 'maxmin' or 'maxmin_pca'.
    pivot_pca_dim: PCA target dim (only used when pivot_strategy='maxmin_pca').
    """
    k_label = n_pivots if isinstance(n_pivots, str) else f"k{n_pivots}"
    pca_tag = f"-pca{pivot_pca_dim}" if pivot_strategy == "maxmin_pca" else ""
    print(f"  [SGD-hybrid{pca_tag}-a{hybrid_alpha:.2f}-{k_label}] "
          f"Running on N={X.shape[0]}...")

    sgd = SGDMDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=max_iter,
        learning_rate_init=lr,
        scheduler="hybrid",
        scheduler_switch_ratio=switch_ratio,
        epsilon=epsilon,
        dissimilarity="lazy",
        sampling_strategy="hybrid",
        n_pivots=n_pivots,
        pivot_strategy=pivot_strategy,
        pivot_pca_dim=pivot_pca_dim,
        hybrid_alpha=hybrid_alpha,
        random_state=random_state,
        n_jobs=1,
    )

    t0 = time()
    embedding = sgd.fit_transform(X)
    total_time = time() - t0

    actual_k = int(sgd.pivot_indices_.shape[0])
    return {
        "algo": f"SGD-hybrid-a{hybrid_alpha:.2f}-{k_label}",
        "embedding": embedding,
        "stress": sgd.stress_,
        "history": sgd.stress_history_,
        "time": total_time,
        "n_iter": sgd.n_iter_,
        "n_pivots": actual_k,
        "hybrid_alpha": hybrid_alpha,
    }


def per_point_stress(embedding, dissimilarity_matrix):
    """
    Per-point raw stress contribution.

    Returns an array of length n_samples where entry i is
    ``sum_j (d_ij(emb) - delta_ij)^2`` (no 0.5 factor; this is the
    sum of squared per-pair residuals incident to point i).

    Used by the decisive Phase-2 plot: histogram of per-point stress,
    colored by pivot vs non-pivot.
    """
    from sklearn.metrics import euclidean_distances
    D_emb = euclidean_distances(embedding, embedding)
    R = (D_emb - dissimilarity_matrix) ** 2
    np.fill_diagonal(R, 0.0)
    return R.sum(axis=1)