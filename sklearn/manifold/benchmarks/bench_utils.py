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
    from sklearn.manifold._sgd_mds import SGDMDS
except ImportError:
    print("Could not import SGDMDS from sklearn.manifold._sgd_mds.")
    print("Make sure you are running this from the benchmarks directory.")
    sys.exit(1)

from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, check_random_state

# Path to: sklearn/manifold/benchmarks/datasets/
DATASET_ROOT = current_dir / "datasets"

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
                      switch_ratio=0.5, lr=0.5, epsilon=0.001):
    """
    Runs SGDMDS. Supports 'precomputed', 'euclidean', 'lazy'.
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


def run_pivot_benchmark(X, dissimilarity="lazy", n_pivots="auto", random_state=None,
                        max_iter=30, switch_ratio=0.5, lr=0.5, epsilon=0.001):
    """
    Runs SGDMDS with pivot sampling strategy.
    dissimilarity: 'lazy' or 'euclidean' (precomputed pivot pairs).
    n_pivots: int or 'auto'.
    """
    k_label = n_pivots if isinstance(n_pivots, str) else f"k{n_pivots}"
    print(f"  [SGD-pivot-{dissimilarity}-{k_label}] Running on N={X.shape[0]}...")

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
        pivot_strategy="maxmin",
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