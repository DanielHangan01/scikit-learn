"""
One-off data prep: full 70k Fashion-MNIST -> datasets/fashion_mnist_full/.

Source: OpenML ("Fashion-MNIST", version 1), fetched via scikit-learn -- no
extra dependencies, no Kaggle account. The 60k/10k train/test split is
irrelevant for MDS (unsupervised), so we keep all 70k rows.

Matches the existing 3000-point `fashion_mnist` convention: X is float32 in
[0, 1] (raw 0-255 / 255), shape (70000, 784); y is int64 in [0, 9]. Written to
a NEW dir so the small `fashion_mnist` used by Experiment J stays untouched.

Run once (from this benchmarks/ directory):
    python prep_fashion_mnist_full.py
"""

from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml

OUT = Path(__file__).parent / "datasets" / "fashion_mnist_full"


def main():
    print("Fetching Fashion-MNIST from OpenML (~55 MB, may take a minute)...")
    d = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = np.asarray(d.data, dtype=np.float32) / 255.0        # (70000, 784) in [0, 1]
    y = np.asarray(d.target).astype(np.int64)               # (70000,) in [0, 9]
    assert X.shape == (70000, 784), X.shape
    assert X.min() >= 0.0 and X.max() <= 1.0

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X.npy", X)
    np.save(OUT / "y.npy", y)
    print(f"Saved X {X.shape} {X.dtype} [{X.min():.3f}, {X.max():.3f}] "
          f"and y {y.shape} to {OUT}/")


if __name__ == "__main__":
    main()
