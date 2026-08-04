"""
Pivot MDS (Brandes & Pich, 2007) — spectral classical-MDS variant on a small
pivot subset.

References
----------
.. [1] Brandes, U., & Pich, C. (2007). Eigensolver methods for progressive
   multidimensional scaling of large data. In *International Symposium on
   Graph Drawing* (pp. 42-53). Springer.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import linalg

from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import svd_flip


def pivot_mds(X, pivot_indices, n_components=2, *, metric="euclidean"):
    """Embed ``X`` with Pivot MDS (Brandes & Pich, 2007).

    Builds the N-by-k matrix of squared Euclidean distances between every
    point and each pivot, applies Brandes-Pich rectangular double-centering,
    and takes the top ``n_components`` left singular vectors (scaled by the
    square root of the corresponding singular value) as the embedding.

    With ``pivot_indices = np.arange(n_samples)`` and Euclidean ``X``, this
    reduces to classical MDS (Torgerson scaling).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Feature matrix when ``metric='euclidean'``; symmetric distance matrix
        when ``metric='precomputed'``.
    pivot_indices : array-like of int, shape (n_pivots,)
        Indices into ``X`` selecting the pivot points. Must contain values
        in ``[0, n_samples)``; duplicates are allowed but produce a
        rank-deficient centered matrix.
    n_components : int, default=2
        Number of output dimensions. Must satisfy ``n_components <= n_pivots``.
    metric : {'euclidean', 'precomputed'}, default='euclidean'
        How to interpret ``X``.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        Sign-stabilized embedding (via ``svd_flip``).
    singular_values : ndarray of shape (n_components,)
        Top ``n_components`` singular values of the centered matrix, in
        decreasing order.
    """
    pivot_indices = np.asarray(pivot_indices, dtype=np.int64).ravel()
    n_samples = X.shape[0]
    k = pivot_indices.shape[0]

    if k < n_components:
        raise ValueError(
            f"n_pivots ({k}) must be >= n_components ({n_components})."
        )
    if (pivot_indices < 0).any() or (pivot_indices >= n_samples).any():
        raise ValueError("pivot_indices contain out-of-range values.")

    if metric == "precomputed":
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(
                "With metric='precomputed', X must be a square distance matrix."
            )
        D_pivot = np.asarray(X[:, pivot_indices], dtype=np.float64)
    elif metric == "euclidean":
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        D_pivot = euclidean_distances(X64, X64[pivot_indices])
    else:
        raise ValueError(
            f"metric must be 'euclidean' or 'precomputed' (got {metric!r})."
        )

    # Brandes & Pich §3 rectangular double-centering of squared distances.
    C = D_pivot ** 2
    row_means = C.mean(axis=1, keepdims=True)
    col_means = C.mean(axis=0, keepdims=True)
    grand_mean = float(C.mean())
    B = -0.5 * (C - row_means - col_means + grand_mean)

    U, S, Vt = linalg.svd(B, full_matrices=False)
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    U, _ = svd_flip(U, Vt)

    embedding = U * np.sqrt(S)
    return embedding, S
