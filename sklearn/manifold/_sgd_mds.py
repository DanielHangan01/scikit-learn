"""Multidimensional Scaling via Stochastic Gradient Descent (SGD-MDS).

This module implements metric and non-metric MDS by minimizing the raw
stress objective with stochastic gradient descent. Compared to SMACOF
(see :mod:`sklearn.manifold._mds`), SGD-MDS scales to large ``n_samples``
and supports a memory-light ``"lazy"`` mode that avoids materializing the
full ``N x N`` dissimilarity matrix.

Three sampling strategies are supported:

- ``"cycle"``  : deterministic shuffle over all ``N(N-1)/2`` unique pairs.
- ``"random"`` : random pair sampling (lazy mode only).
- ``"pivot"``  : SGD restricted to pairs ``(i, p)`` where ``p`` is one of
  ``n_pivots`` carefully chosen pivots, reducing per-epoch cost from
  ``O(N^2)`` to ``O(kN)``. Pivots are selected with the MaxMin greedy
  farthest-point heuristic (Brandes & Pich, 2007, §3.2).
"""

# Author: Daniel Hangan
# SPDX-License-Identifier: BSD-3-Clause

import time
from numbers import Integral, Real

import numpy as np

from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..isotonic import IsotonicRegression
from ..metrics import euclidean_distances
from ..utils import check_random_state
from ..utils import shuffle as sklearn_shuffle
from ..utils._param_validation import Interval, StrOptions
from ..utils.validation import check_array, check_symmetric, validate_data

try:
    from ._sgd_mds_cython import (
        run_sgd_epoch,
        run_sgd_epoch_lazy_random_native,
        run_sgd_epoch_pivot_lazy,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


# =============================================================================
# Learning-rate schedulers
# =============================================================================


class _BaseScheduler:
    """Base class for learning-rate schedulers used by :class:`SGDMDS`."""

    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        self.lr_init = float(lr_init)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)

    def get_rate(self, t):
        raise NotImplementedError


class _ConstantScheduler(_BaseScheduler):
    def get_rate(self, t):
        return self.lr_init


class _ExponentialScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        lr_final = self.epsilon / self.w_max
        if self.lr_init > lr_final:
            self.decay = -np.log(lr_final / self.lr_init) / self.max_iter
        else:
            self.decay = 0.0

    def get_rate(self, t):
        return self.lr_init * np.exp(-self.decay * t)


class _HarmonicScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        lr_final = self.epsilon / self.w_max
        if self.lr_init <= lr_final:
            self.decay = 0.0
        else:
            ratio = self.lr_init / lr_final
            self.decay = (ratio - 1.0) / self.max_iter

    def get_rate(self, t):
        return self.lr_init / (1.0 + self.decay * t)


class _HybridScheduler(_BaseScheduler):
    """Exponential decay (fast unfolding) followed by harmonic decay (refinement).

    Switches at ``switch_ratio * max_iter`` regardless of the weight bounds.
    """

    def __init__(self, lr_init, w_min, w_max, max_iter, switch_ratio=0.4,
                 epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        self.tau = int(max_iter * switch_ratio)

        # Phase 1: exponential to lr_mid at the switch point.
        lr_mid = self.lr_init * 0.1
        if self.tau > 0:
            self.decay_exp = -np.log(lr_mid / self.lr_init) / self.tau
        else:
            self.decay_exp = 0.0
            lr_mid = self.lr_init

        # Phase 2: harmonic from lr_mid to lr_final.
        self.a = lr_mid
        lr_final = self.epsilon / self.w_max
        remaining_iter = max_iter - self.tau
        if remaining_iter > 0 and lr_mid > lr_final:
            self.b = (lr_mid / lr_final - 1) / remaining_iter
        else:
            self.b = 0.0

    def get_rate(self, t):
        if t < self.tau:
            return self.lr_init * np.exp(-self.decay_exp * t)
        return self.a / (1.0 + self.b * (t - self.tau))


_SCHEDULERS = {
    "constant": _ConstantScheduler,
    "exponential": _ExponentialScheduler,
    "harmonic": _HarmonicScheduler,
    "hybrid": _HybridScheduler,
}


def _make_scheduler(name, *, lr_init, w_min, w_max, max_iter, switch_ratio, epsilon):
    """Instantiate a learning-rate scheduler by name."""
    cls = _SCHEDULERS.get(name, _ExponentialScheduler)
    return cls(
        lr_init=lr_init,
        w_min=w_min,
        w_max=w_max,
        max_iter=max_iter,
        switch_ratio=switch_ratio,
        epsilon=epsilon,
    )


# =============================================================================
# Pivot selection (Brandes & Pich, 2007, §3.2)
# =============================================================================


def _resolve_n_pivots(n_pivots, n_components, n_samples):
    """Resolve ``"auto"`` and validate against ``(n_components, n_samples)``."""
    if n_pivots == "auto":
        k = min(50, max(n_components + 1, n_samples // 20))
    else:
        k = int(n_pivots)
    if k <= n_components:
        raise ValueError(
            f"n_pivots ({k}) must be greater than n_components ({n_components})."
        )
    if k > n_samples:
        raise ValueError(f"n_pivots ({k}) must be <= n_samples ({n_samples}).")
    return k


def _resolve_n_updates_per_epoch(n_updates_per_epoch, n_samples):
    """Resolve ``"auto"`` to the full-cycle update count ``N(N-1)/2``.

    Used by ``sampling_strategy='random'`` to control the per-epoch budget.
    """
    if n_updates_per_epoch == "auto":
        return (n_samples * (n_samples - 1)) // 2
    return int(n_updates_per_epoch)


def _maybe_pca_project(features, pca_dim, rng):
    """Project ``features`` to ``pca_dim`` PCA components for pivot selection.

    Caps the target dimensionality at ``min(n_features, n_samples)``. When the
    cap equals ``n_features`` (i.e. PCA cannot reduce), the original feature
    matrix is returned unchanged. The returned array is contiguous float64
    so it can feed directly into :func:`_select_pivots_maxmin`.

    Notes
    -----
    PCA is only used here to reshape the geometry seen by the MaxMin scan;
    the SGD update loop still sees dissimilarities from the original feature
    space.
    """
    n_samples, n_feat = features.shape
    target = min(int(pca_dim), n_feat, n_samples)
    if target >= n_feat:
        return np.ascontiguousarray(features, dtype=np.float64)

    from ..decomposition import PCA

    pca_seed = int(rng.randint(np.iinfo(np.int32).max))
    pca = PCA(n_components=target, random_state=pca_seed)
    projected = pca.fit_transform(features)
    return np.ascontiguousarray(projected, dtype=np.float64)


def _select_pivots_maxmin(data, n_pivots, rng, *, is_lazy):
    """Greedy MaxMin (farthest-point) pivot selection.

    Each next pivot maximizes the minimum distance to all previously selected
    pivots — a 2-approximation of k-center.

    Parameters
    ----------
    data : ndarray
        Feature matrix of shape ``(n_samples, n_features)`` when ``is_lazy``,
        otherwise a dissimilarity matrix of shape ``(n_samples, n_samples)``.
    n_pivots : int
    rng : np.random.RandomState
    is_lazy : bool
        If True, distances to ``data[p]`` are recomputed in feature space.

    Returns
    -------
    pivot_indices : ndarray of shape (n_pivots,), dtype int32
    """
    n_samples = data.shape[0]
    pivots = np.empty(n_pivots, dtype=np.int32)
    pivots[0] = rng.randint(n_samples)
    min_dist = np.full(n_samples, np.inf, dtype=np.float64)

    for k in range(n_pivots):
        p = int(pivots[k])
        if is_lazy:
            diff = data - data[p]
            new_dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        else:
            new_dists = data[p].astype(np.float64, copy=False)

        np.minimum(min_dist, new_dists, out=min_dist)

        if k + 1 < n_pivots:
            pivots[k + 1] = int(np.argmax(min_dist))
    return pivots


# =============================================================================
# Pair construction
# =============================================================================


def _cycle_pairs(n_samples):
    """All ``N(N-1)/2`` unique upper-triangle pairs as ``(rows, cols)`` int32 arrays."""
    rows, cols = np.triu_indices(n_samples, k=1)
    return rows.astype(np.int32, copy=False), cols.astype(np.int32, copy=False)


def _pivot_pairs(pivot_indices, n_samples):
    """Canonical ``(rows, cols)`` int32 arrays for all pairs ``(i, p)``.

    Pivot–pivot pairs would otherwise appear once per endpoint pivot. Pairs
    are canonicalized to ``(min, max)`` and deduplicated via int64 encoding.
    """
    all_idx = np.arange(n_samples, dtype=np.int32)
    rows_chunks, cols_chunks = [], []
    for p in pivot_indices:
        p_int = int(p)
        mask = all_idx != p_int
        i_arr = all_idx[mask]
        j_arr = np.full(int(mask.sum()), p_int, dtype=np.int32)
        rows_chunks.append(np.minimum(i_arr, j_arr))
        cols_chunks.append(np.maximum(i_arr, j_arr))
    rows_raw = np.concatenate(rows_chunks)
    cols_raw = np.concatenate(cols_chunks)
    encoded = rows_raw.astype(np.int64) * n_samples + cols_raw.astype(np.int64)
    unique_enc = np.unique(encoded)
    rows = (unique_enc // n_samples).astype(np.int32)
    cols = (unique_enc % n_samples).astype(np.int32)
    return rows, cols


# =============================================================================
# Weight bounds for the learning-rate schedule
# =============================================================================


def _weight_bounds_from_matrix(dissimilarity_matrix, weighting):
    """Exact ``(w_min, w_max, d_floor)`` from a full dissimilarity matrix."""
    if weighting == "uniform":
        return 1.0, 1.0, 1e-6

    upper = dissimilarity_matrix[np.triu_indices(dissimilarity_matrix.shape[0], k=1)]
    if upper.size == 0:
        return 1.0, 1.0, 1e-6

    d_min = float(np.min(upper))
    d_max = float(np.max(upper))
    floor_val = max(d_min, 1e-9)
    return 1.0 / (d_max ** 2), 1.0 / (floor_val ** 2), floor_val


def _estimate_weight_bounds(X, sample_pairs, weighting, rng, n_samples_est=1000):
    """Sample-based ``(w_min, w_max)`` for lazy mode (no full matrix)."""
    if weighting == "uniform":
        return 1.0, 1.0

    n_pairs = sample_pairs.shape[0]
    indices = rng.choice(n_pairs, size=min(n_pairs, n_samples_est), replace=False)
    pairs = sample_pairs[indices]
    dists = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    dists = dists[dists > 1e-9]
    if dists.size == 0:
        return 1.0, 1.0
    return 1.0 / (np.max(dists) ** 2), 1.0 / (np.min(dists) ** 2)


# =============================================================================
# Stress
# =============================================================================

# Exact stress evaluation (`_full_stress`) materialises a full N x N distance
# matrix -- O(N^2) memory and time, on top of whatever the fit itself used.
# EXACT_STRESS_N_CAP is a *safety guard*, not a technical limit: above this
# many samples, `_safe_stress` skips scoring and returns -1.0 rather than risk
# silently exhausting memory on a call site that did not expect it (e.g. an
# interactive `fit_transform` on a laptop). It is deliberately conservative.
#
# On a machine with enough RAM (a cluster node), raise it before fitting:
#     import sklearn.manifold._sgd_mds as _sgd_mds
#     sklearn.manifold._sgd_mds.EXACT_STRESS_N_CAP = 200_000
# Peak memory for exact scoring is ~16*N^2 bytes (two transient N x N float64
# matrices); for `dissimilarity='euclidean'` fits the distance matrix is also
# resident for the fit itself, on top of the ~28*N^2 bytes the cycle sampler
# needs for its explicit pair list (see run_experiment_k.py for a calibrated
# peak-memory table). This guard does not affect the lazy/budgeted fit path
# itself (O(N*D) memory) -- only whether its resulting stress can be scored.
EXACT_STRESS_N_CAP = 20_000


def _full_stress(embedding, dissimilarity_matrix):
    """Raw MDS stress: ``0.5 * sum_{i,j} (d_ij(emb) - delta_ij)^2``."""
    dis = euclidean_distances(embedding, embedding)
    return 0.5 * float(np.sum((dis - dissimilarity_matrix) ** 2))


# =============================================================================
# SGDMDS estimator
# =============================================================================


class SGDMDS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Multidimensional scaling via stochastic gradient descent.

    Minimizes the raw stress objective with SGD on pairwise distances.
    See module docstring for an overview of supported sampling strategies
    and dissimilarity modes.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.

    metric : bool, default=True
        If True, perform metric MDS; otherwise perform non-metric MDS using
        isotonic regression on the dissimilarity-to-distance map.

    n_init : int, default=1
        Number of independent SGD runs from different initializations. The
        final embedding is the run with the lowest stress.

    max_iter : int, default=30
        Number of SGD epochs per run.

    tol : float, default=1e-4
        Reserved for future use; currently ignored.

    learning_rate_init : float or 'auto', default='auto'
        Initial learning rate. With ``'auto'``, set to ``1 / w_min`` based on
        the weight bounds.

    batch_size : int or None, default=None
        Reserved for future use; currently ignored.

    dissimilarity : {'euclidean', 'precomputed', 'lazy'}, default='euclidean'
        - ``'euclidean'`` : input is feature data; pairwise distances are
          materialized once (``O(N^2)`` memory).
        - ``'precomputed'`` : input is a symmetric dissimilarity matrix.
        - ``'lazy'`` : input is feature data; distances are computed on the
          fly inside the SGD kernel (``O(N)`` extra memory).

    scheduler : {'exponential', 'harmonic', 'constant', 'hybrid'}, \
default='exponential'
        Learning-rate decay schedule.

    scheduler_switch_ratio : float, default=0.5
        Used by the ``'hybrid'`` scheduler. Fraction of ``max_iter`` at which
        the schedule switches from exponential to harmonic decay.

    epsilon : float, default=0.001
        Lower bound on the final learning rate, expressed as ``epsilon / w_max``.

    weighting : {'uniform', 'inverse'}, default='uniform'
        Per-pair weighting. ``'inverse'`` uses ``1 / d_ij^2`` (Sammon-style).

    sampling_strategy : {'cycle', 'random', 'pivot', 'hybrid'}, default='cycle'
        How pairs are visited each epoch. ``'random'`` and ``'hybrid'``
        require ``dissimilarity='lazy'``. ``'hybrid'`` does
        ``hybrid_alpha * k * n_samples`` pivot updates and the remainder
        as uniform random pair updates, holding the per-epoch budget equal
        to ``'pivot'``.

    n_pivots : int or 'auto', default='auto'
        Number of pivots used by ``sampling_strategy='pivot'`` and
        ``'hybrid'``. With ``'auto'``, resolved to
        ``min(50, max(n_components + 1, n_samples // 20))``.

    pivot_strategy : {'maxmin', 'maxmin_pca'}, default='maxmin'
        Pivot selection heuristic. ``'maxmin'`` runs the greedy farthest-point
        scan in the original feature (or distance) space. ``'maxmin_pca'``
        first projects features to ``pivot_pca_dim`` PCA components and runs
        MaxMin in that low-dimensional space — useful on high-dimensional
        data where Euclidean distances concentrate and degrade pivot spread.
        ``'maxmin_pca'`` requires ``dissimilarity='euclidean'`` or ``'lazy'``.
        The SGD update loop always uses dissimilarities from the original
        feature space; only pivot *selection* sees the PCA projection.

    pivot_pca_dim : int, default=30
        Target dimensionality for the PCA projection used by
        ``pivot_strategy='maxmin_pca'``. Capped at ``min(n_features, n_samples)``.
        Ignored when ``pivot_strategy='maxmin'``.

    hybrid_alpha : float in [0, 1], default=0.5
        Mixing weight for ``sampling_strategy='hybrid'``. ``alpha=1.0``
        reduces to pure pivot; ``alpha=0.0`` reduces to ``k*N`` random
        updates per epoch (lower budget than pure ``'random'``). Ignored
        for other sampling strategies.

    n_updates_per_epoch : int or 'auto', default='auto'
        Per-epoch update budget for ``sampling_strategy='random'``. With
        ``'auto'`` (the default), resolved to ``N(N-1)/2`` so that random
        sampling performs the same total update count as full-cycle SGD.
        Setting an integer enables a "fast mode": e.g. ``k * n_samples``
        for a small ``k`` performs far fewer updates per epoch at a
        controlled stress cost. Must be ``"auto"`` for any other
        ``sampling_strategy`` (otherwise a ``ValueError`` is raised).

    compute_stress : bool, default=True
        Whether to evaluate raw stress during and after the fit. Stress is
        purely observational -- no part of the optimization (scheduling,
        stopping) depends on it -- but evaluating it exactly costs O(N^2)
        time and memory (in lazy mode it materializes the full N x N
        distance matrix), which can dominate the fit itself for the budgeted
        fast mode. With ``False``, ``stress_`` is ``-1.0``,
        ``stress_history_`` carries ``-1.0`` placeholders, and the returned
        embedding is bit-identical to ``True`` (same seeds, same
        trajectory). Requires ``n_init=1`` (best-of-n selection needs
        stress). Intended for benchmarking at large N where stress is
        computed externally (e.g. chunked or sampled).

    n_jobs : int, default=None
        Reserved for future parallelization; currently ignored.

    random_state : int, RandomState instance or None, default=None
        Controls reproducibility of initialization, pair sampling, and pivot
        selection.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Embedded coordinates, recentered to zero mean.

    stress_ : float
        Final raw stress for the best run, or ``-1.0`` for very large
        ``n_samples`` (where stress evaluation is skipped).

    stress_history_ : list of (float, float)
        Per-epoch ``(elapsed_seconds, stress)`` pairs from the best run.
        Skipped for ``n_samples > 10000`` to bound logging overhead.

    n_iter_ : int
        Number of SGD epochs performed for the best run.

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples) or None
        Dissimilarity matrix used during ``fit``. ``None`` when
        ``dissimilarity='lazy'``.

    pivot_indices_ : ndarray of shape (n_pivots,), dtype int32
        Indices of selected pivots. Set only when
        ``sampling_strategy='pivot'``.

    n_features_in_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    MDS : SMACOF-based multidimensional scaling.

    References
    ----------
    .. [1] Zheng, J. X., Pawar, S., Goodman, D. F. M. (2018).
       *Graph drawing by stochastic gradient descent.* IEEE Transactions
       on Visualization and Computer Graphics 25(9): 2738-2748.
    .. [2] Brandes, U. & Pich, C. (2007). *Eigensolver methods for
       progressive multidimensional scaling of large data.* International
       Symposium on Graph Drawing, pp. 42-53.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "metric": ["boolean"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "learning_rate_init": [
            Interval(Real, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "dissimilarity": [StrOptions({"euclidean", "precomputed", "lazy"})],
        "scheduler": [
            StrOptions({"exponential", "harmonic", "constant", "hybrid"})
        ],
        "scheduler_switch_ratio": [Interval(Real, 0.0, 1.0, closed="both")],
        "epsilon": [Interval(Real, 0.0, 1.0, closed="both")],
        "weighting": [StrOptions({"uniform", "inverse"})],
        "sampling_strategy": [
            StrOptions({"cycle", "random", "pivot", "hybrid"})
        ],
        "n_pivots": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "pivot_strategy": [StrOptions({"maxmin", "maxmin_pca"})],
        "pivot_pca_dim": [Interval(Integral, 1, None, closed="left")],
        "hybrid_alpha": [Interval(Real, 0.0, 1.0, closed="both")],
        "n_updates_per_epoch": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "compute_stress": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=2,
        *,
        metric=True,
        n_init=1,
        max_iter=30,
        tol=1e-4,
        learning_rate_init="auto",
        batch_size=None,
        dissimilarity="euclidean",
        scheduler="exponential",
        scheduler_switch_ratio=0.5,
        epsilon=0.001,
        weighting="uniform",
        sampling_strategy="cycle",
        n_pivots="auto",
        pivot_strategy="maxmin",
        pivot_pca_dim=30,
        hybrid_alpha=0.5,
        n_updates_per_epoch="auto",
        compute_stress=True,
        n_jobs=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.dissimilarity = dissimilarity
        self.scheduler = scheduler
        self.scheduler_switch_ratio = scheduler_switch_ratio
        self.epsilon = epsilon
        self.weighting = weighting
        self.sampling_strategy = sampling_strategy
        self.n_pivots = n_pivots
        self.pivot_strategy = pivot_strategy
        self.pivot_pca_dim = pivot_pca_dim
        self.hybrid_alpha = hybrid_alpha
        self.n_updates_per_epoch = n_updates_per_epoch
        self.compute_stress = compute_stress
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.dissimilarity == "precomputed"
        return tags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Compute the embedding from data ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity='precomputed'``, this must be a
            symmetric dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, y)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Fit the model and return the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity='precomputed'``, this must be a
            symmetric dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Embedded coordinates.
        """
        if not HAS_CYTHON:
            raise RuntimeError(
                "Cython extension '_sgd_mds_cython' is not compiled. "
                "Cannot run SGDMDS."
            )

        if self.sampling_strategy == "random" and self.dissimilarity != "lazy":
            raise ValueError(
                "sampling_strategy='random' is only supported with "
                "dissimilarity='lazy'. Use 'cycle' or 'pivot' for "
                "precomputed/euclidean dissimilarities."
            )

        if self.sampling_strategy == "hybrid" and self.dissimilarity != "lazy":
            raise ValueError(
                "sampling_strategy='hybrid' is only supported with "
                "dissimilarity='lazy' (mixes the pivot and random lazy kernels)."
            )

        if (
            self.pivot_strategy == "maxmin_pca"
            and self.dissimilarity == "precomputed"
        ):
            raise ValueError(
                "pivot_strategy='maxmin_pca' requires dissimilarity='euclidean' "
                "or 'lazy' (features must be available for PCA)."
            )

        if (
            self.n_updates_per_epoch != "auto"
            and self.sampling_strategy != "random"
        ):
            raise ValueError(
                "n_updates_per_epoch is only configurable for "
                "sampling_strategy='random'; got "
                f"sampling_strategy={self.sampling_strategy!r}."
            )

        if not self.compute_stress and self.n_init > 1:
            raise ValueError(
                "compute_stress=False requires n_init=1: selecting the best "
                f"of n_init={self.n_init} runs needs their stress values."
            )

        solver_input = self._prepare_input(X)
        rng = check_random_state(self.random_state)

        best_stress = np.inf
        best_embedding = None
        best_n_iter = 0
        best_history = None
        best_pivots = None

        for _ in range(self.n_init):
            run_seed = rng.randint(np.iinfo(np.int32).max)
            embedding, stress, n_iter, history, pivots = self._solve_single_run(
                solver_input, run_seed
            )
            if stress < best_stress:
                best_stress = stress
                best_embedding = embedding
                best_n_iter = n_iter
                best_history = history
                best_pivots = pivots

        self.embedding_ = best_embedding
        self.stress_ = best_stress
        self.n_iter_ = best_n_iter
        self.stress_history_ = best_history
        if best_pivots is not None:
            self.pivot_indices_ = best_pivots

        return self.embedding_

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _prepare_input(self, X):
        """Validate ``X`` and return the array consumed by the solver.

        Sets ``self.dissimilarity_matrix_`` and ``self.n_features_in_``.

        Returns
        -------
        solver_input : ndarray
            - ``dissimilarity='precomputed'`` or ``'euclidean'``: full
              dissimilarity matrix.
            - ``dissimilarity='lazy'``: contiguous float64 feature matrix.
        """
        if self.dissimilarity == "precomputed":
            X = validate_data(self, X, dtype=np.float64, ensure_2d=True)
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "Precomputed dissimilarity matrix must be square."
                )
            self.dissimilarity_matrix_ = check_symmetric(X, raise_exception=True)
            self._features_for_pivots_ = None
            return self.dissimilarity_matrix_

        X = validate_data(self, X, dtype=np.float64, ensure_2d=True)

        if self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X, X)
            self._features_for_pivots_ = X
            return self.dissimilarity_matrix_

        # lazy
        self.dissimilarity_matrix_ = None
        contig_X = np.ascontiguousarray(X, dtype=np.float64)
        self._features_for_pivots_ = contig_X
        return contig_X

    # ------------------------------------------------------------------
    # Single-run dispatcher
    # ------------------------------------------------------------------

    def _solve_single_run(self, solver_input, random_state):
        """Run SGD for one initialization. Dispatches by dissimilarity mode.

        Returns
        -------
        embedding, stress, n_iter, history, pivot_indices
        """
        rng = check_random_state(random_state)
        n_samples = solver_input.shape[0]

        embedding = rng.standard_normal(size=(n_samples, self.n_components))
        embedding = np.ascontiguousarray(embedding, dtype=np.float64)

        pivot_indices = None
        if self.sampling_strategy in ("pivot", "hybrid"):
            n_pivots = _resolve_n_pivots(
                self.n_pivots, self.n_components, n_samples
            )
            if self.pivot_strategy == "maxmin_pca":
                features = self._features_for_pivots_
                features_proj = _maybe_pca_project(
                    features, self.pivot_pca_dim, rng
                )
                pivot_indices = _select_pivots_maxmin(
                    features_proj, n_pivots, rng, is_lazy=True,
                )
            else:
                pivot_indices = _select_pivots_maxmin(
                    solver_input, n_pivots, rng,
                    is_lazy=(self.dissimilarity == "lazy"),
                )

        if self.dissimilarity == "lazy":
            history = self._run_lazy(solver_input, embedding, pivot_indices, rng)
        else:
            history = self._run_matrix(solver_input, embedding, pivot_indices, rng)

        embedding -= np.mean(embedding, axis=0)
        stress = self._safe_stress(embedding, solver_input)
        return embedding, stress, self.max_iter, history, pivot_indices

    # ------------------------------------------------------------------
    # Matrix mode (precomputed or euclidean)
    # ------------------------------------------------------------------

    def _run_matrix(self, dissimilarity_matrix, embedding, pivot_indices, rng):
        """SGD epochs against a materialized dissimilarity matrix.

        Supports ``cycle`` and ``pivot`` sampling. Pre-extracts pair distances
        and weights once, then shuffles + dispatches to the Cython kernel
        each epoch.
        """
        n_samples = embedding.shape[0]

        if self.sampling_strategy == "cycle":
            rows, cols = _cycle_pairs(n_samples)
        else:  # pivot (validated upstream)
            rows, cols = _pivot_pairs(pivot_indices, n_samples)

        pairs_canonical = np.column_stack((rows, cols)).astype(np.int32)
        flat_distances = dissimilarity_matrix[rows, cols].astype(np.float64)

        w_min, w_max, d_floor = _weight_bounds_from_matrix(
            dissimilarity_matrix, self.weighting
        )
        if self.weighting == "inverse":
            clipped = np.maximum(flat_distances, d_floor)
            flat_weights = 1.0 / (clipped ** 2)
        else:
            flat_weights = np.ones(pairs_canonical.shape[0], dtype=np.float64)

        scheduler = self._make_scheduler(w_min, w_max)

        # Non-metric setup: pre-sort distances on non-zero entries.
        ir = sorted_valid_idx = x_sorted = None
        if not self.metric:
            ir, sorted_valid_idx, x_sorted = self._prepare_isotonic_matrix(
                flat_distances
            )

        history = [(0.0, self._safe_stress(embedding, dissimilarity_matrix))]
        cumulative_time = 0.0

        for epoch in range(self.max_iter):
            t0 = time.time()
            lr = scheduler.get_rate(epoch)

            if self.metric:
                target_canonical = flat_distances
            else:
                target_canonical = self._isotonic_targets_matrix(
                    embedding, rows, cols, flat_distances,
                    ir, sorted_valid_idx, x_sorted, epoch,
                )

            pairs_epoch, target_epoch, weights_epoch = sklearn_shuffle(
                pairs_canonical, target_canonical, flat_weights, random_state=rng
            )
            run_sgd_epoch(embedding, target_epoch, weights_epoch, pairs_epoch, lr)

            cumulative_time += time.time() - t0
            if self.compute_stress and n_samples <= 10000:
                history.append(
                    (cumulative_time, _full_stress(embedding, dissimilarity_matrix))
                )
            elif not self.compute_stress:
                history.append((cumulative_time, -1.0))
        return history

    @staticmethod
    def _prepare_isotonic_matrix(flat_distances):
        """Common setup for non-metric MDS in matrix mode."""
        valid_idx = np.flatnonzero(flat_distances != 0.0)
        if valid_idx.size == 0:
            raise ValueError(
                "Non-metric MDS: all dissimilarities are 0 (treated as missing)."
            )
        order = np.argsort(flat_distances[valid_idx], kind="mergesort")
        sorted_valid_idx = valid_idx[order]
        x_sorted = np.ascontiguousarray(
            flat_distances[sorted_valid_idx], dtype=np.float64
        )
        ir = IsotonicRegression(out_of_bounds="clip")
        return ir, sorted_valid_idx, x_sorted

    @staticmethod
    def _isotonic_targets_matrix(
        embedding, rows, cols, flat_distances,
        ir, sorted_valid_idx, x_sorted, epoch,
    ):
        """Compute per-pair disparity targets for non-metric MDS (matrix mode)."""
        diffs = embedding[rows] - embedding[cols]
        flat_embed_dist = np.sqrt((diffs * diffs).sum(axis=1))
        y_sorted = flat_embed_dist[sorted_valid_idx]

        if epoch < 1:
            disparities_sorted = x_sorted.copy()
        else:
            disparities_sorted = ir.fit_transform(x_sorted, y_sorted)

        targets = np.zeros_like(flat_distances)
        targets[sorted_valid_idx] = disparities_sorted
        denom = float(np.sum(targets * targets))
        if denom > 0.0:
            targets *= np.sqrt(targets.shape[0] / denom)
        return targets

    # ------------------------------------------------------------------
    # Lazy mode
    # ------------------------------------------------------------------

    def _run_lazy(self, X, embedding, pivot_indices, rng):
        """SGD epochs without materializing the dissimilarity matrix.

        Distances are computed on the fly inside the Cython kernel. Supports
        ``random``, ``pivot``, and ``hybrid`` sampling.
        """
        n_samples = X.shape[0]
        n_pairs = _resolve_n_updates_per_epoch(
            self.n_updates_per_epoch, n_samples
        )
        is_hybrid = self.sampling_strategy == "hybrid"
        is_pivot = (pivot_indices is not None) and not is_hybrid

        sample_pairs = self._lazy_sample_pairs(n_samples, pivot_indices, rng)
        w_min, w_max = _estimate_weight_bounds(X, sample_pairs, self.weighting, rng)

        scheduler = self._make_scheduler(w_min, w_max)
        weighting_code = 1 if self.weighting == "inverse" else 0

        history = [(0.0, self._safe_stress(embedding, X))]
        cumulative_time = 0.0

        # Hybrid budget: total updates per epoch == pure-pivot budget (k * N),
        # split alpha:(1-alpha) between the pivot and random kernels.
        if is_hybrid:
            total_updates = int(pivot_indices.shape[0]) * n_samples
            n_pivot_updates = int(round(self.hybrid_alpha * total_updates))
            n_random_updates = total_updates - n_pivot_updates

        for epoch in range(self.max_iter):
            t0 = time.time()
            lr = scheduler.get_rate(epoch)

            x_knots = y_knots = None
            if not self.metric:
                x_knots, y_knots = self._fit_monotonic_map(embedding, X)

            seed_epoch = rng.randint(1, 2**31 - 1)

            if is_hybrid:
                if n_pivot_updates > 0:
                    run_sgd_epoch_pivot_lazy(
                        embedding, X, pivot_indices, n_pivot_updates, lr,
                        weighting_code, seed_epoch,
                        x_knots=x_knots, y_knots=y_knots,
                    )
                if n_random_updates > 0:
                    seed_random = rng.randint(1, 2**31 - 1)
                    run_sgd_epoch_lazy_random_native(
                        embedding, X, n_random_updates, lr,
                        weighting_code, seed_random,
                        x_knots=x_knots, y_knots=y_knots,
                    )
            elif is_pivot:
                n_updates = int(pivot_indices.shape[0]) * n_samples
                run_sgd_epoch_pivot_lazy(
                    embedding, X, pivot_indices, n_updates, lr,
                    weighting_code, seed_epoch,
                    x_knots=x_knots, y_knots=y_knots,
                )
            else:
                run_sgd_epoch_lazy_random_native(
                    embedding, X, n_pairs, lr, weighting_code, seed_epoch,
                    x_knots=x_knots, y_knots=y_knots,
                )

            cumulative_time += time.time() - t0
            if self.compute_stress and n_samples <= 10000:
                D_temp = euclidean_distances(X, X)
                history.append((cumulative_time, _full_stress(embedding, D_temp)))
            elif not self.compute_stress:
                history.append((cumulative_time, -1.0))
        return history

    @staticmethod
    def _lazy_sample_pairs(n_samples, pivot_indices, rng, size=1000):
        """Draw a random sample of pairs for weight-bound estimation."""
        if pivot_indices is not None:
            k = pivot_indices.shape[0]
            size = min(size, k * n_samples)
            rand_i = rng.randint(0, n_samples, size=size)
            rand_p = rng.randint(0, k, size=size)
            rand_j = pivot_indices[rand_p].astype(np.int32, copy=False)
            mask = rand_i != rand_j
            return np.column_stack(
                (rand_i[mask].astype(np.int32, copy=False), rand_j[mask])
            )
        rand_i = rng.randint(0, n_samples, size=size)
        rand_j = rng.randint(0, n_samples, size=size)
        mask = rand_i != rand_j
        return np.column_stack((rand_i[mask], rand_j[mask])).astype(np.int32)

    def _fit_monotonic_map(self, embedding, X, n_samples_isotonic=2000):
        """Fit isotonic map ``f(delta) -> disparity`` on a random pair sample.

        Returns ``(x_knots, y_knots)`` as contiguous float64 arrays, scaled
        to preserve the disparity normalization used by SMACOF.
        """
        n_points = embedding.shape[0]
        rng = check_random_state(self.random_state)

        idx_i = rng.randint(0, n_points, size=n_samples_isotonic)
        idx_j = rng.randint(0, n_points, size=n_samples_isotonic)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        dist_vec = np.linalg.norm(embedding[idx_i] - embedding[idx_j], axis=1)
        delta_vec = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        disparity = ir.fit_transform(delta_vec, dist_vec)

        x_knots = np.ascontiguousarray(ir.f_.x, dtype=np.float64)
        y_knots = np.ascontiguousarray(ir.f_.y, dtype=np.float64)

        denom = float(np.sum(disparity ** 2))
        if denom > 0:
            y_knots *= np.sqrt(len(disparity) / denom)
        return x_knots, y_knots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_scheduler(self, w_min, w_max):
        """Build the learning-rate scheduler for the current run."""
        lr_init = (
            (1.0 / w_min)
            if self.learning_rate_init == "auto"
            else float(self.learning_rate_init)
        )
        return _make_scheduler(
            self.scheduler,
            lr_init=lr_init,
            w_min=w_min,
            w_max=w_max,
            max_iter=self.max_iter,
            switch_ratio=self.scheduler_switch_ratio,
            epsilon=self.epsilon,
        )

    def _safe_stress(self, embedding, solver_input):
        """Compute stress, skipping for very large ``n_samples``.

        Returns ``-1.0`` without computing anything when
        ``compute_stress=False`` (stress is observational only -- see the
        parameter docs). Otherwise the threshold is the module-level
        ``EXACT_STRESS_N_CAP`` (default 20,000) -- a memory/time safety
        guard, not a hard technical limit. Raise it
        (``sklearn.manifold._sgd_mds.EXACT_STRESS_N_CAP = ...``) on a
        machine with enough RAM to score larger fits exactly.
        """
        if not self.compute_stress:
            return -1.0
        n_samples = embedding.shape[0]
        if n_samples > EXACT_STRESS_N_CAP:
            return -1.0
        if self.dissimilarity == "lazy":
            return _full_stress(embedding, euclidean_distances(solver_input, solver_input))
        return _full_stress(embedding, solver_input)
