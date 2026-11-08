"""
Multi-dimensional scaling by stochastic gradient descent.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold._sgd_mds_cython import (
    run_sgd_epoch,
    run_sgd_epoch_blocked,
    run_sgd_epoch_random,
)
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.utils import check_random_state, check_symmetric, shuffle
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data

# Above this many samples the final stress is not evaluated: it needs two
# full distance matrices, and `stress_` is set to -1.0 instead.
EXACT_STRESS_N_CAP = 20_000


class _BaseScheduler:
    """Base class for the learning rate schedules of :class:`SGDMDS`."""

    def __init__(self, lr_init, w_max, max_iter, epsilon):
        self.lr_init = float(lr_init)
        self.w_max = float(w_max)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)

    def get_rate(self, t):
        raise NotImplementedError


class _ConstantScheduler(_BaseScheduler):
    def get_rate(self, t):
        return self.lr_init


class _ExponentialScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_max, max_iter, epsilon):
        super().__init__(lr_init, w_max, max_iter, epsilon)
        lr_final = self.epsilon / self.w_max
        if self.lr_init > lr_final:
            self.decay = -np.log(lr_final / self.lr_init) / self.max_iter
        else:
            self.decay = 0.0

    def get_rate(self, t):
        return self.lr_init * np.exp(-self.decay * t)


class _HarmonicScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_max, max_iter, epsilon):
        super().__init__(lr_init, w_max, max_iter, epsilon)
        lr_final = self.epsilon / self.w_max
        if self.lr_init <= lr_final:
            self.decay = 0.0
        else:
            ratio = self.lr_init / lr_final
            self.decay = (ratio - 1.0) / self.max_iter

    def get_rate(self, t):
        return self.lr_init / (1.0 + self.decay * t)


class _HybridScheduler(_BaseScheduler):
    """Exponential decay followed by harmonic decay."""

    def __init__(self, lr_init, w_max, max_iter, epsilon, switch_ratio):
        super().__init__(lr_init, w_max, max_iter, epsilon)
        self.tau = int(max_iter * switch_ratio)

        lr_mid = self.lr_init * 0.1
        if self.tau > 0:
            self.decay_exp = -np.log(lr_mid / self.lr_init) / self.tau
        else:
            self.decay_exp = 0.0
            lr_mid = self.lr_init

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
}


def _weight_bounds_from_matrix(dissimilarities, weighting):
    """Return `(w_min, w_max, d_floor)` of a dissimilarity matrix."""
    if weighting == "uniform":
        return 1.0, 1.0, 1e-6

    upper = dissimilarities[np.triu_indices(dissimilarities.shape[0], k=1)]
    if upper.size == 0:
        return 1.0, 1.0, 1e-6

    d_min = float(np.min(upper))
    d_max = float(np.max(upper))
    floor_val = max(d_min, 1e-9)
    return 1.0 / (d_max**2), 1.0 / (floor_val**2), floor_val


def _estimate_weight_bounds(X, weighting, rng, n_pairs=1000):
    """Estimate `(w_min, w_max)` from a sample of pairs of points."""
    n_samples = X.shape[0]
    rows = rng.randint(0, n_samples, size=n_pairs)
    cols = rng.randint(0, n_samples, size=n_pairs)
    mask = rows != cols
    pairs = np.column_stack((rows[mask], cols[mask])).astype(np.int32)

    if weighting == "uniform":
        return 1.0, 1.0

    n_valid = pairs.shape[0]
    pairs = pairs[rng.choice(n_valid, size=min(n_valid, n_pairs), replace=False)]
    distances = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    distances = distances[distances > 1e-9]
    if distances.size == 0:
        return 1.0, 1.0
    return 1.0 / (np.max(distances) ** 2), 1.0 / (np.min(distances) ** 2)


def _stress(embedding, dissimilarities):
    """Return the raw stress of an embedding."""
    distances = euclidean_distances(embedding)
    return 0.5 * float(np.sum((distances - dissimilarities) ** 2))


class SGDMDS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Multidimensional scaling by stochastic gradient descent.

    Minimizes raw stress with stochastic gradient descent over pairs of points
    [1]_, instead of the batch majorization of :class:`MDS`. It converges in
    tens of epochs and, with `sampling_strategy='random'`, does not need the
    dissimilarity matrix, so it scales to large numbers of samples.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.

    metric_mds : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.

    n_init : int, default=1
        Number of times the algorithm will be run with different
        initializations. The final result is the run with the smallest final
        stress.

    max_iter : int, default=30
        Number of epochs of a single run.

    learning_rate_init : float or 'auto', default=0.5
        Initial learning rate. If `'auto'`, it is set to the inverse of the
        smallest pair weight.

    scheduler : {'exponential', 'harmonic', 'constant', 'hybrid'}, \
            default='hybrid'
        Learning rate schedule. `'hybrid'` decays exponentially and then
        harmonically.

    scheduler_switch_ratio : float, default=0.5
        Fraction of `max_iter` at which `scheduler='hybrid'` switches from
        exponential to harmonic decay.

    epsilon : float, default=0.001
        The learning rate decays to `epsilon / w_max`, where `w_max` is the
        largest pair weight. Ignored by `scheduler='constant'`.

    weighting : {'uniform', 'inverse'}, default='uniform'
        Pair weights. `'inverse'` uses the inverse squared dissimilarity, as in
        Sammon mapping.

    sampling_strategy : {'cycle', 'random'}, default='cycle'
        The pairs visited in each epoch:

        - 'cycle': every pair, in a new random order. The dissimilarity matrix
          is stored, using memory quadratic in `n_samples`.
        - 'random': pairs drawn uniformly at random. Dissimilarities are
          computed when needed, so memory is linear in `n_samples`. Only
          available with `metric='euclidean'`.

    n_updates_per_epoch : int or 'auto', default='auto'
        Number of pair updates per epoch with `sampling_strategy='random'`.
        If `'auto'`, it is `n_samples * (n_samples - 1) / 2`, the number of
        pairs. Setting it to `k * n_samples` with a fixed `k`, such as 200,
        makes the cost of an epoch linear in `n_samples`.

    block_size : int, default=16
        Number of points drawn at a time with `sampling_strategy='random'`.
        Every pair of a block is updated, and the dissimilarities are computed
        from one read of the rows of its points [2]_, which is faster when
        `n_features` is large. Budgets are counted in pairs whatever the
        value. Capped at `n_samples`. With 2, pairs are drawn independently.

    compute_stress : bool, default=True
        Whether to evaluate the final stress, which takes time and memory
        quadratic in `n_samples`. If ``False``, `stress_` is -1.0 and `n_init`
        must be 1. The embedding does not depend on this parameter.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used for the initialization and
        for sampling pairs. Pass an int for reproducible results across
        multiple function calls. See :term:`Glossary <random_state>`.

    metric : str or callable, default='euclidean'
        Metric to use for dissimilarity computation.

        If metric is a string, it must be one of the options allowed by
        `scipy.spatial.distance.pdist` for its metric parameter, or a metric
        listed in :func:`sklearn.metrics.pairwise.distance_metrics`.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors.

    metric_params : dict, default=None
        Additional keyword arguments for the dissimilarity computation.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final raw stress. It is -1.0 if `compute_stress=False`, or if
        `n_samples` is too large to evaluate it.

    n_iter_ : int
        The number of epochs run.

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Only available with
        `sampling_strategy='cycle'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    MDS : Metric and non-metric MDS by majorization.
    ClassicalMDS : Classical MDS.
    TSNE : T-distributed Stochastic Neighbor Embedding.

    References
    ----------
    .. [1] Zheng, J. X., Pawar, S., Goodman, D. F. M. "Graph drawing by
       stochastic gradient descent." IEEE Transactions on Visualization and
       Computer Graphics, 25(9), (2019)

    .. [2] Lambert, P., Bibal, A., Dumas, M., Frenay, B., Lee, J. A. "Squadmds:
       A lean stochastic quartet MDS improving global structure preservation in
       neighbor embedding like t-SNE and UMAP." Neurocomputing, 503, (2022)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold._sgd_mds import SGDMDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = SGDMDS(n_components=2, random_state=0)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "metric_mds": ["boolean"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "learning_rate_init": [
            Interval(Real, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "scheduler": [StrOptions({"exponential", "harmonic", "constant", "hybrid"})],
        "scheduler_switch_ratio": [Interval(Real, 0.0, 1.0, closed="both")],
        "epsilon": [Interval(Real, 0.0, 1.0, closed="both")],
        "weighting": [StrOptions({"uniform", "inverse"})],
        "sampling_strategy": [StrOptions({"cycle", "random"})],
        "n_updates_per_epoch": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "block_size": [Interval(Integral, 2, None, closed="left")],
        "compute_stress": ["boolean"],
        "random_state": ["random_state"],
        "metric": [str, callable],
        "metric_params": [dict, None],
    }

    # Called as `hook(n_epochs_done, embedding)` at the start of a run and after
    # each epoch. Benchmarks use it to record the embedding during a fit.
    _epoch_hook = None

    def __init__(
        self,
        n_components=2,
        *,
        metric_mds=True,
        n_init=1,
        max_iter=30,
        learning_rate_init=0.5,
        scheduler="hybrid",
        scheduler_switch_ratio=0.5,
        epsilon=0.001,
        weighting="uniform",
        sampling_strategy="cycle",
        n_updates_per_epoch="auto",
        block_size=16,
        compute_stress=True,
        random_state=None,
        metric="euclidean",
        metric_params=None,
    ):
        self.n_components = n_components
        self.metric_mds = metric_mds
        self.n_init = n_init
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.scheduler = scheduler
        self.scheduler_switch_ratio = scheduler_switch_ratio
        self.epsilon = epsilon
        self.weighting = weighting
        self.sampling_strategy = sampling_strategy
        self.n_updates_per_epoch = n_updates_per_epoch
        self.block_size = block_size
        self.compute_stress = compute_stress
        self.random_state = random_state
        self.metric = metric
        self.metric_params = metric_params

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags

    def fit(self, X, y=None):
        """
        Compute the position of the points in the embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``metric=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``metric=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        if self.sampling_strategy == "random" and self.metric != "euclidean":
            raise ValueError(
                "sampling_strategy='random' only supports metric='euclidean'; "
                f"got metric={self.metric!r}."
            )
        if self.n_updates_per_epoch != "auto" and self.sampling_strategy != "random":
            raise ValueError(
                "n_updates_per_epoch can only be set with "
                f"sampling_strategy='random'; got {self.sampling_strategy!r}."
            )
        if not self.compute_stress and self.n_init > 1:
            raise ValueError(
                "compute_stress=False requires n_init=1: selecting the best of "
                f"n_init={self.n_init} runs needs their stress."
            )

        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.metric != "precomputed":
            warnings.warn(
                "The provided input is a square matrix. Note that ``fit`` constructs "
                "a dissimilarity matrix from data and will treat rows as samples "
                "and columns as features. To use a pre-computed dissimilarity matrix, "
                "set ``metric='precomputed'``."
            )

        if self.sampling_strategy == "random":
            data = np.ascontiguousarray(X, dtype=np.float64)
        else:
            if self.metric == "precomputed":
                data = check_symmetric(X, raise_exception=True)
            else:
                data = pairwise_distances(
                    X,
                    metric=self.metric,
                    **(self.metric_params if self.metric_params is not None else {}),
                )
            self.dissimilarity_matrix_ = data

        random_state = check_random_state(self.random_state)
        best_embedding, best_stress = None, np.inf
        for _ in range(self.n_init):
            seed = random_state.randint(np.iinfo(np.int32).max)
            embedding, stress = self._fit_single(data, seed)
            if stress < best_stress:
                best_embedding, best_stress = embedding, stress

        self.embedding_ = best_embedding
        self.stress_ = best_stress
        self.n_iter_ = self.max_iter
        self._n_features_out = self.embedding_.shape[1]
        return self.embedding_

    def _fit_single(self, data, seed):
        """Run the optimization once and return the embedding and its stress."""
        rng = check_random_state(seed)
        embedding = rng.standard_normal(size=(data.shape[0], self.n_components))
        embedding = np.ascontiguousarray(embedding, dtype=np.float64)

        if self.sampling_strategy == "random":
            self._run_random(data, embedding, rng)
        else:
            self._run_cycle(data, embedding, rng)
        embedding -= np.mean(embedding, axis=0)

        if not self.compute_stress or embedding.shape[0] > EXACT_STRESS_N_CAP:
            return embedding, -1.0
        if self.sampling_strategy == "random":
            data = euclidean_distances(data)
        return embedding, _stress(embedding, data)

    def _make_scheduler(self, w_min, w_max):
        lr_init = self.learning_rate_init
        if lr_init == "auto":
            lr_init = 1.0 / w_min
        if self.scheduler == "hybrid":
            return _HybridScheduler(
                lr_init,
                w_max,
                self.max_iter,
                self.epsilon,
                self.scheduler_switch_ratio,
            )
        return _SCHEDULERS[self.scheduler](
            lr_init, w_max, self.max_iter, self.epsilon
        )

    def _run_cycle(self, dissimilarities, embedding, rng):
        """Run the epochs over all pairs of a dissimilarity matrix."""
        rows, cols = np.triu_indices(embedding.shape[0], k=1)
        rows = rows.astype(np.int32, copy=False)
        cols = cols.astype(np.int32, copy=False)
        pairs = np.column_stack((rows, cols)).astype(np.int32)
        distances = dissimilarities[rows, cols].astype(np.float64)

        w_min, w_max, d_floor = _weight_bounds_from_matrix(
            dissimilarities, self.weighting
        )
        if self.weighting == "inverse":
            weights = 1.0 / (np.maximum(distances, d_floor) ** 2)
        else:
            weights = np.ones(pairs.shape[0], dtype=np.float64)

        scheduler = self._make_scheduler(w_min, w_max)
        if not self.metric_mds:
            isotonic = self._prepare_isotonic(distances)

        if self._epoch_hook is not None:
            self._epoch_hook(0, embedding)

        for epoch in range(self.max_iter):
            lr = scheduler.get_rate(epoch)
            if self.metric_mds:
                targets = distances
            else:
                targets = self._isotonic_targets(
                    embedding, rows, cols, distances, epoch, *isotonic
                )
            pairs_epoch, targets_epoch, weights_epoch = shuffle(
                pairs, targets, weights, random_state=rng
            )
            run_sgd_epoch(embedding, targets_epoch, weights_epoch, pairs_epoch, lr)

            if self._epoch_hook is not None:
                self._epoch_hook(epoch + 1, embedding)

    @staticmethod
    def _prepare_isotonic(distances):
        """Sort the nonzero dissimilarities for non-metric MDS."""
        valid = np.flatnonzero(distances != 0.0)
        if valid.size == 0:
            raise ValueError(
                "Non-metric MDS: all dissimilarities are 0 (treated as missing)."
            )
        order = valid[np.argsort(distances[valid], kind="mergesort")]
        x_sorted = np.ascontiguousarray(distances[order], dtype=np.float64)
        return IsotonicRegression(out_of_bounds="clip"), order, x_sorted

    @staticmethod
    def _isotonic_targets(embedding, rows, cols, distances, epoch, ir, order, x_sorted):
        """Compute the disparities of all pairs for non-metric MDS."""
        diffs = embedding[rows] - embedding[cols]
        y_sorted = np.sqrt((diffs * diffs).sum(axis=1))[order]

        if epoch < 1:
            disparities = x_sorted.copy()
        else:
            disparities = ir.fit_transform(x_sorted, y_sorted)

        targets = np.zeros_like(distances)
        targets[order] = disparities
        norm = float(np.sum(targets * targets))
        if norm > 0.0:
            targets *= np.sqrt(targets.shape[0] / norm)
        return targets

    def _run_random(self, X, embedding, rng):
        """Run the epochs over random pairs, computing distances on the fly."""
        n_samples = X.shape[0]
        n_updates = self.n_updates_per_epoch
        if n_updates == "auto":
            n_updates = (n_samples * (n_samples - 1)) // 2
        block_size = min(self.block_size, n_samples)

        w_min, w_max = _estimate_weight_bounds(X, self.weighting, rng)
        scheduler = self._make_scheduler(w_min, w_max)
        weighting_code = 1 if self.weighting == "inverse" else 0

        if self._epoch_hook is not None:
            self._epoch_hook(0, embedding)

        for epoch in range(self.max_iter):
            lr = scheduler.get_rate(epoch)
            x_knots = y_knots = None
            if not self.metric_mds:
                x_knots, y_knots = self._fit_monotonic_map(embedding, X)

            seed = rng.randint(1, 2**31 - 1)
            if block_size == 2:
                run_sgd_epoch_random(
                    embedding, X, n_updates, lr, weighting_code, seed,
                    x_knots=x_knots, y_knots=y_knots,
                )  # fmt: skip
            else:
                run_sgd_epoch_blocked(
                    embedding, X, n_updates, block_size, lr, weighting_code, seed,
                    x_knots=x_knots, y_knots=y_knots,
                )  # fmt: skip

            if self._epoch_hook is not None:
                self._epoch_hook(epoch + 1, embedding)

    def _fit_monotonic_map(self, embedding, X, n_pairs=2000):
        """Fit the isotonic map of non-metric MDS on a sample of pairs."""
        n_samples = embedding.shape[0]
        rng = check_random_state(self.random_state)
        rows = rng.randint(0, n_samples, size=n_pairs)
        cols = rng.randint(0, n_samples, size=n_pairs)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        distances = np.linalg.norm(embedding[rows] - embedding[cols], axis=1)
        dissimilarities = np.linalg.norm(X[rows] - X[cols], axis=1)

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        disparities = ir.fit_transform(dissimilarities, distances)

        x_knots = np.ascontiguousarray(ir.f_.x, dtype=np.float64)
        y_knots = np.ascontiguousarray(ir.f_.y, dtype=np.float64)

        norm = float(np.sum(disparities**2))
        if norm > 0:
            y_knots *= np.sqrt(len(disparities) / norm)
        return x_knots, y_knots
