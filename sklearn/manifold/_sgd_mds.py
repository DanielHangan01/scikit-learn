"""
Stochastic Gradient Descent Multidimensional Scaling (SGD-MDS)
"""
# Author: Daniel Hangan
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numbers import Integral, Real

from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
from ..utils import check_random_state
from ..utils.validation import check_array
from ..utils._param_validation import Interval, StrOptions
from sklearn.metrics import euclidean_distances
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.utils.validation import check_symmetric
from sklearn.isotonic import IsotonicRegression
import time

try:
    from ._sgd_mds_cython import (
        run_sgd_epoch,
        run_sgd_epoch_lazy_random_native,
        run_sgd_epoch_pivot_lazy,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class _BaseScheduler:
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        self.lr_init = float(lr_init)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)

    def get_rate(self, t):
        raise NotImplementedError()

class _ConstantScheduler(_BaseScheduler):
    def get_rate(self, t):
        return self.lr_init

class _ExponentialScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        self.lr_final = self.epsilon / self.w_max
        if self.lr_init > self.lr_final:
            self.decay = -np.log(self.lr_final / self.lr_init) / self.max_iter
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
    """
    Combines Exponential decay (fast unfolding) with Harmonic decay (fine convergence).
    Forces the switch to happen at 'switch_ratio' of max_iter regardless of weights.
    """
    def __init__(self, lr_init, w_min, w_max, max_iter, switch_ratio=0.4, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        
        self.tau = int(max_iter * switch_ratio)
        
        # --- Phase 1: Exponential Parameters ---
        # Target an intermediate rate 'lr_mid' at the switch point.
        # Decay to 10% of init rate allows significant early movement.
        lr_mid = self.lr_init * 0.1
        
        if self.tau > 0:
            self.decay_exp = -np.log(lr_mid / self.lr_init) / self.tau
        else:
            self.decay_exp = 0.0
            lr_mid = self.lr_init

        # --- Phase 2: Harmonic (1/t) Parameters ---
        # Function: f(t) = a / (1 + b*(t - tau))
        # Match value at switch: f(tau) = lr_mid  => a = lr_mid
        # Match value at end:    f(max_iter) = lr_final
        
        self.a = lr_mid
        lr_final = self.epsilon / self.w_max
        
        remaining_iter = max_iter - self.tau
        if remaining_iter > 0 and lr_mid > lr_final:
            # Solve for b: lr_mid / (1 + b * remaining) = lr_final
            self.b = (lr_mid / lr_final - 1) / remaining_iter
        else:
            self.b = 0.0

    def get_rate(self, t):
        if t < self.tau:
            # Exponential Phase: Fast exploration
            return self.lr_init * np.exp(-self.decay_exp * t)
        else:
            # Harmonic Phase: Guaranteed convergence (1/t)
            return self.a / (1.0 + self.b * (t - self.tau))

class SGDMDS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """
    Multidimensional Scaling (MDS) using Stochastic Gradient Descent.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimenions in which to immerse the dissimilarities.

    metric : bool, default=True
        If True, perform metric MDS; otherwise, perform nonmetric MDS.

    n_init : int, default=1
        Number of times the algorithm will be run with different initializations.
        The final results will be the best output of the runs in terms of stress.

    max_iter : int, default=300
        Maximum number of iterations of the algorithm for a single run.

    tol : float, default=1e-4
        Tolerance for stopping criterion.

    learning_rate_init : float or 'auto', default='auto'
        The initial learning rate for the SGD updates.

    scheduler : {'exponential', 'harmonic', 'constant', 'hybrid'}, default='exponential'
        The decay schedule for the learning rate.
    
    scheduler_switch_ratio : float, default=0.4
        Only used for 'hybrid' scheduler. The fraction of max_iter at which 
        the scheduler switches from exponential decay (exploration) to 
        harmonic decay (refinement).

    batch_size : int or None, default=None
        Number of pairs to sample per step. If None, defaults to min(50_000, n_pairs)

    dissimililarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:
        - 'euclidean': pairwise Euclidean distances between points in X.
        - 'precomputed': X is assumed to be a dissimilarity matrix.
    
    n_jobs : int, default=None
        The number of jobs to use for the computation.
        (Placeholder for future parallelization)

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
    """

    # Parameter constraints for automatic validation (sklearn >= 1.2)
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "metric": ["boolean"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "learning_rate_init": [Interval(Real, 0, None, closed="left"), StrOptions({"auto"})],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "dissimilarity": [StrOptions({"euclidean", "precomputed", "lazy"})],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "algorithm": [StrOptions({"mini_batch", "true_sgd"})],
        "scheduler": [StrOptions({"exponential", "harmonic", "constant", "hybrid"})],
        "scheduler_switch_ratio": [Interval(Real, 0.0, 1.0, closed="both")],
        "epsilon": [Interval(Real, 0.0, 1.0, closed="both")],
        "weighting": [StrOptions({"uniform", "inverse"})],
        "sampling_strategy": [StrOptions({"cycle", "random", "pivot"})],
        "n_pivots": [Interval(Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "pivot_strategy": [StrOptions({"maxmin"})],
    }

    def __init__(self, n_components=2, *, metric=True, n_init=1,
                 max_iter=30, tol=1e-4, learning_rate_init="auto",
                 batch_size=None, dissimilarity="euclidean",
                 scheduler="exponential", scheduler_switch_ratio=0.5, epsilon=0.001,
                 weighting="uniform", sampling_strategy="cycle",
                 n_pivots="auto", pivot_strategy="maxmin",
                 n_jobs=None, random_state=None):
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
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Compute the position of the points in the embeddings space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        
        Y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator. 
        """
        self._validate_params()

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            if self.dissimilarity_matrix_.shape[0] != self.dissimilarity_matrix_.shape[1]:
                raise ValueError("Precomputed dissimilarity matrix must be square.")
            check_symmetric(self.dissimilarity_matrix_, raise_exception=True)
            
            solver_input = self.dissimilarity_matrix_

        elif self.dissimilarity == "euclidean":
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            # Store full matrix (O(N^2) memory)
            self.dissimilarity_matrix_ = euclidean_distances(X, X)
            
            solver_input = self.dissimilarity_matrix_

        elif self.dissimilarity == "lazy":
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            X = np.ascontiguousarray(X, dtype=np.float64)
            
            self.dissimilarity_matrix_ = None 
            
            solver_input = X
        
        self.n_features_in_ = X.shape[1]

        best_stress = np.inf
        best_embedding = None
        best_n_iter = 0
        best_history = None 

        rng = check_random_state(self.random_state)

        for i in range(self.n_init):
            run_seed = rng.randint(np.iinfo(np.int32).max)
            
            embedding, stress, n_iter, history = self._solve_true_sgd(solver_input, run_seed, n_epochs=self.max_iter)

            if stress < best_stress:
                best_stress = stress
                best_embedding = embedding
                best_n_iter = n_iter
                best_history = history

        self.embedding_ = best_embedding
        self.stress_ = best_stress
        self.n_iter_ = best_n_iter
        self.stress_history_ = best_history

        return self
    
    def fit_transform(self, X, y=None):
        """
        Fit the model from data in X and transform X.
        """
        self.fit(X, y)
        return self.embedding_
    
    def _get_scheduler(self, lr_init, w_min, w_max):
        kwargs = {
            "lr_init": lr_init,
            "max_iter": self.max_iter,
            "w_min": w_min,
            "w_max": w_max,
            "epsilon": self.epsilon,
            "switch_ratio": self.scheduler_switch_ratio,
        }

        schedulers = {
            "constant": _ConstantScheduler,
            "exponential": _ExponentialScheduler,
            "harmonic": _HarmonicScheduler,
            "hybrid": _HybridScheduler,
        }

        SchedulerClass = schedulers.get(self.scheduler)
        if SchedulerClass is None:
            SchedulerClass = _ExponentialScheduler
        
        return SchedulerClass(**kwargs)
    
    def _compute_weight_bounds(self, dissimilarity_matrix):
        if self.weighting == "uniform":
            return 1.0, 1.0, 1e-6
        
        upper_triangle_d = dissimilarity_matrix[np.triu_indices(dissimilarity_matrix.shape[0], k=1)]

        if upper_triangle_d.size == 0:
            return 1.0, 1.0, 1e-6
        
        d_min = np.min(upper_triangle_d)
        d_max = np.max(upper_triangle_d)

        floor_val = max(d_min, 1e-9)

        w_max = 1.0 / (floor_val ** 2)
        w_min = 1.0 / (d_max ** 2)

        return w_min, w_max, floor_val
    
    def _estimate_weight_bounds(self, X, all_pairs, rng, n_samples_est=1000):
        """
        Estimate min/max distances for learning rate initialization 
        without computing the full matrix (O(N) instead of O(N^2)).
        """
        if self.weighting == "uniform":
            return 1.0, 1.0
            
        n_pairs = all_pairs.shape[0]

        indices = rng.choice(n_pairs, size=min(n_pairs, n_samples_est), replace=False)
        sample_pairs = all_pairs[indices]
        
        X_i = X[sample_pairs[:, 0]]
        X_j = X[sample_pairs[:, 1]]
        
        dists = np.linalg.norm(X_i - X_j, axis=1)
        dists = dists[dists > 1e-9]
        
        if dists.size == 0:
             return 1.0, 1.0 
           
        d_min = np.min(dists)
        d_max = np.max(dists)

        w_max = 1.0 / (d_min ** 2)
        w_min = 1.0 / (d_max ** 2)
        
        return w_min, w_max
    
    def _resolve_n_pivots(self, n_samples):
        if self.n_pivots == "auto":
            k = min(50, max(self.n_components + 1, n_samples // 20))
        else:
            k = int(self.n_pivots)
        if k <= self.n_components:
            raise ValueError(
                f"n_pivots ({k}) must be greater than n_components ({self.n_components})."
            )
        if k > n_samples:
            raise ValueError(
                f"n_pivots ({k}) must be <= n_samples ({n_samples})."
            )
        return k

    def _select_pivots_maxmin(self, data, n_pivots, rng, is_lazy):
        """
        Greedy MaxMin (farthest-point) pivot selection.

        Brandes & Pich (2007), §3.2. Provides a 2-approximation of k-center.

        Parameters
        ----------
        data : ndarray
            Either a feature matrix (n_samples, n_features) when ``is_lazy``,
            or a precomputed dissimilarity matrix (n_samples, n_samples).
        n_pivots : int
        rng : RandomState
        is_lazy : bool
            If True, ``data`` is the feature matrix and per-pivot distances
            are recomputed; otherwise ``data`` is the dissimilarity matrix.

        Returns
        -------
        pivot_indices : ndarray of shape (n_pivots,), dtype int32
        """
        n_samples = data.shape[0]
        pivots = np.empty(n_pivots, dtype=np.int32)
        pivots[0] = rng.randint(n_samples)
        min_dist = np.full(n_samples, np.inf, dtype=np.float64)

        for k in range(n_pivots):
            p = pivots[k] if k == 0 else pivots[k]
            if is_lazy:
                diff = data - data[p]
                new_dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            else:
                new_dists = data[p].astype(np.float64, copy=False)

            np.minimum(min_dist, new_dists, out=min_dist)

            if k + 1 < n_pivots:
                pivots[k + 1] = int(np.argmax(min_dist))

        return pivots

    def _compute_full_stress(self, embedding, dissimilarity_matrix):
        dis = euclidean_distances(embedding, embedding)
        stress = 0.5 * np.sum((dis - dissimilarity_matrix) ** 2)
        return stress
    
    def _fit_monotonic_map(self, embedding, X, n_samples_isotonic = 2000):
        """
        Approximates the monotonic map f(delta) -> disparity using a random sample.
        Returns the sorted knot arrays (x_knots, y_knots) for interpolation.
        """
        n_points = embedding.shape[0]
        rng = check_random_state(self.random_state)

        idx_i = rng.randint(0, n_points, size=n_samples_isotonic)
        idx_j = rng.randint(0, n_points, size=n_samples_isotonic)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        X_emb_i = embedding[idx_i]
        X_emb_j = embedding[idx_j]
        dist_vector = np.linalg.norm(X_emb_i - X_emb_j, axis=1)

        X_raw_i = X[idx_i]
        X_raw_j = X[idx_j]
        delta_vector = np.linalg.norm(X_raw_i - X_raw_j, axis=1)

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        dist_disparity = ir.fit_transform(delta_vector, dist_vector)

        x_knots = np.ascontiguousarray(ir.f_.x, dtype=np.float64)
        y_knots = np.ascontiguousarray(ir.f_.y, dtype=np.float64)

        denom = np.sum(dist_disparity ** 2)
        if denom > 0:
            scale = np.sqrt(len(dist_disparity) / denom)
            y_knots *= scale
        
        return x_knots, y_knots
    
    def _solve_true_sgd(self, X, random_state, n_epochs=15):
        """
        Implements true SGD using Cython.
        Supports both Precomputed Matrix (O(N^2) RAM) and Lazy Calculation (O(N) RAM).
        """
        if not HAS_CYTHON:
            raise RuntimeError("Cython extension not compiled. Cannot run True SGD fast.")

        rng = check_random_state(random_state)
        n_samples = X.shape[0]
        n_pairs = (n_samples * (n_samples - 1)) // 2
        
        embedding = rng.standard_normal(size=(n_samples, self.n_components))
        embedding = np.ascontiguousarray(embedding, dtype=np.float64)

        is_pivot = self.sampling_strategy == "pivot"
        pivot_indices = None
        if is_pivot:
            n_pivots = self._resolve_n_pivots(n_samples)
            is_lazy_input = self.dissimilarity == "lazy"
            pivot_indices = self._select_pivots_maxmin(
                X, n_pivots, rng, is_lazy=is_lazy_input
            )
            self.pivot_indices_ = pivot_indices

        rows = None
        cols = None
        if self.sampling_strategy == "cycle":
            rows, cols = np.triu_indices(n_samples, k=1)
            all_pairs_canonical = np.column_stack((rows, cols)).astype(np.int32)
        elif is_pivot and self.dissimilarity in ["precomputed", "euclidean"]:
            all_idx = np.arange(n_samples, dtype=np.int32)
            rows_chunks = []
            cols_chunks = []
            for p in pivot_indices:
                p_int = int(p)
                mask = all_idx != p_int
                i_arr = all_idx[mask]
                j_arr = np.full(int(mask.sum()), p_int, dtype=np.int32)
                rows_chunks.append(np.minimum(i_arr, j_arr))
                cols_chunks.append(np.maximum(i_arr, j_arr))
            rows_raw = np.concatenate(rows_chunks)
            cols_raw = np.concatenate(cols_chunks)
            # Deduplicate: pivot-pivot pairs appear once per endpoint pivot,
            # so canonicalize (min,max) and take unique via int64 encoding.
            encoded = rows_raw.astype(np.int64) * n_samples + cols_raw.astype(np.int64)
            unique_enc = np.unique(encoded)
            rows = (unique_enc // n_samples).astype(np.int32)
            cols = (unique_enc % n_samples).astype(np.int32)
            all_pairs_canonical = np.column_stack((rows, cols)).astype(np.int32)
        else:
            all_pairs_canonical = None

        if self.dissimilarity in ["precomputed", "euclidean"]:
            if self.sampling_strategy == "random":
                raise ValueError("sampling_strategy='random' is not supported with precomputed/euclidean "
                                 "dissimilarity because random access to weights/distances is inefficient. "
                                 "Use dissimilarity='lazy'.")

            flat_distances = X[rows, cols].astype(np.float64)
            w_min, w_max, d_floor = self._compute_weight_bounds(X)

            if self.weighting == "inverse":
                clipped_d = np.maximum(flat_distances, d_floor)
                flat_weights_canonical = 1.0 / (clipped_d ** 2)
            else:
                flat_weights_canonical = np.ones(all_pairs_canonical.shape[0], dtype=np.float64)

            is_lazy = False

        else: # self.dissimilarity == "lazy"
            if all_pairs_canonical is not None:
                sample_pairs = all_pairs_canonical
            elif is_pivot:
                # Sample pivot-pairs for weight bound estimation.
                k = pivot_indices.shape[0]
                size = min(1000, k * n_samples)
                rand_i = rng.randint(0, n_samples, size=size)
                rand_p_idx = rng.randint(0, k, size=size)
                rand_j = pivot_indices[rand_p_idx].astype(np.int32, copy=False)
                mask = rand_i != rand_j
                sample_pairs = np.column_stack(
                    (rand_i[mask].astype(np.int32, copy=False), rand_j[mask])
                )
            else:
                dummy_rows = rng.randint(0, n_samples, size=1000)
                dummy_cols = rng.randint(0, n_samples, size=1000)
                mask = dummy_rows != dummy_cols # Filter out i=j
                sample_pairs = np.column_stack((dummy_rows[mask], dummy_cols[mask])).astype(np.int32)

            w_min, w_max = self._estimate_weight_bounds(X, sample_pairs, rng)

            flat_distances = None
            flat_weights_canonical = None
            is_lazy = True

        if self.learning_rate_init == "auto":
            lr_init = 1.0 / w_min
        else:
            lr_init = float(self.learning_rate_init)
        
        scheduler = self._get_scheduler(lr_init, w_min, w_max)

        history = []
        cumulative_time = 0.0
        
        def safe_compute_stress():
            if n_samples > 20000:
                return -1.0
            
            if is_lazy:
                D_temp = euclidean_distances(X, X)
                return self._compute_full_stress(embedding, D_temp)
            else:
                return self._compute_full_stress(embedding, X)

        current_stress = safe_compute_stress()
        history.append((0.0, current_stress))

        weighting_code = 1 if self.weighting == "inverse" else 0

        metric_mds = bool(self.metric)

        if not metric_mds and not is_lazy:
            ir = IsotonicRegression(out_of_bounds="clip")

            valid_idx = np.flatnonzero(flat_distances != 0.0)
            if valid_idx.size == 0:
                raise ValueError(
                "Non-metric MDS: all dissimilarities are 0 (treated as missing)."
                )
            
            order_valid = np.argsort(flat_distances[valid_idx], kind="mergesort")
            sorted_valid_idx = valid_idx[order_valid]

            x_sorted = np.ascontiguousarray(flat_distances[sorted_valid_idx], dtype=np.float64)
        
        for epoch in range(n_epochs):
            t0 = time.time()
            lr = scheduler.get_rate(epoch)

            if not is_lazy:
                if metric_mds:
                    target_canonical = flat_distances
                else:
                    diffs = embedding[rows] - embedding[cols]
                    flat_embed_dist = np.sqrt((diffs * diffs).sum(axis=1))

                    y_sorted = flat_embed_dist[sorted_valid_idx]

                    if epoch < 1:
                        disparities_sorted = x_sorted.copy()
                    else:
                        disparities_sorted = ir.fit_transform(x_sorted, y_sorted)
                    
                    target_canonical = np.zeros_like(flat_distances)
                    target_canonical[sorted_valid_idx] = disparities_sorted

                    denom = np.sum(target_canonical * target_canonical)
                    if denom > 0.0:
                        target_canonical *= np.sqrt(target_canonical.shape[0] / denom)
                
                pairs_epoch, target_epoch, weights_epoch = sklearn_shuffle(
                    all_pairs_canonical, target_canonical, flat_weights_canonical, random_state=rng
                )
                
                run_sgd_epoch(
                    embedding, 
                    target_epoch, 
                    weights_epoch, 
                    pairs_epoch,
                    lr
                )

            else:
                x_knots = None
                y_knots = None

                if not metric_mds:
                    x_knots, y_knots = self._fit_monotonic_map(embedding, X)

                current_seed = rng.randint(1, 2**31 - 1)

                if is_pivot:
                    n_updates = int(pivot_indices.shape[0]) * n_samples
                    run_sgd_epoch_pivot_lazy(
                        embedding,
                        X,
                        pivot_indices,
                        n_updates,
                        lr,
                        weighting_code,
                        current_seed,
                        x_knots=x_knots,
                        y_knots=y_knots,
                    )
                else:
                    run_sgd_epoch_lazy_random_native(
                        embedding,
                        X,
                        n_pairs,
                        lr,
                        weighting_code,
                        current_seed,
                        x_knots=x_knots,
                        y_knots=y_knots,
                    )

            t1 = time.time()
            cumulative_time += t1 - t0

            # Logging (skip if unsafe)
            if n_samples <= 10000:
                current_stress = safe_compute_stress()
                history.append((cumulative_time, current_stress))
        
        embedding -= np.mean(embedding, axis=0)
        
        stress = safe_compute_stress()

        return embedding, stress, n_epochs, history