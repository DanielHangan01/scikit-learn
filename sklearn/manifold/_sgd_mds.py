"""
Stochastic Gradient Descent Multidimensional Scaling (SGD-MDS)
"""
# Author: Daniel Hangan
# License: BSD 3 clause

import numpy as np
from numbers import Integral, Real

from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
from ..utils import check_random_state
from ..utils.validation import check_array
from ..utils._param_validation import Interval, StrOptions
from sklearn.metrics import euclidean_distances
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.utils.validation import check_symmetric
import time

# Try importing the Cython extension
try:
    from ._sgd_mds_cython import run_sgd_epoch, run_sgd_epoch_lazy, run_sgd_epoch_lazy_random_native
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
    
class _ConvergentScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)

        lr_final = self.epsilon / self.w_max

        if self.lr_init < lr_final:
            self.decay = 0.0
        else:
            self.decay = -np.log(lr_final / self.lr_init) / (self.max_iter - 1)
        
        m = self.w_max / self.w_min

        
        if self.decay == 0 or m <= 1.0000001:
            self.tau = 0
        else:
            self.tau = np.log(m) / self.decay
        
    def get_rate(self, t):
        if t < self.tau:
            return self.lr_init * np.exp(-self.decay * t)
        
        return 1.0 / (self.w_max * (1.0 + self.decay * (t - self.tau)))
        
class _CosineScheduler(_BaseScheduler):
    def __init__(self, lr_init, w_min, w_max, max_iter, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)

        self.eta_max = self.lr_init
        self.eta_min = self.epsilon / self.w_max

    def get_rate(self, t):
        if t >= self.max_iter:
            return self.eta_min

        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(t / self.max_iter * np.pi))
    
class _SineScheduler(_BaseScheduler):
    def get_rate(self, t):
        if t >= self.max_iter:
            return 0.0
        
        fraction = (self.max_iter - t) / (4 * self.max_iter)
        return 2 * self.lr_init * (np.sin(fraction * np.pi) ** 2)
    
class _HybridScheduler(_BaseScheduler):
    """
    Combines Exponential decay (fast unfolding) with Harmonic decay (fine convergence).
    Forces the switch to happen at 'switch_ratio' of max_iter regardless of weights.
    """
    def __init__(self, lr_init, w_min, w_max, max_iter, switch_ratio=0.4, epsilon=0.01, **kwargs):
        super().__init__(lr_init, w_min, w_max, max_iter, epsilon, **kwargs)
        
        # Force the switch at 40% of epochs (tunable)
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
        "scheduler": [StrOptions({"exponential", "harmonic", "convergent", "cosine", "sine", "constant", "hybrid"})],
        "weighting": [StrOptions({"uniform", "inverse"})],
        "sampling_strategy": [StrOptions({"cycle", "random"})],
    }

    def __init__(self, n_components=2, *, metric=True, n_init=1,
                 max_iter=300, tol=1e-4, learning_rate_init="auto",
                 batch_size=None, dissimilarity="euclidean",
                 algorithm="mini_batch", scheduler="exponential", weighting="uniform",
                 sampling_strategy="cycle", n_jobs=None, random_state=None):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.dissimilarity = dissimilarity
        self.algorithm = algorithm
        self.scheduler = scheduler
        self.weighting = weighting
        self.sampling_strategy = sampling_strategy
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

        # 1. PRECOMPUTED: User provides the matrix
        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            if self.dissimilarity_matrix_.shape[0] != self.dissimilarity_matrix_.shape[1]:
                raise ValueError("Precomputed dissimilarity matrix must be square.")
            check_symmetric(self.dissimilarity_matrix_, raise_exception=True)
            
            solver_input = self.dissimilarity_matrix_

        # 2. EUCLIDEAN: User provides features, we precalculate matrix
        elif self.dissimilarity == "euclidean":
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            # Store full matrix (O(N^2) memory)
            self.dissimilarity_matrix_ = euclidean_distances(X, X)
            
            solver_input = self.dissimilarity_matrix_

        # 3. LAZY: User provides features, we use them raw (Scalable for N > 10k)
        elif self.dissimilarity == "lazy":
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
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
            
            if self.algorithm == "true_sgd":
                embedding, stress, n_iter, history = self._solve_true_sgd(solver_input, run_seed, n_epochs=self.max_iter)
            else:
                embedding, stress, n_iter, history = self._solve_sgd(solver_input, run_seed)

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
            "epsilon": 0.1,
        }

        schedulers = {
            "constant": _ConstantScheduler,
            "exponential": _ExponentialScheduler,
            "harmonic": _HarmonicScheduler,
            "convergent": _ConvergentScheduler,
            "cosine": _CosineScheduler,
            "sine": _SineScheduler,
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
        # Sample random pairs to estimate bounds
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
    
    def _compute_full_stress(self, embedding, dissimilarity_matrix):
        dis = euclidean_distances(embedding, embedding)
        stress = 0.5 * np.sum((dis - dissimilarity_matrix) ** 2)
        return stress

    def _solve_sgd(self, dissimilarity_matrix, random_state):
        """
        Single run of the SGD optimization loop.
        """
        rng = check_random_state(random_state)
        n_samples = dissimilarity_matrix.shape[0]

        embedding = rng.standard_normal(size=(n_samples, self.n_components))

        w_min, w_max, d_floor = self._compute_weight_bounds(dissimilarity_matrix)

        if self.learning_rate_init == "auto":
            lr_init = 1.0 / w_min
        else:
            lr_init = float(self.learning_rate_init)
        
        scheduler = self._get_scheduler(lr_init, w_min, w_max)

        if self.batch_size is None:
            batch_size = min(50_000, n_samples * (n_samples - 1) // 2)
        else:
            batch_size = self.batch_size

        history = []
        cumulative_time = 0.0

        current_stress = self._compute_full_stress(embedding, dissimilarity_matrix)
        history.append((0.0, current_stress))

        for it in range(self.max_iter):
            t0 = time.time()

            i_idx = rng.randint(0, n_samples, size=batch_size)
            j_idx = rng.randint(0, n_samples, size=batch_size)

            X_i = embedding[i_idx]
            X_j = embedding[j_idx]
            delta = dissimilarity_matrix[i_idx, j_idx]

            diff = X_i - X_j
            dist = np.linalg.norm(diff, axis=1)

            if self.weighting == "inverse":
                delta_safe = np.maximum(delta, d_floor)
                w = 1.0 / (delta_safe ** 2)
            else:
                w = 1.0

            lr = scheduler.get_rate(it)
            
            m = np.minimum(w * lr, 1.0)

            ratio = (dist - delta) / (2 * np.maximum(dist, 1e-12)) 
            
            scale = m * ratio
            update = scale[:, None] * diff

            update_norms = np.linalg.norm(update, axis=1, keepdims=True)
            scaling_factors = np.minimum(1.0, 1.0 / (update_norms + 1e-12))
            update *= scaling_factors

            np.add.at(embedding, i_idx, -update)
            np.add.at(embedding, j_idx, update)

            t1 = time.time()
            cumulative_time += t1 - t0

            if (it % 100 == 0):
                current_stress = self._compute_full_stress(embedding, dissimilarity_matrix)
                history.append((cumulative_time, current_stress))

            max_move = np.max(update_norms)
            if max_move < self.tol:
                break

        embedding -= np.mean(embedding, axis=0)

        stress = self._compute_full_stress(embedding, dissimilarity_matrix)
        history.append((cumulative_time, stress))

        return embedding, stress, it + 1, history
    
    
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

        if self.sampling_strategy == "cycle":
            rows, cols = np.triu_indices(n_samples, k=1)
            all_pairs = np.column_stack((rows, cols)).astype(np.int32)
        else:
            all_pairs = None
        
        if self.dissimilarity in ["precomputed", "euclidean"]:
            if self.sampling_strategy == "random":
                raise ValueError("sampling_strategy='random' is not supported with precomputed/euclidean "
                                 "dissimilarity because random access to weights/distances is inefficient. "
                                 "Use dissimilarity='lazy'.")
            
            flat_distances = X[rows, cols].astype(np.float64)
            w_min, w_max, d_floor = self._compute_weight_bounds(X)

            if self.weighting == "inverse":
                clipped_d = np.maximum(flat_distances, d_floor)
                flat_weights = 1.0 / (clipped_d ** 2)
            else:
                flat_weights = np.ones(all_pairs.shape[0], dtype=np.float64)
            
            is_lazy = False

        else: # self.dissimilarity == "lazy"
            if all_pairs is None:
                dummy_rows = rng.randint(0, n_samples, size=1000)
                dummy_cols = rng.randint(0, n_samples, size=1000)
                mask = dummy_rows != dummy_cols # Filter out i=j
                sample_pairs = np.column_stack((dummy_rows[mask], dummy_cols[mask])).astype(np.int32)
            else:
                sample_pairs = all_pairs
            
            w_min, w_max = self._estimate_weight_bounds(X, sample_pairs, rng)
            
            flat_distances = None
            flat_weights = None
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

        # Use the helper
        # current_stress = safe_compute_stress()
        # history.append((0.0, current_stress))
        
        for epoch in range(n_epochs):
            t0 = time.time()
            lr = scheduler.get_rate(epoch)

            if not is_lazy:
                # Standard Mode: We must shuffle strict alignment of pairs, distances, and weights.
                all_pairs, flat_distances, flat_weights = sklearn_shuffle(
                    all_pairs, flat_distances, flat_weights, random_state=rng
                )
                
                run_sgd_epoch(
                    embedding, 
                    flat_distances, 
                    flat_weights, 
                    all_pairs, 
                    lr
                )

            else:
                # We don't generate arrays in Python. We pass the count.
                # This achieves O(1) auxiliary memory usage.
                
                # Pass a fresh seed every epoch for the C-level RNG
                current_seed = rng.randint(0, 2**30)
                
                run_sgd_epoch_lazy_random_native(
                    embedding,
                    X,
                    n_pairs,
                    lr,
                    self.weighting,
                    current_seed
                )

            t1 = time.time()
            cumulative_time += t1 - t0

            # Logging (skip if unsafe)
            # if n_samples < 20000:
            #   current_stress = safe_compute_stress()
            #   history.append((cumulative_time, current_stress))
        
        embedding -= np.mean(embedding, axis=0)
        
        # stress = safe_compute_stress()
        stress = -1.0

        return embedding, stress, n_epochs, history