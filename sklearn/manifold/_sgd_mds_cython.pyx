# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython kernels for SGD-MDS.

Three kernels are exposed, all sharing the same SGD update rule:

- ``run_sgd_epoch``                    : matrix mode (cycle / pivot).
  Pre-extracted ``target_distances`` and ``weights`` are streamed
  alongside a shuffled pair list.
- ``run_sgd_epoch_lazy_random_native`` : lazy mode, random pair sampling.
- ``run_sgd_epoch_pivot_lazy``         : lazy mode, pivot-restricted sampling.

The two lazy kernels differ only in how ``j`` is drawn; the inner gradient
step is factored into the inlined helper :c:func:`_lazy_sgd_step`. Knot
arrays for the non-metric isotonic map are passed as raw ``double*`` so
that ``NULL`` is a valid "no-op" sentinel inside ``nogil``.
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
cimport numpy as cn
cimport cython

cn.import_array()


# =============================================================================
# Inline helpers
# =============================================================================

cdef inline cn.uint32_t xorshift32(cn.uint32_t* state) nogil:
    """Lightweight xorshift32 PRNG (Marsaglia, 2003)."""
    cdef cn.uint32_t x = state[0]
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    state[0] = x
    return x


cdef inline double _interp_isotonic(
    double val,
    double* x_knots,
    double* y_knots,
    int n_knots,
) nogil:
    """Linear interpolation on sorted ``(x_knots, y_knots)`` via binary search.

    Used to evaluate the non-metric isotonic dissimilarity-to-disparity map.
    """
    if val <= x_knots[0]:
        return y_knots[0]
    if val >= x_knots[n_knots - 1]:
        return y_knots[n_knots - 1]

    cdef int left = 0
    cdef int right = n_knots - 1
    cdef int mid

    while left < right - 1:
        mid = (left + right) // 2
        if x_knots[mid] <= val:
            left = mid
        else:
            right = mid

    cdef double x0 = x_knots[left]
    cdef double x1 = x_knots[left + 1]
    cdef double y0 = y_knots[left]
    cdef double y1 = y_knots[left + 1]

    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (val - x0) / (x1 - x0)


cdef inline void _lazy_sgd_step(
    double[:, ::1] embedding,
    double[:, ::1] X_original,
    int i,
    int j,
    int n_features,
    int n_components,
    double lr,
    int weighting_code,
    int is_non_metric,
    double* x_knots,        # may be NULL when is_non_metric == 0
    double* y_knots,        # may be NULL when is_non_metric == 0
    int n_knots,
    double* diff_vector,
) nogil:
    """One SGD update for a (i, j) pair in lazy mode.

    Computes the feature-space distance ``delta``, the embedding distance
    ``dist``, and applies the symmetric clipped step
    ``(dist - target) * step_val / 2`` to both rows.

    Skips silently if ``i == j`` or if the embedding distance is ~0.
    """
    if i == j:
        return

    cdef int f, d
    cdef double diff
    cdef double delta = 0.0
    cdef double dist = 0.0
    cdef double w, target, step_val, ratio, grad

    # delta = ||X[i] - X[j]||  (feature space)
    for f in range(n_features):
        diff = X_original[i, f] - X_original[j, f]
        delta += diff * diff
    delta = sqrt(delta)

    if weighting_code == 1:  # inverse weighting (Sammon-style)
        if delta < 1e-6:
            delta = 1e-6
        w = 1.0 / (delta * delta)
    else:
        w = 1.0

    if is_non_metric:
        target = _interp_isotonic(delta, x_knots, y_knots, n_knots)
    else:
        target = delta

    # dist = ||embedding[i] - embedding[j]||  (embedding space)
    for d in range(n_components):
        diff = embedding[i, d] - embedding[j, d]
        diff_vector[d] = diff
        dist += diff * diff
    dist = sqrt(dist)

    if dist < 1e-12:
        return

    step_val = w * lr
    if step_val > 1.0:
        step_val = 1.0

    ratio = step_val * (dist - target) / (2.0 * dist)

    for d in range(n_components):
        grad = ratio * diff_vector[d]
        embedding[i, d] -= grad
        embedding[j, d] += grad


# =============================================================================
# Matrix-mode kernel (cycle or pivot sampling against a precomputed matrix)
# =============================================================================

cpdef void run_sgd_epoch(
    double[:, ::1] embedding,        # (n_samples, n_components)
    double[::1] target_distances,    # (n_pairs,)         pre-extracted
    double[::1] weights,             # (n_pairs,)         pre-computed
    cn.int32_t[:, ::1] pairs,        # (n_pairs, 2)
    double lr,
):
    """One SGD epoch over a fixed pair list with pre-extracted distances/weights.

    Used for both ``sampling_strategy='cycle'`` (all upper-triangle pairs)
    and ``sampling_strategy='pivot'`` against a precomputed/euclidean
    dissimilarity matrix. The caller is expected to shuffle ``(pairs,
    target_distances, weights)`` together before each call to retain SGD
    randomness.
    """
    cdef int k, i, j, d
    cdef int n_pairs = pairs.shape[0]
    cdef int n_components = embedding.shape[1]

    cdef double dist, delta, move_mag, diff_val, ratio, w_ij, mu

    with nogil:
        for k in range(n_pairs):
            i = pairs[k, 0]
            j = pairs[k, 1]

            dist = 0.0
            for d in range(n_components):
                diff_val = embedding[i, d] - embedding[j, d]
                dist += diff_val * diff_val
            dist = sqrt(dist)

            if dist <= 1e-12:
                continue

            delta = target_distances[k]
            w_ij = weights[k]

            mu = w_ij * lr
            if mu > 1.0:
                mu = 1.0

            move_mag = (dist - delta) * 0.5 * mu
            ratio = move_mag / dist

            for d in range(n_components):
                diff_val = embedding[i, d] - embedding[j, d]
                embedding[i, d] -= ratio * diff_val
                embedding[j, d] += ratio * diff_val


# =============================================================================
# Lazy-mode kernels (distances computed on the fly)
# =============================================================================

cpdef void run_sgd_epoch_lazy_random_native(
    double[:, ::1] embedding,        # (n_samples, n_components), in-place
    double[:, ::1] X_original,       # (n_samples, n_features)
    long n_updates,                  # number of (i, j) draws this epoch
    double lr,
    int weighting_code,              # 1 => inverse, 0 => uniform
    cn.uint32_t seed,
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    """One SGD epoch with random pair sampling (lazy mode).

    Each iteration draws ``i, j`` uniformly from ``[0, n_samples)`` and
    applies one update via :c:func:`_lazy_sgd_step`.
    """
    cdef:
        long k
        int i, j
        int n_samples = X_original.shape[0]
        int n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        cn.uint32_t rng_state = seed if seed != 0 else 1
        double* diff_vector = <double*> malloc(n_components * sizeof(double))
        double* x_knots_ptr = NULL
        double* y_knots_ptr = NULL

    if diff_vector == NULL:
        raise MemoryError("Failed to allocate diff_vector")

    if is_non_metric:
        n_knots = x_knots.shape[0]
        x_knots_ptr = &x_knots[0]
        y_knots_ptr = &y_knots[0]

    try:
        with nogil:
            for k in range(n_updates):
                i = <int>(xorshift32(&rng_state) % n_samples)
                j = <int>(xorshift32(&rng_state) % n_samples)
                _lazy_sgd_step(
                    embedding, X_original, i, j,
                    n_features, n_components, lr, weighting_code,
                    is_non_metric, x_knots_ptr, y_knots_ptr, n_knots,
                    diff_vector,
                )
    finally:
        free(diff_vector)


cpdef void run_sgd_epoch_pivot_lazy(
    double[:, ::1] embedding,        # (n_samples, n_components), in-place
    double[:, ::1] X_original,       # (n_samples, n_features)
    cn.int32_t[::1] pivot_indices,   # (n_pivots,)
    long n_updates,                  # typically n_pivots * n_samples
    double lr,
    int weighting_code,
    cn.uint32_t seed,
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    """One SGD epoch with pivot-restricted random sampling (lazy mode).

    Each iteration draws ``i`` uniformly from ``[0, n_samples)`` and ``j``
    uniformly from ``pivot_indices``. This restricts SGD updates to
    ``O(k * n_samples)`` pairs per epoch instead of ``O(n_samples^2)``.
    """
    cdef:
        long k_iter
        int i, j, p_idx
        int n_samples = X_original.shape[0]
        int n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        int n_pivots = pivot_indices.shape[0]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        cn.uint32_t rng_state = seed if seed != 0 else 1
        double* diff_vector = <double*> malloc(n_components * sizeof(double))
        double* x_knots_ptr = NULL
        double* y_knots_ptr = NULL

    if diff_vector == NULL:
        raise MemoryError("Failed to allocate diff_vector")

    if is_non_metric:
        n_knots = x_knots.shape[0]
        x_knots_ptr = &x_knots[0]
        y_knots_ptr = &y_knots[0]

    try:
        with nogil:
            for k_iter in range(n_updates):
                i = <int>(xorshift32(&rng_state) % n_samples)
                p_idx = <int>(xorshift32(&rng_state) % n_pivots)
                j = pivot_indices[p_idx]
                _lazy_sgd_step(
                    embedding, X_original, i, j,
                    n_features, n_components, lr, weighting_code,
                    is_non_metric, x_knots_ptr, y_knots_ptr, n_knots,
                    diff_vector,
                )
    finally:
        free(diff_vector)
