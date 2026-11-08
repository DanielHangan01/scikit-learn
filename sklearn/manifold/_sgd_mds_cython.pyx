"""Cython kernels for SGD-MDS."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport sqrt
from libc.stdlib cimport free, malloc

import numpy as np

from sklearn.utils._typedefs cimport int32_t, uint32_t


cdef inline uint32_t xorshift32(uint32_t* state) noexcept nogil:
    """Xorshift32 pseudo-random number generator (Marsaglia, 2003)."""
    cdef uint32_t x = state[0]
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
) noexcept nogil:
    """Evaluate the isotonic map of non-metric MDS at `val`."""
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


cdef inline void _pair_update(
    double[:, ::1] embedding,
    const double[:, ::1] X,
    int i,
    int j,
    int n_features,
    int n_components,
    double lr,
    int weighting_code,
    int is_non_metric,
    double* x_knots,
    double* y_knots,
    int n_knots,
    double* diff_vector,
) noexcept nogil:
    """Update the pair (i, j), computing its distance in feature space."""
    if i == j:
        return

    cdef int f, d
    cdef double diff
    cdef double delta = 0.0
    cdef double dist = 0.0
    cdef double w, target, step_val, ratio, grad

    for f in range(n_features):
        diff = X[i, f] - X[j, f]
        delta += diff * diff
    delta = sqrt(delta)

    if weighting_code == 1:
        if delta < 1e-6:
            delta = 1e-6
        w = 1.0 / (delta * delta)
    else:
        w = 1.0

    if is_non_metric:
        target = _interp_isotonic(delta, x_knots, y_knots, n_knots)
    else:
        target = delta

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


cpdef void run_sgd_epoch(
    double[:, ::1] embedding,
    double[::1] target_distances,
    double[::1] weights,
    int32_t[:, ::1] pairs,
    double lr,
):
    """Run one epoch over a shuffled list of pairs with known distances."""
    cdef Py_ssize_t k, i, j, d
    cdef Py_ssize_t n_pairs = pairs.shape[0]
    cdef Py_ssize_t n_components = embedding.shape[1]
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


cpdef void run_sgd_epoch_random(
    double[:, ::1] embedding,
    const double[:, ::1] X,
    Py_ssize_t n_updates,
    double lr,
    int weighting_code,
    uint32_t seed,
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    """Run one epoch of `n_updates` uniformly drawn pairs."""
    cdef:
        Py_ssize_t k
        int i, j
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_components = embedding.shape[1]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        uint32_t rng_state = seed if seed != 0 else 1
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
                _pair_update(
                    embedding, X, i, j, n_features, n_components, lr,
                    weighting_code, is_non_metric, x_knots_ptr, y_knots_ptr,
                    n_knots, diff_vector,
                )
    finally:
        free(diff_vector)


cdef inline int _pair_move(
    const double* emb,
    Py_ssize_t i,
    Py_ssize_t j,
    double delta,
    int n_components,
    double lr,
    int weighting_code,
    int is_non_metric,
    double* x_knots,
    double* y_knots,
    int n_knots,
    double* diff_vector,
    double* ratio_out,
) noexcept nogil:
    """Compute the step of pair (i, j) given its distance `delta`.

    Returns 0 if the pair must be skipped.
    """
    if i == j:
        return 0

    cdef int d
    cdef double diff
    cdef double dist = 0.0
    cdef double w, target, step_val

    if weighting_code == 1:
        if delta < 1e-6:
            delta = 1e-6
        w = 1.0 / (delta * delta)
    else:
        w = 1.0

    if is_non_metric:
        target = _interp_isotonic(delta, x_knots, y_knots, n_knots)
    else:
        target = delta

    for d in range(n_components):
        diff = emb[i * n_components + d] - emb[j * n_components + d]
        diff_vector[d] = diff
        dist += diff * diff
    dist = sqrt(dist)

    if dist < 1e-12:
        return 0

    step_val = w * lr
    if step_val > 1.0:
        step_val = 1.0

    ratio_out[0] = step_val * (dist - target) / (2.0 * dist)
    return 1


cdef inline void _next_block(
    int* idx,
    int m,
    int n_samples,
    uint32_t* rng_state,
) noexcept nogil:
    """Draw `m` points uniformly, with replacement."""
    cdef int a
    for a in range(m):
        idx[a] = <int>(xorshift32(rng_state) % n_samples)


cdef inline void _block_distances(
    const double* X,
    Py_ssize_t n_features,
    int* idx,
    int* pair_left,
    int* pair_right,
    int n_pairs,
    double* delta,
) noexcept nogil:
    """Compute the distances of the first `n_pairs` pairs of a block.

    Pairs are processed four at a time: four independent running sums overlap
    in the pipeline, where a single one is limited by the latency of an add.
    """
    cdef int p = 0
    cdef Py_ssize_t f
    cdef const double* ri
    cdef const double* rj
    cdef const double* a0
    cdef const double* b0
    cdef const double* a1
    cdef const double* b1
    cdef const double* a2
    cdef const double* b2
    cdef const double* a3
    cdef const double* b3
    cdef double diff, acc, t0, t1, t2, t3
    cdef double s0, s1, s2, s3

    while p + 4 <= n_pairs:
        a0 = X + idx[pair_left[p]] * n_features
        b0 = X + idx[pair_right[p]] * n_features
        a1 = X + idx[pair_left[p + 1]] * n_features
        b1 = X + idx[pair_right[p + 1]] * n_features
        a2 = X + idx[pair_left[p + 2]] * n_features
        b2 = X + idx[pair_right[p + 2]] * n_features
        a3 = X + idx[pair_left[p + 3]] * n_features
        b3 = X + idx[pair_right[p + 3]] * n_features
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for f in range(n_features):
            t0 = a0[f] - b0[f]
            s0 += t0 * t0
            t1 = a1[f] - b1[f]
            s1 += t1 * t1
            t2 = a2[f] - b2[f]
            s2 += t2 * t2
            t3 = a3[f] - b3[f]
            s3 += t3 * t3
        delta[p] = sqrt(s0)
        delta[p + 1] = sqrt(s1)
        delta[p + 2] = sqrt(s2)
        delta[p + 3] = sqrt(s3)
        p += 4

    while p < n_pairs:
        ri = X + idx[pair_left[p]] * n_features
        rj = X + idx[pair_right[p]] * n_features
        acc = 0.0
        for f in range(n_features):
            diff = ri[f] - rj[f]
            acc += diff * diff
        delta[p] = sqrt(acc)
        p += 1


cdef inline void _block_distances_4(
    const double* X,
    Py_ssize_t n_features,
    int* idx,
    double* delta,
) noexcept nogil:
    """Compute the six distances of a block of four in one pass over `X`."""
    cdef Py_ssize_t f
    cdef const double* r0 = X + idx[0] * n_features
    cdef const double* r1 = X + idx[1] * n_features
    cdef const double* r2 = X + idx[2] * n_features
    cdef const double* r3 = X + idx[3] * n_features
    cdef double x0, x1, x2, x3, t
    cdef double s01 = 0.0, s02 = 0.0, s03 = 0.0
    cdef double s12 = 0.0, s13 = 0.0, s23 = 0.0

    for f in range(n_features):
        x0 = r0[f]
        x1 = r1[f]
        x2 = r2[f]
        x3 = r3[f]
        t = x0 - x1
        s01 += t * t
        t = x0 - x2
        s02 += t * t
        t = x0 - x3
        s03 += t * t
        t = x1 - x2
        s12 += t * t
        t = x1 - x3
        s13 += t * t
        t = x2 - x3
        s23 += t * t

    delta[0] = sqrt(s01)
    delta[1] = sqrt(s02)
    delta[2] = sqrt(s03)
    delta[3] = sqrt(s12)
    delta[4] = sqrt(s13)
    delta[5] = sqrt(s23)


cpdef Py_ssize_t run_sgd_epoch_blocked(
    double[:, ::1] embedding,
    const double[:, ::1] X,
    Py_ssize_t n_updates,
    int block_size,
    double lr,
    int weighting_code,
    uint32_t seed,
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    """Run one epoch of `n_updates` pair updates grouped in blocks of points.

    Each block draws `block_size` points and computes the distances between
    all their pairs from a single read of their rows, then updates the pairs
    sequentially. The last block only processes the pairs still owed.

    Returns the number of blocks drawn.
    """
    cdef:
        int m = block_size
        int n_block_pairs = m * (m - 1) // 2
        int n_samples = X.shape[0]
        Py_ssize_t n_features = X.shape[1]
        int n_components = embedding.shape[1]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        uint32_t rng_state = seed if seed != 0 else 1
        Py_ssize_t done = 0
        Py_ssize_t n_blocks = 0
        Py_ssize_t i, j
        int n_do, p, a, b, d
        double ratio, grad
        double* emb = &embedding[0, 0]
        const double* X_ptr = &X[0, 0]
        double* x_knots_ptr = NULL
        double* y_knots_ptr = NULL
        int* idx = NULL
        int* pair_left = NULL
        int* pair_right = NULL
        double* delta = NULL
        double* diff_vector = NULL

    if m < 2 or m > n_samples:
        raise ValueError(
            f"block_size must be in [2, n_samples], got {m} with "
            f"n_samples={n_samples}."
        )
    if n_updates <= 0:
        return 0

    if is_non_metric:
        n_knots = x_knots.shape[0]
        x_knots_ptr = &x_knots[0]
        y_knots_ptr = &y_knots[0]

    try:
        idx = <int*> malloc(m * sizeof(int))
        pair_left = <int*> malloc(n_block_pairs * sizeof(int))
        pair_right = <int*> malloc(n_block_pairs * sizeof(int))
        delta = <double*> malloc(n_block_pairs * sizeof(double))
        diff_vector = <double*> malloc(n_components * sizeof(double))
        if (idx == NULL or pair_left == NULL or pair_right == NULL
                or delta == NULL or diff_vector == NULL):
            raise MemoryError("Failed to allocate block buffers")

        p = 0
        for a in range(m):
            for b in range(a + 1, m):
                pair_left[p] = a
                pair_right[p] = b
                p += 1

        with nogil:
            while done < n_updates:
                _next_block(idx, m, n_samples, &rng_state)
                n_do = n_block_pairs
                if n_updates - done < n_do:
                    n_do = <int>(n_updates - done)

                if m == 4:
                    _block_distances_4(X_ptr, n_features, idx, delta)
                else:
                    _block_distances(X_ptr, n_features, idx, pair_left,
                                     pair_right, n_do, delta)

                for p in range(n_do):
                    i = idx[pair_left[p]]
                    j = idx[pair_right[p]]
                    if _pair_move(emb, i, j, delta[p], n_components, lr,
                                  weighting_code, is_non_metric, x_knots_ptr,
                                  y_knots_ptr, n_knots, diff_vector, &ratio):
                        for d in range(n_components):
                            grad = ratio * diff_vector[d]
                            emb[i * n_components + d] -= grad
                            emb[j * n_components + d] += grad

                done += n_do
                n_blocks += 1
    finally:
        free(idx)
        free(pair_left)
        free(pair_right)
        free(delta)
        free(diff_vector)

    return n_blocks


def draw_blocks(int n_samples, int block_size, Py_ssize_t n_blocks, uint32_t seed):
    """Return the first blocks drawn by `run_sgd_epoch_blocked` (for testing)."""
    cdef:
        int32_t[:, ::1] out = np.empty((n_blocks, block_size), dtype=np.int32)
        int* idx = <int*> malloc(block_size * sizeof(int))
        uint32_t rng_state = seed if seed != 0 else 1
        Py_ssize_t k
        int a

    if idx == NULL:
        raise MemoryError("Failed to allocate idx")
    try:
        for k in range(n_blocks):
            _next_block(idx, block_size, n_samples, &rng_state)
            for a in range(block_size):
                out[k, a] = idx[a]
    finally:
        free(idx)
    return np.asarray(out)
