# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython kernels for SGD-MDS.

Four kernels are exposed, all sharing the same SGD update rule:

- ``run_sgd_epoch``                    : matrix mode (cycle / pivot).
  Pre-extracted ``target_distances`` and ``weights`` are streamed
  alongside a shuffled pair list.
- ``run_sgd_epoch_lazy_random_native`` : lazy mode, random pair sampling.
- ``run_sgd_epoch_pivot_lazy``         : lazy mode, pivot-restricted sampling.
- ``run_sgd_epoch_lazy_blocked``       : lazy mode, point-blocked random
  sampling (experimental). Draws ``m`` points and updates all
  ``m(m-1)/2`` pairs among them, computing their feature-space distances
  from a single read of the ``m`` rows.

The two original lazy kernels differ only in how ``j`` is drawn; the inner
gradient step is factored into the inlined helper :c:func:`_lazy_sgd_step`.
The blocked kernel reproduces the random kernel bit for bit at ``m=2`` with
random blocks (see its docstring). Knot arrays for the non-metric isotonic
map are passed as raw ``double*`` so that ``NULL`` is a valid "no-op"
sentinel inside ``nogil``.
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset
cimport numpy as cn
cimport cython
import numpy as np

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
    # Py_ssize_t, not int: the cycle pair list is N(N-1)/2 entries, which
    # passes 2**31-1 at N = 65,537. A 32-bit counter there makes the loop
    # bound negative and the epoch a silent no-op. Py_ssize_t rather than
    # long because long is 32 bits on Windows (LLP64); it is also the type
    # of .shape[], so no conversion happens.
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


# =============================================================================
# Lazy-mode kernels (distances computed on the fly)
# =============================================================================

cpdef void run_sgd_epoch_lazy_random_native(
    double[:, ::1] embedding,        # (n_samples, n_components), in-place
    double[:, ::1] X_original,       # (n_samples, n_features)
    Py_ssize_t n_updates,            # number of (i, j) draws this epoch
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
        Py_ssize_t k
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
    Py_ssize_t n_updates,            # typically n_pivots * n_samples
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
        Py_ssize_t k_iter
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


# =============================================================================
# Point-blocked lazy kernel (experimental)
# =============================================================================
#
# The random kernel above loads two feature rows per pair update. Drawing m
# points instead and updating every pair among them loads m rows for
# m(m-1)/2 pairs, i.e. 2*D/(m-1) row reads per pair instead of 2*D. The
# update rule, the objective (raw stress) and the per-epoch budget (counted
# in PAIRS) are unchanged; only which pairs are grouped together changes.
#
# Everything here works on raw pointers into C-contiguous arrays, so no
# memoryview slice is passed per pair.

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
    """Step size for pair (i, j) given its feature-space distance ``delta``.

    The second half of :c:func:`_lazy_sgd_step`, written with the same
    expressions in the same order so that it is bit-identical to it. Fills
    ``diff_vector`` with ``emb[i] - emb[j]`` and ``ratio_out`` with the
    factor each row moves by along it. Returns 0 (no update) if ``i == j``
    or the embedding distance is ~0, exactly where the original skips.
    """
    if i == j:
        return 0

    cdef int d
    cdef double diff
    cdef double dist = 0.0
    cdef double w, target, step_val

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
    bint partition,
    cn.int32_t* perm,
    Py_ssize_t* pos,
    cn.uint32_t* rng_state,
) noexcept nogil:
    """Fill ``idx[0:m]`` with the next block's point indices.

    random    : m independent uniform draws, WITH replacement. Every pair of
                slots is then an independent uniform (i, j) draw -- the
                random kernel's pair distribution exactly -- and at m=2 the
                two draws are the random kernel's i and j, in order.
                Duplicates within a block give i == j pairs, which are
                skipped but still counted, as the random kernel does.
    partition : the next m entries of a shuffled permutation of all points,
                reshuffled (Fisher-Yates) whenever fewer than m remain. The
                blocks of one shuffle are disjoint; the N mod m leftover
                points sit that shuffle out, as in SQuaD-MDS.
    """
    cdef int a, r
    cdef cn.int32_t t
    cdef Py_ssize_t s

    if not partition:
        for a in range(m):
            idx[a] = <int>(xorshift32(rng_state) % n_samples)
        return

    if pos[0] + m > n_samples:
        for s in range(n_samples - 1, 0, -1):
            r = <int>(xorshift32(rng_state) % <cn.uint32_t>(s + 1))
            t = perm[s]
            perm[s] = perm[r]
            perm[r] = t
        pos[0] = 0
    for a in range(m):
        idx[a] = perm[pos[0] + a]
    pos[0] += m


cdef inline void _block_deltas_generic(
    const double* X,
    Py_ssize_t n_features,
    int* idx,
    int* pair_left,
    int* pair_right,
    int n_pairs,
    double* delta,
) noexcept nogil:
    """Feature-space distances of the block's first ``n_pairs`` pairs.

    Pairs outer, features inner: every pair re-reads its two rows, but after
    the first touch those rows come from L1/L2 rather than memory. Each sum
    runs over f = 0..D-1 in order, exactly as :c:func:`_lazy_sgd_step`.

    Pairs are taken four at a time with four independent running sums. One
    sum per pair is a serial chain of D dependent additions, bounded by the
    add latency rather than by loads; four chains overlap. Each pair's sum
    is still accumulated alone and in order, so the result is bit-identical
    to one pair at a time.
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


cdef inline void _block_deltas_4(
    const double* X,
    Py_ssize_t n_features,
    int* idx,
    double* delta,
) noexcept nogil:
    """All six distances of a 4-point block in ONE pass over the features.

    The same fusion as ``_squad_mds_cython.run_squad_iteration``: four loads
    per feature feed six running sums held in registers. Pair order is the
    lexicographic one used everywhere else (01, 02, 03, 12, 13, 23), and each
    sum still accumulates over f in order, so the result is bit-identical to
    :c:func:`_block_deltas_generic`.
    """
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


def _check_blocked_args(int n_samples, int block_size, bint partition,
                        perm):
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}.")
    if block_size > n_samples:
        raise ValueError(
            f"block_size={block_size} exceeds n_samples={n_samples}."
        )
    if partition and (perm is None or perm.shape[0] != n_samples):
        raise ValueError(
            "partition sampling needs an int32 perm buffer of length "
            "n_samples."
        )


cpdef Py_ssize_t run_sgd_epoch_lazy_blocked(
    double[:, ::1] embedding,        # (n_samples, n_components), in-place
    const double[:, ::1] X_original, # (n_samples, n_features)
    Py_ssize_t n_updates,            # PAIR updates this epoch
    int block_size,                  # m: points per block
    bint partition,                  # False: random blocks; True: partition
    int update_mode,                 # 0 sequential, 1 sum, 2 mean
    double lr,
    int weighting_code,              # 1 => inverse, 0 => uniform
    cn.uint32_t seed,
    cn.int32_t[::1] perm=None,       # (n_samples,) scratch, partition only
    bint fused=True,                 # use the fused 4-point distance pass
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    """One SGD epoch with point-blocked sampling (lazy mode).

    Repeatedly draws a block of ``block_size`` points (see
    :c:func:`_next_block`), computes the feature-space distances of all
    ``P = m(m-1)/2`` pairs among them, then updates those pairs in
    lexicographic order. The budget is counted in pairs: the last block of
    the epoch processes only the ``n_updates mod P`` pairs still owed.

    ``update_mode``:

    - 0 ``sequential`` : each pair's step is applied immediately, so later
      pairs see the moved points -- the random kernel's rule.
    - 1 ``sum``        : every pair's step is computed from the block's
      starting positions, summed per point, and applied once (the gradient
      step on the block's stress).
    - 2 ``mean``       : as ``sum``, divided by the number of pairs the point
      took part in (m-1 in a full block).

    With ``block_size=2`` and random blocks, every mode is bit-identical to
    :func:`run_sgd_epoch_lazy_random_native` given the same seed.

    ``perm`` is only read in partition mode; it is shuffled in place and may
    be reused across epochs (each epoch starts with a fresh shuffle).

    Returns the number of blocks drawn.
    """
    cdef:
        int m = block_size
        int n_block_pairs = m * (m - 1) // 2
        int n_samples = X_original.shape[0]
        Py_ssize_t n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        int use_fused = fused and m == 4
        cn.uint32_t rng_state = seed if seed != 0 else 1
        Py_ssize_t done = 0
        Py_ssize_t n_blocks = 0
        Py_ssize_t pos = n_samples   # forces a shuffle on the first block
        Py_ssize_t i, j
        int n_do, p, a, b, d, q
        double ratio, grad
        double* emb = &embedding[0, 0]
        const double* X = &X_original[0, 0]
        cn.int32_t* perm_ptr = NULL
        double* x_knots_ptr = NULL
        double* y_knots_ptr = NULL
        int* idx = NULL
        int* pair_left = NULL
        int* pair_right = NULL
        int* count = NULL
        double* delta = NULL
        double* acc = NULL
        double* diff_vector = NULL

    _check_blocked_args(n_samples, m, partition, perm)
    if update_mode not in (0, 1, 2):
        raise ValueError(f"update_mode must be 0, 1 or 2, got {update_mode}.")
    if n_updates <= 0:
        return 0

    if partition:
        perm_ptr = &perm[0]
    if is_non_metric:
        n_knots = x_knots.shape[0]
        x_knots_ptr = &x_knots[0]
        y_knots_ptr = &y_knots[0]

    try:
        idx = <int*> malloc(m * sizeof(int))
        count = <int*> malloc(m * sizeof(int))
        pair_left = <int*> malloc(n_block_pairs * sizeof(int))
        pair_right = <int*> malloc(n_block_pairs * sizeof(int))
        delta = <double*> malloc(n_block_pairs * sizeof(double))
        acc = <double*> malloc(m * n_components * sizeof(double))
        diff_vector = <double*> malloc(n_components * sizeof(double))
        if (idx == NULL or count == NULL or pair_left == NULL
                or pair_right == NULL or delta == NULL or acc == NULL
                or diff_vector == NULL):
            raise MemoryError("Failed to allocate block buffers")

        # Lexicographic pair table: (0,1), (0,2), ..., (m-2, m-1).
        p = 0
        for a in range(m):
            for b in range(a + 1, m):
                pair_left[p] = a
                pair_right[p] = b
                p += 1

        with nogil:
            while done < n_updates:
                _next_block(idx, m, n_samples, partition, perm_ptr, &pos,
                            &rng_state)
                n_do = n_block_pairs
                if n_updates - done < n_do:
                    n_do = <int>(n_updates - done)

                if use_fused:
                    _block_deltas_4(X, n_features, idx, delta)
                else:
                    _block_deltas_generic(X, n_features, idx, pair_left,
                                          pair_right, n_do, delta)

                if update_mode == 0:
                    for p in range(n_do):
                        i = idx[pair_left[p]]
                        j = idx[pair_right[p]]
                        if _pair_move(emb, i, j, delta[p], n_components, lr,
                                      weighting_code, is_non_metric,
                                      x_knots_ptr, y_knots_ptr, n_knots,
                                      diff_vector, &ratio):
                            for d in range(n_components):
                                grad = ratio * diff_vector[d]
                                emb[i * n_components + d] -= grad
                                emb[j * n_components + d] += grad
                else:
                    # Every step is computed from the block's starting
                    # positions: nothing is written to emb until all n_do
                    # pairs have been evaluated.
                    memset(acc, 0, m * n_components * sizeof(double))
                    memset(count, 0, m * sizeof(int))
                    for p in range(n_do):
                        a = pair_left[p]
                        b = pair_right[p]
                        count[a] += 1
                        count[b] += 1
                        if _pair_move(emb, idx[a], idx[b], delta[p],
                                      n_components, lr, weighting_code,
                                      is_non_metric, x_knots_ptr,
                                      y_knots_ptr, n_knots, diff_vector,
                                      &ratio):
                            for d in range(n_components):
                                grad = ratio * diff_vector[d]
                                acc[a * n_components + d] -= grad
                                acc[b * n_components + d] += grad
                    for a in range(m):
                        if count[a] == 0:
                            continue
                        i = idx[a]
                        for d in range(n_components):
                            if update_mode == 2:
                                emb[i * n_components + d] += (
                                    acc[a * n_components + d] / count[a]
                                )
                            else:
                                emb[i * n_components + d] += (
                                    acc[a * n_components + d]
                                )

                done += n_do
                n_blocks += 1
    finally:
        free(idx)
        free(count)
        free(pair_left)
        free(pair_right)
        free(delta)
        free(acc)
        free(diff_vector)

    return n_blocks


def draw_blocks(int n_samples, int block_size, Py_ssize_t n_blocks,
                bint partition, cn.uint32_t seed, cn.int32_t[::1] perm=None):
    """The blocks :func:`run_sgd_epoch_lazy_blocked` would draw (test hook).

    Runs the kernel's own :c:func:`_next_block` with the same seed handling,
    so it returns exactly the first ``n_blocks`` blocks of an epoch as an
    ``(n_blocks, block_size)`` int32 array. The kernel draws nothing else
    from its generator, so this is its complete sampling sequence.
    """
    if partition and perm is None:
        perm = np.arange(n_samples, dtype=np.int32)
    _check_blocked_args(n_samples, block_size, partition, perm)

    cdef:
        cn.int32_t[:, ::1] out = np.empty((n_blocks, block_size),
                                          dtype=np.int32)
        int* idx = <int*> malloc(block_size * sizeof(int))
        cn.uint32_t rng_state = seed if seed != 0 else 1
        Py_ssize_t pos = n_samples
        Py_ssize_t k
        int a
        cn.int32_t* perm_ptr = &perm[0] if partition else NULL

    if idx == NULL:
        raise MemoryError("Failed to allocate idx")
    try:
        for k in range(n_blocks):
            _next_block(idx, block_size, n_samples, partition, perm_ptr,
                        &pos, &rng_state)
            for a in range(block_size):
                out[k, a] = idx[a]
    finally:
        free(idx)
    return np.asarray(out)
