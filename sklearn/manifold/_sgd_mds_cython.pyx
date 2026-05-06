#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
cimport numpy as cn
cimport cython

cn.import_array()

cdef inline cn.uint32_t xorshift32(cn.uint32_t* state) nogil:
    cdef cn.uint32_t x = state[0]
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    state[0] = x
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double interpolate_isotonic(
    double val,
    double[::1] x_knots,
    double[::1] y_knots,
    int n_knots
) nogil:
    """
    Finds y corresponding to val using linear interpolation on sorted knots.
    Performs Binary Search.
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

cpdef void run_sgd_epoch(
    double[:, ::1] embedding,      # Shape (n_samples, n_components)
    double[::1] target_distances,  # Shape (n_pairs) - PRE-EXTRACTED
    double[::1] weights,           # Shape (n_pairs) - PRE-CALCULATED
    cn.int32_t[:, ::1] pairs,      # Shape (n_pairs, 2)
    double lr
):
    """
    Optimized SGD Epoch. 
    Requires 'target_distances' and 'weights' to be 1D arrays aligned 
    with 'pairs' to ensure cache locality.
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


cpdef void run_sgd_epoch_lazy_random_native(
    double[:, ::1] embedding,      # Modified in-place
    double[:, ::1] X_original,     # Read-only features
    long n_updates,                # Number of updates to perform (e.g. n_pairs)
    double lr,
    int weighting_code,            # 1 => inverse, 0 => uniform
    cn.uint32_t seed,              # RNG seed
    double[::1] x_knots=None,
    double[::1] y_knots=None
):
    cdef:
        long k
        int i, j, f, d
        int n_samples = X_original.shape[0]
        int n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)
        
        double delta, dist, w, ratio, step_val
        double diff, grad
        double target

        cn.uint32_t rng_state = seed if seed != 0 else 1
        double* diff_vector = <double*> malloc(n_components * sizeof(double))

    if diff_vector == NULL:
        raise MemoryError("Failed to allocate diff_vector")
    
    if is_non_metric:
        n_knots = x_knots.shape[0]
    
    try:
        with nogil:
            for k in range(n_updates):
                # Fast RNG-based sampling
                i = <int>(xorshift32(&rng_state) % n_samples)
                j = <int>(xorshift32(&rng_state) % n_samples)
                if i == j:
                    continue
                
                delta = 0.0
                for f in range(n_features):
                    diff = X_original[i, f] - X_original[j, f]
                    delta += diff * diff
                delta = sqrt(delta)
                
                if weighting_code == 1:
                    if delta < 1e-6:
                        delta = 1e-6
                    w = 1.0 / (delta * delta)
                else:
                    w = 1.0

                if is_non_metric:
                    target = interpolate_isotonic(delta, x_knots, y_knots, n_knots)
                else:
                    target = delta

                dist = 0.0
                for d in range(n_components):
                    diff = embedding[i, d] - embedding[j, d]
                    diff_vector[d] = diff
                    dist += diff * diff
                dist = sqrt(dist)

                if dist < 1e-12:
                    continue
                
                step_val = w * lr
                if step_val > 1.0:
                    step_val = 1.0
                
                ratio = step_val * (dist - target) / (2.0 * dist)
                
                for d in range(n_components):
                    grad = ratio * diff_vector[d]
                    embedding[i, d] -= grad
                    embedding[j, d] += grad

    finally:
        free(diff_vector)


cpdef void run_sgd_epoch_pivot_lazy(
    double[:, ::1] embedding,      # Modified in-place
    double[:, ::1] X_original,     # Read-only features
    cn.int32_t[::1] pivot_indices, # Shape (k,)
    long n_updates,                # Number of updates (typically k * n_samples)
    double lr,
    int weighting_code,            # 1 => inverse, 0 => uniform
    cn.uint32_t seed,              # RNG seed
    double[::1] x_knots=None,
    double[::1] y_knots=None,
):
    cdef:
        long k_iter
        int i, j, p_idx, f, d
        int n_samples = X_original.shape[0]
        int n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        int n_pivots = pivot_indices.shape[0]
        int n_knots = 0
        int is_non_metric = (x_knots is not None)

        double delta, dist, w, ratio, step_val
        double diff, grad
        double target

        cn.uint32_t rng_state = seed if seed != 0 else 1
        double* diff_vector = <double*> malloc(n_components * sizeof(double))

    if diff_vector == NULL:
        raise MemoryError("Failed to allocate diff_vector")

    if is_non_metric:
        n_knots = x_knots.shape[0]

    try:
        with nogil:
            for k_iter in range(n_updates):
                i = <int>(xorshift32(&rng_state) % n_samples)
                p_idx = <int>(xorshift32(&rng_state) % n_pivots)
                j = pivot_indices[p_idx]
                if i == j:
                    continue

                delta = 0.0
                for f in range(n_features):
                    diff = X_original[i, f] - X_original[j, f]
                    delta += diff * diff
                delta = sqrt(delta)

                if weighting_code == 1:
                    if delta < 1e-6:
                        delta = 1e-6
                    w = 1.0 / (delta * delta)
                else:
                    w = 1.0

                if is_non_metric:
                    target = interpolate_isotonic(delta, x_knots, y_knots, n_knots)
                else:
                    target = delta

                dist = 0.0
                for d in range(n_components):
                    diff = embedding[i, d] - embedding[j, d]
                    diff_vector[d] = diff
                    dist += diff * diff
                dist = sqrt(dist)

                if dist < 1e-12:
                    continue

                step_val = w * lr
                if step_val > 1.0:
                    step_val = 1.0

                ratio = step_val * (dist - target) / (2.0 * dist)

                for d in range(n_components):
                    grad = ratio * diff_vector[d]
                    embedding[i, d] -= grad
                    embedding[j, d] += grad

    finally:
        free(diff_vector)

    