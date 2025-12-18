#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport sqrt
import numpy as np
cimport numpy as cn

cn.import_array()

def run_sgd_epoch(
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
            
            # Gradient Clipping
            # if move_mag > 1.0:
            #    move_mag = 1.0
            # elif move_mag < -1.0:
            #    move_mag = -1.0
            
            ratio = move_mag / dist
            
            for d in range(n_components):
                diff_val = embedding[i, d] - embedding[j, d]
                embedding[i, d] -= ratio * diff_val
                embedding[j, d] += ratio * diff_val

    return np.asarray(embedding)