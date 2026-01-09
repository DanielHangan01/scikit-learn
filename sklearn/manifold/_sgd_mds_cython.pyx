#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport sqrt, pow
from libc.stdlib cimport rand, RAND_MAX
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

def run_sgd_epoch_lazy_random_native(
    double[:, ::1] embedding,      # Modified in-place
    double[:, ::1] X_original,     # Read-only features
    long n_updates,                # Number of updates to perform (e.g. n_pairs)
    double lr,
    str weighting_type,
    int seed                       # Seed for C-level rand (optional, or use Python seed)
):
    cdef:
        int k, i, j, f, d
        int n_samples = X_original.shape[0]
        int n_features = X_original.shape[1]
        int n_components = embedding.shape[1]
        
        double delta, dist, w, ratio, step_val
        double diff, grad
        double[10] diff_vector  # Buffer for component diffs (assuming low dim)

    # Note: For strict reproducibility, proper LCG needed
    # or PCG random generator here using 'seed'
    
    for k in range(n_updates):
        i = rand() % n_samples
        j = rand() % n_samples
        
        if i == j: 
            continue
            
        delta = 0.0
        for f in range(n_features):
            diff = X_original[i, f] - X_original[j, f]
            delta += diff * diff
        delta = sqrt(delta)
        
        if weighting_type == "inverse":
            if delta < 1e-6: delta = 1e-6
            w = 1.0 / (delta * delta)
        else:
            w = 1.0

        dist = 0.0
        for d in range(n_components):
            diff = embedding[i, d] - embedding[j, d]
            diff_vector[d] = diff
            dist += diff * diff
        dist = sqrt(dist)
        
        step_val = w * lr
        if step_val > 1.0: step_val = 1.0
        
        if dist < 1e-9: dist = 1e-9
        
        ratio = step_val * (dist - delta) / (2.0 * dist)
        
        for d in range(n_components):
            grad = ratio * diff_vector[d]
            embedding[i, d] -= grad
            embedding[j, d] += grad