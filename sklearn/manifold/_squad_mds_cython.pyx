# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython kernel for SQuaD-MDS (Lambert et al., 2022).

One function, :func:`run_squad_iteration`, does a whole quartet pass: for
every quartet it computes the six high-dimensional distances, normalises
them, computes the relative-distance gradient, and applies the (optionally
trust-region-limited) step. See ``_squad_mds.py`` for the objective and for
the pure-NumPy reference implementation this must agree with.

WHY THIS EXISTS. SGD-MDS's own hot loop is Cython, so timing it against a
NumPy SQuaD-MDS would compare kernels, not algorithms. Profiling the NumPy
path showed the high-dimensional distance phase taking 93% of the time at
D=784 and running ~9x slower than a single streaming pass over X -- not
because the work is large but because it materialises an ``(M, 4, D)``
gather plus six ``(M, D)`` difference temporaries, 2.2x more bytes in
temporaries than in the gather itself.

The fix is fusion, which is what a compiled kernel buys: the loop below
walks the four rows of a quartet ONCE, accumulating all six squared
distances in registers, so the traffic is one read of each point's features
per iteration and nothing else. Everything downstream (gradient, update) is
scalar arithmetic on 6 or 4*n_components values that never reaches memory.
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
cimport numpy as cn
cimport cython

cn.import_array()

# The six pairs of a quartet, in the reference implementation's order:
# ab, ac, ad, bc, bd, cd. Kept as module-level C arrays so the inner loop
# indexes them without touching Python.
cdef int[6] PAIR_L
cdef int[6] PAIR_R
PAIR_L[0] = 0; PAIR_L[1] = 0; PAIR_L[2] = 0; PAIR_L[3] = 1; PAIR_L[4] = 1; PAIR_L[5] = 2
PAIR_R[0] = 1; PAIR_R[1] = 2; PAIR_R[2] = 3; PAIR_R[3] = 2; PAIR_R[4] = 3; PAIR_R[5] = 3

cdef double EPS = 1e-12


def run_squad_iteration(
    const double[:, ::1] X,          # (n_samples, n_features), C-contiguous
    double[:, ::1] emb,              # (n_samples, n_components), updated in place
    const cn.int64_t[::1] quartets,  # (4 * n_quartets,) flat point indices
    double lr,
    bint squared_d,                  # exaggeration: skip the square root
    double max_step,                 # trust region; <= 0 disables it
    bint want_loss,
):
    """One quartet pass. Returns ``(n_clipped, loss_sum)``.

    ``loss_sum`` is the summed per-quartet cost when ``want_loss``, else 0.0;
    it is free to accumulate since the residuals are already in registers.
    """
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_components = emb.shape[1]
    cdef Py_ssize_t n_quartets = quartets.shape[0] // 4
    cdef Py_ssize_t q, d, p, slot
    cdef Py_ssize_t li, ri, pi
    cdef Py_ssize_t[4] idx

    cdef double x0, x1, x2, x3, t
    cdef double s01, s02, s03, s12, s13, s23
    cdef double hd_sum, ld_sum, c, u_p, step, nrm, scale
    cdef long n_clipped = 0
    cdef double loss_sum = 0.0
    cdef double resid

    # Scratch, allocated once per pass rather than per quartet.
    cdef double* hd = <double*> malloc(6 * sizeof(double))
    cdef double* dist = <double*> malloc(6 * sizeof(double))
    cdef double* u = <double*> malloc(6 * sizeof(double))
    cdef double* unit = <double*> malloc(6 * n_components * sizeof(double))
    cdef double* grad = <double*> malloc(4 * n_components * sizeof(double))
    cdef int[4][3] slot_pair
    cdef double[4][3] slot_sign

    if hd == NULL or dist == NULL or u == NULL or unit == NULL or grad == NULL:
        free(hd); free(dist); free(u); free(unit); free(grad)
        raise MemoryError("run_squad_iteration: scratch allocation failed")

    # For each quartet slot, the three pairs it belongs to and the sign with
    # which that pair's unit vector enters its gradient (+1 as the pair's
    # left member, -1 as its right member).
    slot_pair[0][0] = 0; slot_pair[0][1] = 1; slot_pair[0][2] = 2
    slot_sign[0][0] = 1.0; slot_sign[0][1] = 1.0; slot_sign[0][2] = 1.0
    slot_pair[1][0] = 0; slot_pair[1][1] = 3; slot_pair[1][2] = 4
    slot_sign[1][0] = -1.0; slot_sign[1][1] = 1.0; slot_sign[1][2] = 1.0
    slot_pair[2][0] = 1; slot_pair[2][1] = 3; slot_pair[2][2] = 5
    slot_sign[2][0] = -1.0; slot_sign[2][1] = -1.0; slot_sign[2][2] = 1.0
    slot_pair[3][0] = 2; slot_pair[3][1] = 4; slot_pair[3][2] = 5
    slot_sign[3][0] = -1.0; slot_sign[3][1] = -1.0; slot_sign[3][2] = -1.0

    with nogil:
        for q in range(n_quartets):
            idx[0] = quartets[4 * q]
            idx[1] = quartets[4 * q + 1]
            idx[2] = quartets[4 * q + 2]
            idx[3] = quartets[4 * q + 3]

            # ---- six high-dimensional distances, ONE pass over the four
            # rows, all accumulators in registers. This is the whole point
            # of the kernel: no (M,4,D) gather, no (M,D) temporaries.
            s01 = 0.0; s02 = 0.0; s03 = 0.0
            s12 = 0.0; s13 = 0.0; s23 = 0.0
            for d in range(n_features):
                x0 = X[idx[0], d]; x1 = X[idx[1], d]
                x2 = X[idx[2], d]; x3 = X[idx[3], d]
                t = x0 - x1; s01 += t * t
                t = x0 - x2; s02 += t * t
                t = x0 - x3; s03 += t * t
                t = x1 - x2; s12 += t * t
                t = x1 - x3; s13 += t * t
                t = x2 - x3; s23 += t * t
            hd[0] = s01; hd[1] = s02; hd[2] = s03
            hd[3] = s12; hd[4] = s13; hd[5] = s23
            if not squared_d:
                for p in range(6):
                    hd[p] = sqrt(hd[p])

            # Sum-normalise the targets. The guard matters: a quartet of
            # four high-dimensionally identical points has hd_sum == 0, and
            # the reference divides by it unguarded.
            hd_sum = 0.0
            for p in range(6):
                hd_sum += hd[p]
            if hd_sum < EPS:
                hd_sum = EPS
            for p in range(6):
                hd[p] = hd[p] / hd_sum

            # ---- six embedding distances and unit vectors ----
            ld_sum = 0.0
            for p in range(6):
                li = idx[PAIR_L[p]]
                ri = idx[PAIR_R[p]]
                t = 0.0
                for d in range(n_components):
                    x0 = emb[li, d] - emb[ri, d]
                    unit[p * n_components + d] = x0
                    t += x0 * x0
                t = sqrt(t) + EPS
                dist[p] = t
                ld_sum += t
                for d in range(n_components):
                    unit[p * n_components + d] = unit[p * n_components + d] / t

            # ---- u_p = 2 (d_hat_p - delta_hat_p) / S,  c = sum u_p d_hat_p
            c = 0.0
            for p in range(6):
                resid = dist[p] / ld_sum - hd[p]
                if want_loss:
                    loss_sum += resid * resid
                u_p = 2.0 * resid / ld_sum
                u[p] = u_p
                c += u_p * (dist[p] / ld_sum)

            # ---- gradient: part A (pairs this slot is in) minus c * G_q
            for slot in range(4):
                for d in range(n_components):
                    grad[slot * n_components + d] = 0.0
                # Part A and part B share the same three pairs and signs,
                # so they fold into a single (u_p - c) factor.
                for p in range(3):
                    pi = slot_pair[slot][p]
                    t = slot_sign[slot][p]
                    for d in range(n_components):
                        grad[slot * n_components + d] += (
                            (u[pi] - c) * t * unit[pi * n_components + d]
                        )

            # ---- apply the step, with the optional trust region ----
            for slot in range(4):
                ri = idx[slot]
                scale = lr
                if max_step > 0.0:
                    nrm = 0.0
                    for d in range(n_components):
                        t = lr * grad[slot * n_components + d]
                        nrm += t * t
                    nrm = sqrt(nrm)
                    if nrm > max_step:
                        scale = lr * (max_step / nrm)
                        n_clipped += 1
                for d in range(n_components):
                    emb[ri, d] -= scale * grad[slot * n_components + d]

    free(hd); free(dist); free(u); free(unit); free(grad)
    return n_clipped, loss_sum
