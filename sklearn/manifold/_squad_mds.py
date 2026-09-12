"""
SQuaD-MDS (Lambert, de Bodt, Verleysen & Lee, 2022) -- stochastic quartet
descent MDS.

RESEARCH BASELINE ONLY. Like :mod:`sklearn.manifold._pivot_mds`, this module
exists to give the SGD-MDS work a published point of comparison; it is not
part of the upstream-facing API and is not exported from ``__init__.py``.

References
----------
.. [1] Lambert, P., de Bodt, C., Verleysen, M., & Lee, J. A. (2022).
   SQuadMDS: A lean Stochastic Quartet MDS improving global structure
   preservation in neighbor embedding like t-SNE and UMAP.
   *Neurocomputing*, 503, 17-27. doi:10.1016/j.neucom.2022.06.099
   Reference implementation: https://github.com/PierreLambert3/SQuaD-MDS

WHAT IS OPTIMIZED (this is the whole point of the baseline)
-----------------------------------------------------------
Each iteration partitions the N points into ``floor(N/4)`` disjoint random
quartets. A quartet ``Q = {i,j,k,l}`` has six pairwise distances in each
space, and its cost is the squared error between the *relative* (sum-
normalized) distances, NOT the distances themselves::

    delta_hat_p = delta_p / sum_q delta_q        (high-dimensional)
    d_hat_p     = d_p     / sum_q d_q            (embedding)

    L_Q = sum_{p in the 6 pairs} (delta_hat_p - d_hat_p)**2

Both normalizations are internal to the quartet, so ``L_Q`` is invariant to
a global rescaling of the embedding *and* to a rescaling of the quartet's
high-dimensional distances. The total loss ``sum_Q L_Q`` is therefore
**scale-free**, and the method deliberately imposes no scale constraint
(paper section 2.2). Raw stress is not optimized, not even up to a constant:
consequently any stress measured on a SQuaD-MDS embedding must be taken
after the closed-form stress-optimal rescale
``alpha = <D_emb, Delta> / <D_emb, D_emb>`` -- see
``bench_utils.stress_and_optimal_scale_chunked``.

The paper introduces the absolute-distance quartet stress first and then
replaces it with the relative form; the released code hard-codes the
relative form as ``Dhd_quartet /= np.sum(Dhd_quartet)``. Both agree.

RELATION TO THE PAPER'S TEXT
----------------------------
Two documented divergences between the paper and the authors' released
Python implementation, both resolved here in favour of the released code
(which is what the published numbers were produced with):

1. **Learning-rate decay.** The paper (section 2.1) states a Robbins-Monro
   ``(a*t + b)**-1`` decay. The released code uses a geometric decay,
   ``lr *= exp(log(1e-3)/n_iter)`` each iteration, i.e. the learning rate
   ends at ``1e-3`` times its initial value. ``lr_decay='geometric'``
   (default) reproduces the code; ``'robbins_monro'`` is provided so the
   paper's variant can be measured.
2. **Momentum.** The paper describes Nesterov momentum. The released
   standalone Python implementation applies plain gradient steps with no
   momentum at all (``X_LD -= LR * grad_acc``). ``momentum=0.0`` (default)
   reproduces the code; a positive value enables Nesterov momentum.

A third feature is code-only and absent from the paper: ``exaggerate_d``
uses *squared* high-dimensional distances for the first
``stop_exaggeration`` fraction of the iterations. The reference ``main.py``
turns it on by default, so it is swept here rather than assumed.

DUPLICATE POINTS BLOW THIS METHOD UP (``max_step_frac``)
--------------------------------------------------------
The per-quartet gradient carries a factor ``2 (d_hat - delta_hat) / S``
where ``S`` is the sum of the quartet's six *embedding* distances. Since
``|d_hat - delta_hat| <= 1``, the step is ``O(lr / S)`` -- unbounded as a
quartet collapses.

The failure needs a quartet whose four points are all *identical in the
high-dimensional space*. Such points start at the same embedding
coordinate, so they are individually harmless; but each copy lands in a
different quartet each iteration and so receives a different gradient,
which drifts them apart by ~1e-9 within a few iterations. A quartet of four
of them then has ``S`` ~ 2e-8 while its six unit vectors are perfectly
well-formed, giving ``|u| <= 2/S`` ~ 9e7, a step of ~1e10, and those points
leave the embedding permanently -- from far away every quartet containing
them has a large ``S`` and hence a negligible gradient. This was observed
directly: on ``orl`` the first such event occurs at iteration 3.

Whether it happens at all is governed by the multiplicity of the most
repeated row, since the expected number of collapsed quartets per iteration
is ``(N/4) * sum_v m_v(m_v-1)(m_v-2)(m_v-3) / N(N-1)(N-2)(N-3)``. Across
this benchmark collection that is 65.5 for ``orl`` (whose 400 rows contain
one value repeated **360** times), 0.12 for ``sentiment``, 0.008 for
``hatespeech``, and effectively zero everywhere else. So this is one
pathological data set rather than a general hazard -- but on that data set,
unguarded SQuaD-MDS reaches ~440x full-cycle SGD stress, a number about
float behaviour rather than about distance preservation.

``max_step_frac`` caps each point's per-iteration displacement at that
multiple of the embedding's current RMS radius, rescaling an over-long step
without changing its direction. It is a trust region, not a change of
objective: it alters step *length* only, and cannot move the optimum. The
default of 1.0 still permits a point to cross the entire embedding in one
iteration, and is inert on clean data -- the largest legitimate step
observed on coil20 and fashion_mnist is 0.65x the RMS radius, against 1e9x
on orl. ``info['n_clipped']`` reports how often it fired, so any run can be
checked for inertness rather than trusted; pass ``None`` to disable it and
reproduce the reference exactly.

The cap is a real hyperparameter and is swept, not assumed, in Experiment
M: 4.0 leaves clean data untouched while still removing the rare
catastrophic step (on ``cnae9`` it takes the median from 2.42x to 2.12x
cycle stress and collapses the seed spread from 2.42-4.30 to 2.12-2.13, at
a 0.008% clip rate), whereas 1.0 starts shortening legitimate steps and
costs quality there. Do not treat any single value as neutral.

The RMS radius is a sound scale reference because the quartet gradient
conserves the centroid exactly: every pair contributes ``+u*e`` to one slot
and ``-u*e`` to another, so each quartet's four gradients sum to zero
(verified to ~1e-15 in the tests).

IMPLEMENTATION NOTE
-------------------
The reference is a scalar numba kernel over one quartet at a time. This is
a pure-NumPy port vectorized *across quartets*: every array below carries a
leading axis of length ``n_quartets = N // 4``. The gradient is algebraically
identical (verified against the reference kernel to ~1e-12 in
``tests/test_squad_mds.py``), and unlike the reference it is not restricted
to 2 output dimensions -- the derivation never uses the embedding's
dimensionality.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

# The six pairs of a quartet (a,b,c,d), in the reference implementation's
# order: ab, ac, ad, bc, bd, cd.
_PAIR_LEFT = (0, 0, 0, 1, 1, 2)
_PAIR_RIGHT = (1, 2, 3, 2, 3, 3)

# For each quartet slot q, the pairs it belongs to and the sign with which
# the unit vector of that pair enters q's gradient (+1 when q is the pair's
# left member, -1 when it is the right member). Derived from the tuples
# above; written out so the gradient is a handful of array adds.
_SLOT_PAIRS = (
    ((0, +1), (1, +1), (2, +1)),   # a: ab, ac, ad
    ((0, -1), (3, +1), (4, +1)),   # b: ab, bc, bd
    ((1, -1), (3, -1), (5, +1)),   # c: ac, bc, cd
    ((2, -1), (4, -1), (5, -1)),   # d: ad, bd, cd
)

_EPS = 1e-12

# Target size, in bytes, of the transient (block, 4, n_features) gather used
# when computing high-dimensional quartet distances. Keeps peak memory flat
# in D: at D=3072 this is ~1300 quartets per block, at D=3 the whole array.
_HD_BLOCK_BYTES = 128 * 1024 * 1024


def _hd_block_size(n_features):
    """Quartets per block so the (block, 4, D) gather stays ~_HD_BLOCK_BYTES."""
    per_quartet = 4 * max(int(n_features), 1) * 8
    return int(np.clip(_HD_BLOCK_BYTES // per_quartet, 256, 1 << 20))


def _quartet_hd_distances(X, quartets, squared, block_size, out):
    """Six high-dimensional distances per quartet, into ``out`` (M, 6).

    ``squared=True`` leaves them as squared distances -- the reference's
    "distance exaggeration", which skips the square root rather than
    squaring afterwards (identical, but that is how the code reads).
    """
    n_quartets = quartets.shape[0]
    for start in range(0, n_quartets, block_size):
        stop = min(start + block_size, n_quartets)
        pts = X[quartets[start:stop]]          # (b, 4, D)
        for p, (left, right) in enumerate(zip(_PAIR_LEFT, _PAIR_RIGHT)):
            diff = pts[:, left] - pts[:, right]
            out[start:stop, p] = np.einsum("ij,ij->i", diff, diff)
    if not squared:
        np.sqrt(out, out=out)
    return out


def _relative_rbf(d_hd):
    """The paper's optional nonlinear distance transform, per quartet.

    ``1 - exp(-(d - min d) / (2 * std d))``, then sum-normalized. This is a
    *non-metric* variant: it deliberately distorts the target distances, so
    it is not expected to do well on raw stress. Provided because the
    reference exposes it as ``metric='relative rbf distance'``.
    """
    lo = d_hd.min(axis=1, keepdims=True)
    sd = d_hd.std(axis=1, keepdims=True)
    rel = 1.0 - np.exp((d_hd - lo) / (-2.0 * np.maximum(sd, _EPS)))
    rel /= np.maximum(rel.sum(axis=1, keepdims=True), _EPS)
    return rel


def _quartet_gradients(pts, target, out_loss=None):
    """Gradient of the relative-distance quartet cost, vectorized.

    Parameters
    ----------
    pts : ndarray of shape (n_quartets, 4, n_components)
        Current embedding coordinates of each quartet's four points.
    target : ndarray of shape (n_quartets, 6)
        Sum-normalized high-dimensional distances (``delta_hat``).
    out_loss : list, optional
        If given, the mean per-quartet cost is appended to it.

    Returns
    -------
    grad : ndarray of shape (n_quartets, 4, n_components)

    Notes
    -----
    Writing ``S = sum_p d_p``, ``d_hat_p = d_p / S``, ``e_p`` for the unit
    vector from the pair's left point to its right point, and
    ``u_p = 2 (d_hat_p - delta_hat_p) / S``, the paper's per-term gradient
    collapses over the six terms to

        grad_q = sum_{p containing q} sign(p, q) * u_p * e_p
                 - (sum_p u_p * d_hat_p) * G_q

    with ``G_q = sum_{b != q in the quartet} (x_q - x_b) / d_qb``. The first
    term is the paper's part A (a direct pull along the pairs q takes part
    in); the second is part B, the shared denominator's effect, which is why
    every point in a quartet feels every pair.
    """
    diff = pts[:, _PAIR_LEFT, :] - pts[:, _PAIR_RIGHT, :]      # (M, 6, C)
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) + _EPS    # (M, 6)
    unit = diff / d[:, :, None]                                 # (M, 6, C)

    s = d.sum(axis=1)                                           # (M,)
    d_hat = d / s[:, None]
    resid = d_hat - target                                      # (M, 6)
    if out_loss is not None:
        out_loss.append(float(np.einsum("ij,ij->", resid, resid) / len(resid)))

    u = (2.0 / s)[:, None] * resid                              # (M, 6)
    c = np.einsum("ij,ij->i", u, d_hat)                         # (M,)
    weighted = u[:, :, None] * unit                             # (M, 6, C)

    grad = np.empty_like(pts)
    for slot, pairs in enumerate(_SLOT_PAIRS):
        (p0, s0), (p1, s1), (p2, s2) = pairs
        # part A: pulls along the three pairs this slot belongs to
        acc = s0 * weighted[:, p0] + s1 * weighted[:, p1] + s2 * weighted[:, p2]
        # part B: -c * G_q, and G_q is the same three unit vectors, unweighted
        g = s0 * unit[:, p0] + s1 * unit[:, p1] + s2 * unit[:, p2]
        grad[:, slot] = acc - c[:, None] * g
    return grad


def _init_embedding(X, n_components, init, random_state, init_std):
    """PCA-whitened (reference default) or random init, scaled to ``init_std``.

    The reference hard-codes ``PCA(whiten=True)`` followed by
    ``Xld *= 10/np.std(Xld)``, and its recommended learning-rate range
    (50-1500) is calibrated to exactly that scale. Random init is scaled the
    same way so that the same learning rates remain meaningful.
    """
    n_samples = X.shape[0]
    if init == "pca":
        from sklearn.decomposition import PCA

        emb = PCA(
            n_components=n_components, whiten=True, copy=True,
            random_state=random_state,
        ).fit_transform(X)
        emb = np.ascontiguousarray(emb, dtype=np.float64)
    elif init == "random":
        emb = random_state.normal(size=(n_samples, n_components))
    else:
        raise ValueError(f"init must be 'pca' or 'random' (got {init!r}).")

    sd = float(np.std(emb))
    if sd > 0:
        emb *= init_std / sd
    return emb


def squad_mds(
    X,
    n_components=2,
    *,
    n_iter=1000,
    lr=550.0,
    exaggerate_d=True,
    stop_exaggeration=0.6,
    metric="euclidean",
    init="pca",
    init_std=10.0,
    lr_decay="geometric",
    final_lr_ratio=1e-3,
    momentum=0.0,
    max_step_frac=1.0,
    backend="cython",
    random_state=None,
    record_loss=False,
):
    """Embed ``X`` with SQuaD-MDS (Lambert et al., 2022).

    Minimizes the per-quartet *relative*-distance cost described in this
    module's docstring by stochastic gradient descent over random disjoint
    quartets. Each iteration costs ``O(N * n_features)`` time and the method
    never materializes a distance matrix, so memory is ``O(N * n_features)``.

    The returned embedding has **no meaningful global scale** (the cost is
    scale-free); rescale before computing stress.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix. Only Euclidean feature input is supported -- the
        method's whole advantage is never forming a distance matrix, so a
        ``'precomputed'`` mode would defeat its purpose.
    n_components : int, default=2
        Output dimensionality.
    n_iter : int, default=1000
        Number of quartet passes. The reference's default; its ``main.py``
        notes 1000 "is plenty if initialised with PCA".
    lr : float, default=550.0
        Initial learning rate, calibrated to ``init_std=10``. The reference
        reports 50-1500 as a reasonable range at that scale.
    exaggerate_d : bool, default=True
        Use squared high-dimensional distances for the first
        ``stop_exaggeration`` fraction of iterations (reference default).
    stop_exaggeration : float, default=0.6
        Fraction of ``n_iter`` after which exaggeration stops.
    metric : {'euclidean', 'relative_rbf'}, default='euclidean'
        Target distances. ``'relative_rbf'`` applies the paper's nonlinear
        transform and is non-metric.
    init : {'pca', 'random'}, default='pca'
        Initialization. ``'pca'`` is the reference's choice.
    init_std : float, default=10.0
        Standard deviation the initial embedding is scaled to.
    lr_decay : {'geometric', 'robbins_monro'}, default='geometric'
        ``'geometric'`` reproduces the released code; ``'robbins_monro'``
        follows the paper's stated ``(a*t + b)**-1``. Both reach
        ``final_lr_ratio * lr`` at the last iteration.
    final_lr_ratio : float, default=1e-3
        Learning rate at the final iteration, as a fraction of ``lr``.
    momentum : float, default=0.0
        Nesterov momentum coefficient. ``0.0`` reproduces the released code;
        the paper describes a positive value.
    backend : {'cython', 'numpy'}, default='cython'
        Which kernel runs the quartet pass. ``'cython'`` fuses the whole
        pass (high-dimensional distances, gradient, update) into one loop
        that keeps every intermediate in registers; ``'numpy'`` is the
        vectorised reference implementation. They agree to floating-point
        rounding -- the equivalence is pinned in ``tests/test_squad_mds.py``
        -- and exist as two backends only so the comparison against
        SGD-MDS, whose own hot loop is Cython, is kernel-for-kernel fair.
        ``momentum != 0`` and ``metric != 'euclidean'`` fall back to
        ``'numpy'``.
    max_step_frac : float or None, default=1.0
        Trust region: no point may move more than this multiple of the
        embedding's RMS radius in one iteration. Guards the ``O(lr / S)``
        blow-up on data containing duplicate points -- see this module's
        docstring. ``None`` disables it, reproducing the reference exactly.
        Check ``info['n_clipped']`` to confirm it never fired.
    random_state : int, RandomState instance or None, default=None
        Controls the initialization and the per-iteration quartet shuffle.
    record_loss : bool, default=False
        Accumulate the mean per-quartet cost at each iteration into the
        returned ``info['loss_history']``. Costs nothing beyond one
        reduction over an array already in cache.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
    info : dict
        ``n_iter``, ``n_quartets``, ``n_unused`` (points dropped by the
        ``N % 4`` truncation each iteration), ``n_clipped`` (point-steps the
        trust region shortened, out of ``n_iter * 4 * n_quartets``), and
        ``loss_history``.
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if n_samples < 4:
        raise ValueError(
            f"SQuaD-MDS needs at least 4 samples to form a quartet "
            f"(got {n_samples})."
        )
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1 (got {n_iter}).")
    if metric not in ("euclidean", "relative_rbf"):
        raise ValueError(
            f"metric must be 'euclidean' or 'relative_rbf' (got {metric!r})."
        )
    if lr_decay not in ("geometric", "robbins_monro"):
        raise ValueError(
            f"lr_decay must be 'geometric' or 'robbins_monro' "
            f"(got {lr_decay!r})."
        )

    emb = _init_embedding(X, n_components, init, rng, init_std)

    n_quartets = n_samples // 4
    n_used = n_quartets * 4
    perm = np.arange(n_samples)
    block = _hd_block_size(n_features)
    d_hd = np.empty((n_quartets, 6), dtype=np.float64)

    # Learning-rate schedules, both normalized to end at final_lr_ratio * lr.
    if lr_decay == "geometric":
        # Reference: LR *= decay at the TOP of each iteration, so iteration t
        # (0-based) uses lr * decay**(t+1) and the last uses lr * ratio.
        decay = np.exp(np.log(final_lr_ratio) / n_iter)
        lr_schedule = lr * decay ** np.arange(1, n_iter + 1)
    else:
        # (a*t + b)**-1 with b = 1/lr and a fixed so the final value matches.
        b = 1.0 / lr
        a = b * (1.0 / final_lr_ratio - 1.0) / max(n_iter - 1, 1)
        lr_schedule = 1.0 / (a * np.arange(n_iter) + b)

    stop_exa = int(n_iter * stop_exaggeration) if exaggerate_d else 0
    velocity = np.zeros_like(emb) if momentum else None
    loss_history = [] if record_loss else None
    n_clipped = 0

    def trust_region(delta):
        """Shorten over-long point steps, keeping their direction."""
        nonlocal n_clipped
        if max_step_frac is None:
            return delta
        radius = np.sqrt((emb ** 2).sum(axis=1).mean())
        cap = max_step_frac * max(radius, _EPS)
        norms = np.sqrt((delta ** 2).sum(axis=1))
        over = norms > cap
        n_over = int(over.sum())
        if n_over:
            n_clipped += n_over
            delta[over] *= (cap / norms[over])[:, None]
        return delta

    # The kernel hard-codes the euclidean sum-normalised target and takes no
    # momentum, so both fall back to the NumPy path rather than being
    # silently ignored.
    use_cython = (
        backend == "cython" and not momentum and metric == "euclidean"
    )
    if backend not in ("cython", "numpy"):
        raise ValueError(
            f"backend must be 'cython' or 'numpy' (got {backend!r})."
        )
    if use_cython:
        from sklearn.manifold._squad_mds_cython import run_squad_iteration

        X = np.ascontiguousarray(X, dtype=np.float64)
        emb = np.ascontiguousarray(emb, dtype=np.float64)

    for t in range(n_iter):
        step = lr_schedule[t]
        squared = exaggerate_d and t < stop_exa

        rng.shuffle(perm)
        quartets = perm[:n_used].reshape(n_quartets, 4)

        if use_cython:
            # The trust region's cap is the embedding's RMS radius at the
            # START of the pass, matching the NumPy path. Quartets are
            # disjoint, so the kernel may update points as it goes without
            # any quartet observing another's write.
            cap = -1.0
            if max_step_frac is not None:
                radius = np.sqrt((emb ** 2).sum(axis=1).mean())
                cap = max_step_frac * max(radius, _EPS)
            clipped, loss = run_squad_iteration(
                X, emb, np.ascontiguousarray(quartets.ravel(), dtype=np.int64),
                float(step), bool(squared), float(cap), bool(record_loss),
            )
            n_clipped += clipped
            if record_loss:
                loss_history.append(loss / n_quartets)
            continue

        _quartet_hd_distances(X, quartets, squared, block, d_hd)
        if metric == "relative_rbf":
            target = _relative_rbf(d_hd)
        else:
            target = d_hd / np.maximum(
                d_hd.sum(axis=1, keepdims=True), _EPS
            )

        flat = quartets.ravel()
        if momentum:
            # Nesterov: evaluate the gradient at the look-ahead point.
            look = emb + momentum * velocity
            grad = _quartet_gradients(look[quartets], target, loss_history)
            velocity *= momentum
            velocity[flat] -= trust_region(
                step * grad.reshape(-1, n_components)
            )
            emb += velocity
        else:
            grad = _quartet_gradients(emb[quartets], target, loss_history)
            # Quartets are disjoint within an iteration, so the points a
            # quartet touches are unique and this is a scatter, not an
            # accumulate. Points beyond n_used simply get no update this
            # iteration -- the reference behaves the same way.
            emb[flat] -= trust_region(step * grad.reshape(-1, n_components))

    return emb, {
        "n_iter": int(n_iter),
        "n_quartets": int(n_quartets),
        "n_unused": int(n_samples - n_used),
        "n_clipped": int(n_clipped),
        "n_point_steps": int(n_iter * n_used),
        "loss_history": loss_history,
    }
