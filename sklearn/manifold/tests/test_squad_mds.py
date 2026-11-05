"""Tests for the standalone SQuaD-MDS (Lambert et al., 2022) function.

The load-bearing test here is
:func:`test_gradient_matches_reference_implementation`. Our implementation is
vectorized across quartets while the authors' released code is a scalar numba
kernel over one quartet at a time, so the two look nothing alike; the oracle
below is that kernel, transcribed verbatim (numba stripped), and it pins the
port to the published algorithm. This plays the same role Experiment L's
"Pivot MDS reduces to ClassicalMDS at k=N" check played for that baseline.
"""
import numpy as np
import pytest
from sklearn.datasets import make_s_curve
from sklearn.manifold._squad_mds import (
    _PAIR_LEFT,
    _PAIR_RIGHT,
    _quartet_gradients,
    squad_mds,
)
from sklearn.metrics import euclidean_distances


@pytest.fixture(scope="module")
def small_X():
    X, _ = make_s_curve(n_samples=200, random_state=0)
    return np.ascontiguousarray(X, dtype=np.float64)


def _stress(embedding, dissimilarity_matrix):
    D_emb = euclidean_distances(embedding, embedding)
    return 0.5 * float(np.sum((D_emb - dissimilarity_matrix) ** 2))


def _optimal_scale(embedding, dissimilarity_matrix):
    """alpha = <D_emb, D> / <D_emb, D_emb>, the stress-optimal rescale."""
    D_emb = euclidean_distances(embedding, embedding)
    return float(np.sum(D_emb * dissimilarity_matrix) / np.sum(D_emb ** 2))


# ---------------------------------------------------------------------------
# Reference oracle: the authors' quartet gradient kernel, transcribed from
# https://github.com/PierreLambert3/SQuaD-MDS (SQuaD MDS/gradients.py) with
# the @numba.jit decorators removed and nothing else changed.
# ---------------------------------------------------------------------------

def _ref_ABCD_grad(xa, ya, xb, yb, xc, yc, xd, yd,
                   dab, dac, dad, dbc, dbd, dcd, pab):
    sum_dist = dab + dac + dad + dbc + dbd + dcd
    dr_ab = (dab / sum_dist)
    f = 2 * ((pab - dr_ab) / sum_dist)
    r = (dab / sum_dist)

    gxA = f * (r * ((xa-xb)/dab + (xa-xc)/dac + (xa-xd)/dad) - (xa-xb)/dab)
    gyA = f * (r * ((ya-yb)/dab + (ya-yc)/dac + (ya-yd)/dad) - (ya-yb)/dab)
    gxB = f * (r * ((xb-xa)/dab + (xb-xc)/dbc + (xb-xd)/dbd) - (xb-xa)/dab)
    gyB = f * (r * ((yb-ya)/dab + (yb-yc)/dbc + (yb-yd)/dbd) - (yb-ya)/dab)
    gxC = f * (r * ((xc-xa)/dac + (xc-xb)/dbc + (xc-xd)/dcd))
    gyC = f * (r * ((yc-ya)/dac + (yc-yb)/dbc + (yc-yd)/dcd))
    gxD = f * (r * ((xd-xa)/dad + (xd-xb)/dbd + (xd-xc)/dcd))
    gyD = f * (r * ((yd-ya)/dad + (yd-yb)/dbd + (yd-yc)/dcd))
    return gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD


def _ref_quartet_grads(points, Dhd):
    xa, ya = points[0]
    xb, yb = points[1]
    xc, yc = points[2]
    xd, yd = points[3]

    d_ab = np.sqrt((xa-xb)**2 + (ya-yb)**2) + 1e-12
    d_ac = np.sqrt((xa-xc)**2 + (ya-yc)**2) + 1e-12
    d_ad = np.sqrt((xa-xd)**2 + (ya-yd)**2) + 1e-12
    d_bc = np.sqrt((xb-xc)**2 + (yb-yc)**2) + 1e-12
    d_bd = np.sqrt((xb-xd)**2 + (yb-yd)**2) + 1e-12
    d_cd = np.sqrt((xc-xd)**2 + (yc-yd)**2) + 1e-12

    pab, pac, pad, pbc, pbd, pcd = Dhd

    A1 = _ref_ABCD_grad(xa, ya, xb, yb, xc, yc, xd, yd,
                        d_ab, d_ac, d_ad, d_bc, d_bd, d_cd, pab)
    A2 = _ref_ABCD_grad(xa, ya, xc, yc, xb, yb, xd, yd,
                        d_ac, d_ab, d_ad, d_bc, d_cd, d_bd, pac)
    A3 = _ref_ABCD_grad(xa, ya, xd, yd, xc, yc, xb, yb,
                        d_ad, d_ac, d_ab, d_cd, d_bd, d_bc, pad)
    A4 = _ref_ABCD_grad(xb, yb, xc, yc, xa, ya, xd, yd,
                        d_bc, d_ab, d_bd, d_ac, d_cd, d_ad, pbc)
    A5 = _ref_ABCD_grad(xb, yb, xd, yd, xa, ya, xc, yc,
                        d_bd, d_ab, d_bc, d_ad, d_cd, d_ac, pbd)
    A6 = _ref_ABCD_grad(xc, yc, xd, yd, xa, ya, xb, yb,
                        d_cd, d_ac, d_bc, d_ad, d_bd, d_ab, pcd)

    # The permuted calls return their eight values in the order of the points
    # they were GIVEN, so each result is unpacked back onto (a, b, c, d).
    out = np.zeros((4, 2))
    for res, order in (
        (A1, (0, 1, 2, 3)), (A2, (0, 2, 1, 3)), (A3, (0, 3, 2, 1)),
        (A4, (1, 2, 0, 3)), (A5, (1, 3, 0, 2)), (A6, (2, 3, 0, 1)),
    ):
        for slot, point in enumerate(order):
            out[point, 0] += res[2 * slot]
            out[point, 1] += res[2 * slot + 1]
    return out


# ---------------------------------------------------------------------------
# The port reproduces the published algorithm
# ---------------------------------------------------------------------------

def test_gradient_matches_reference_implementation():
    """Vectorized gradient == the authors' scalar kernel, to float precision."""
    rng = np.random.RandomState(0)
    n_quartets = 64
    pts = rng.normal(size=(n_quartets, 4, 2)) * 7.0
    dhd = rng.uniform(0.1, 5.0, size=(n_quartets, 6))
    target = dhd / dhd.sum(axis=1, keepdims=True)

    mine = _quartet_gradients(pts, target)
    theirs = np.stack([_ref_quartet_grads(pts[m], target[m])
                       for m in range(n_quartets)])

    assert np.abs(mine - theirs).max() / np.abs(theirs).max() < 1e-12


def test_gradient_matches_finite_differences():
    """The gradient is the gradient of the RELATIVE-distance quartet cost.

    This is the claim the whole baseline rests on: the optimized loss is
    ``sum_p (delta_p/sum(delta) - d_p/sum(d))**2``, not stress over the
    quartet. Differentiating that loss numerically must reproduce the
    analytic gradient.
    """
    rng = np.random.RandomState(1)
    pts = rng.normal(size=(6, 4, 2)) * 3.0
    dhd = rng.uniform(0.5, 4.0, size=(6, 6))
    target = dhd / dhd.sum(axis=1, keepdims=True)

    def relative_quartet_loss(p):
        diff = p[:, _PAIR_LEFT, :] - p[:, _PAIR_RIGHT, :]
        d = np.sqrt((diff ** 2).sum(-1)) + 1e-12
        return float((((d / d.sum(axis=1, keepdims=True)) - target) ** 2).sum())

    analytic = _quartet_gradients(pts, target)
    numeric = np.zeros_like(analytic)
    h = 1e-6
    for m in range(pts.shape[0]):
        for q in range(4):
            for c in range(2):
                up, dn = pts.copy(), pts.copy()
                up[m, q, c] += h
                dn[m, q, c] -= h
                numeric[m, q, c] = (
                    relative_quartet_loss(up) - relative_quartet_loss(dn)
                ) / (2 * h)

    assert np.abs(analytic - numeric).max() / np.abs(numeric).max() < 1e-6


def test_cost_is_scale_free():
    """Rescaling the embedding leaves the quartet cost unchanged.

    The reason every stress measured on a SQuaD-MDS embedding must be taken
    after a stress-optimal rescale: the optimizer cannot see global scale at
    all, so it never fixes one.
    """
    rng = np.random.RandomState(2)
    pts = rng.normal(size=(32, 4, 2))
    dhd = rng.uniform(0.1, 5.0, size=(32, 6))
    target = dhd / dhd.sum(axis=1, keepdims=True)

    losses = []
    for factor in (0.01, 1.0, 100.0):
        out = []
        _quartet_gradients(pts * factor, target, out_loss=out)
        losses.append(out[0])

    assert np.allclose(losses, losses[0], rtol=1e-9)


def test_target_normalization_is_scale_free_in_hd():
    """Scaling X leaves the optimization path identical (up to the init)."""
    X, _ = make_s_curve(n_samples=120, random_state=3)
    X = np.ascontiguousarray(X, dtype=np.float64)

    a, _ = squad_mds(X, n_iter=40, init="random", random_state=0)
    b, _ = squad_mds(X * 1000.0, n_iter=40, init="random", random_state=0)

    # The init is scaled to a fixed std either way, and the targets are
    # sum-normalized, so the two runs coincide exactly.
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-9


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_components", [2, 3])
def test_output_shape_and_finite(small_X, n_components):
    emb, info = squad_mds(small_X, n_components=n_components, n_iter=50,
                          random_state=0)
    assert emb.shape == (small_X.shape[0], n_components)
    assert np.isfinite(emb).all()
    assert info["n_quartets"] == small_X.shape[0] // 4


@pytest.mark.parametrize("exaggerate_d", [False, True])
@pytest.mark.parametrize("max_step_frac", [None, 1.0])
def test_cython_and_numpy_backends_agree(small_X, exaggerate_d, max_step_frac):
    """The two kernels must be the same algorithm, not merely similar.

    SGD-MDS's hot loop is Cython, so SQuaD-MDS gets a Cython kernel too --
    otherwise a wall-clock comparison measures kernels rather than methods.
    The NumPy path stays as the readable reference, and this pins the fast
    path to it.
    """
    a, ia = squad_mds(small_X, n_iter=250, exaggerate_d=exaggerate_d,
                      max_step_frac=max_step_frac, backend="numpy",
                      random_state=0)
    b, ib = squad_mds(small_X, n_iter=250, exaggerate_d=exaggerate_d,
                      max_step_frac=max_step_frac, backend="cython",
                      random_state=0)

    assert np.abs(a - b).max() / np.abs(a).max() < 1e-10
    assert ia["n_clipped"] == ib["n_clipped"]


def test_backend_loss_histories_agree(small_X):
    _, ia = squad_mds(small_X, n_iter=100, exaggerate_d=False,
                      backend="numpy", random_state=0, record_loss=True)
    _, ib = squad_mds(small_X, n_iter=100, exaggerate_d=False,
                      backend="cython", random_state=0, record_loss=True)
    assert np.allclose(ia["loss_history"], ib["loss_history"], rtol=1e-9)


def test_bad_backend_raises(small_X):
    with pytest.raises(ValueError, match="backend"):
        squad_mds(small_X, n_iter=5, backend="numba")


def test_deterministic_given_random_state(small_X):
    a, _ = squad_mds(small_X, n_iter=60, random_state=7)
    b, _ = squad_mds(small_X, n_iter=60, random_state=7)
    assert np.array_equal(a, b)


def test_different_seeds_differ(small_X):
    a, _ = squad_mds(small_X, n_iter=60, init="random", random_state=7)
    b, _ = squad_mds(small_X, n_iter=60, init="random", random_state=8)
    assert not np.allclose(a, b)


def test_loss_decreases(small_X):
    """Mean quartet cost falls over the run (exaggeration off, one objective).

    With exaggeration on, the target switches from squared to plain
    distances partway through, so the loss history is NOT comparable across
    that boundary -- see ``benchmarks/validate_squad_mds.py``.
    """
    _, info = squad_mds(small_X, n_iter=600, exaggerate_d=False,
                        random_state=0, record_loss=True)
    hist = np.asarray(info["loss_history"])
    assert hist[:50].mean() > hist[-50:].mean()


def test_reduces_stress_relative_to_its_own_init(small_X):
    """After the stress-optimal rescale, the run improves on PCA init."""
    D = euclidean_distances(small_X)
    from sklearn.decomposition import PCA

    init = PCA(n_components=2, whiten=True).fit_transform(small_X)
    init = init * (10.0 / np.std(init))
    emb, _ = squad_mds(small_X, n_iter=800, exaggerate_d=False, random_state=0)

    s_init = _stress(init * _optimal_scale(init, D), D)
    s_out = _stress(emb * _optimal_scale(emb, D), D)
    assert s_out < s_init


def test_quartet_gradient_conserves_the_centroid():
    """Each quartet's four gradients sum to zero.

    Every pair contributes ``+u*e`` to one slot and ``-u*e`` to the other, so
    the embedding's centroid never moves. This is what makes the RMS radius a
    sound scale reference for the ``max_step_frac`` trust region.
    """
    rng = np.random.RandomState(4)
    pts = rng.normal(size=(50, 4, 2)) * 5.0
    dhd = rng.uniform(0.1, 5.0, size=(50, 6))
    grad = _quartet_gradients(pts, dhd / dhd.sum(axis=1, keepdims=True))

    per_quartet = np.abs(grad.sum(axis=1)).max()
    assert per_quartet < 1e-12 * max(np.abs(grad).max(), 1.0)


def test_trust_region_is_inert_on_clean_data(small_X):
    """On data without duplicates the guard never fires, so it changes nothing.

    This is what lets Experiment M sweep ``max_step_frac`` without the guard
    quietly becoming part of the algorithm being measured.
    """
    guarded, info = squad_mds(small_X, n_iter=300, exaggerate_d=False,
                              max_step_frac=4.0, random_state=0)
    exact, _ = squad_mds(small_X, n_iter=300, exaggerate_d=False,
                         max_step_frac=None, random_state=0)

    assert info["n_clipped"] == 0
    assert np.array_equal(guarded, exact)


def test_trust_region_bounds_the_blow_up_on_a_duplicate_cluster():
    """A large identical-point cluster explodes unguarded; the guard stops it.

    Reproduces ``orl``, whose 400 rows contain one value repeated 360 times.
    Four high-dimensionally identical points in one quartet start at the same
    embedding coordinate, drift apart by ~1e-9 because each copy lands in a
    different quartet, and then produce ``S`` ~ 1e-8 with well-formed unit
    vectors -- a step of ~1e10.

    Note the trigger is one *high-multiplicity* cluster, not duplicates in
    general: the expected number of collapsed quartets per iteration goes as
    the fourth power of the multiplicity, so many small duplicate groups
    (as in ``spambase`` or ``cnae9``) never trip it.
    """
    rng = np.random.RandomState(5)
    X = np.vstack([np.repeat(rng.normal(size=(1, 6)), 360, axis=0),
                   rng.normal(size=(40, 6))])

    loose, info_loose = squad_mds(X, n_iter=300, exaggerate_d=False,
                                  max_step_frac=None, random_state=0)
    tight, info_tight = squad_mds(X, n_iter=300, exaggerate_d=False,
                                  max_step_frac=1.0, random_state=0)

    assert info_loose["n_clipped"] == 0
    assert info_tight["n_clipped"] > 0
    # Unguarded, the embedding leaves any sane scale; guarded, it does not.
    assert np.abs(loose).max() > 1e4
    assert np.abs(tight).max() < 1e3


def test_n_unused_tracks_the_mod_4_truncation():
    rng = np.random.RandomState(0)
    for n in (100, 101, 102, 103):
        X = rng.normal(size=(n, 5))
        _, info = squad_mds(X, n_iter=3, random_state=0)
        assert info["n_unused"] == n % 4
        assert info["n_quartets"] == n // 4


def test_momentum_and_robbins_monro_run(small_X):
    """The paper's two documented divergences from the released code."""
    a, _ = squad_mds(small_X, n_iter=50, momentum=0.6, random_state=0)
    b, _ = squad_mds(small_X, n_iter=50, lr_decay="robbins_monro",
                     random_state=0)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_relative_rbf_metric_runs(small_X):
    emb, _ = squad_mds(small_X, n_iter=50, metric="relative_rbf",
                       random_state=0)
    assert np.isfinite(emb).all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 4 samples"):
        squad_mds(np.zeros((3, 2)), n_iter=1)


def test_bad_n_iter_raises(small_X):
    with pytest.raises(ValueError, match="n_iter"):
        squad_mds(small_X, n_iter=0)


def test_bad_metric_raises(small_X):
    with pytest.raises(ValueError, match="metric"):
        squad_mds(small_X, metric="cosine")


def test_bad_init_raises(small_X):
    with pytest.raises(ValueError, match="init"):
        squad_mds(small_X, n_iter=5, init="spectral")


def test_bad_lr_decay_raises(small_X):
    with pytest.raises(ValueError, match="lr_decay"):
        squad_mds(small_X, n_iter=5, lr_decay="cosine")
