"""Tests for the standalone Pivot MDS (Brandes & Pich, 2007) function."""
import numpy as np
import pytest
from sklearn.datasets import make_s_curve
from sklearn.manifold._classical_mds import ClassicalMDS
from sklearn.manifold._pivot_mds import pivot_mds
from sklearn.metrics import euclidean_distances


@pytest.fixture(scope="module")
def small_X():
    X, _ = make_s_curve(n_samples=120, random_state=0)
    return X.astype(np.float64)


@pytest.fixture(scope="module")
def small_D(small_X):
    return euclidean_distances(small_X)


def _stress(embedding, dissimilarity_matrix):
    D_emb = euclidean_distances(embedding, embedding)
    return 0.5 * float(np.sum((D_emb - dissimilarity_matrix) ** 2))


# ---------------------------------------------------------------------------
# Output shape and finiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_components", [2, 3])
def test_pivot_mds_output_shape(small_X, n_components):
    pivots = np.array([0, 5, 10, 20, 40, 80], dtype=np.int64)
    embedding, S = pivot_mds(small_X, pivots, n_components=n_components)

    assert embedding.shape == (small_X.shape[0], n_components)
    assert S.shape == (n_components,)
    assert np.isfinite(embedding).all()
    assert np.isfinite(S).all()
    # Singular values are non-negative and in decreasing order.
    assert (S >= 0).all()
    assert (np.diff(S) <= 1e-10).all()


def test_pivot_mds_precomputed_metric(small_X, small_D):
    pivots = np.array([0, 5, 10, 20, 40, 80], dtype=np.int64)

    emb_euclid, _ = pivot_mds(small_X, pivots, 2, metric="euclidean")
    emb_precom, _ = pivot_mds(small_D, pivots, 2, metric="precomputed")

    # Stress should match between modes since the input distances are identical.
    assert abs(_stress(emb_euclid, small_D) - _stress(emb_precom, small_D)) < 1e-6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_pivot_mds_deterministic(small_X):
    pivots = np.array([3, 17, 29, 55, 88, 110], dtype=np.int64)
    e1, s1 = pivot_mds(small_X, pivots, 2)
    e2, s2 = pivot_mds(small_X, pivots, 2)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(s1, s2)


def test_pivot_mds_accepts_list_pivots(small_X):
    e_arr, _ = pivot_mds(small_X, np.array([0, 5, 10, 20]), 2)
    e_lst, _ = pivot_mds(small_X, [0, 5, 10, 20], 2)
    np.testing.assert_array_equal(e_arr, e_lst)


# ---------------------------------------------------------------------------
# Reduction to ClassicalMDS when k == N (the load-bearing correctness check)
# ---------------------------------------------------------------------------

def test_pivot_mds_reduces_to_classical_at_full_k(small_X, small_D):
    """With all points as pivots, Pivot MDS should match ClassicalMDS stress."""
    N = small_X.shape[0]
    pivots = np.arange(N, dtype=np.int64)

    emb_pivot, _ = pivot_mds(small_X, pivots, n_components=2)
    emb_classical = ClassicalMDS(n_components=2).fit_transform(small_X)

    # Stress is invariant to rotation/reflection — the most robust comparison.
    stress_pivot = _stress(emb_pivot, small_D)
    stress_classical = _stress(emb_classical, small_D)
    assert abs(stress_pivot - stress_classical) / max(stress_classical, 1e-12) < 1e-6


def test_pivot_mds_reduces_to_classical_precomputed(small_D):
    N = small_D.shape[0]
    pivots = np.arange(N, dtype=np.int64)

    emb_pivot, _ = pivot_mds(small_D, pivots, 2, metric="precomputed")
    emb_classical = ClassicalMDS(n_components=2, metric="precomputed").fit_transform(small_D)

    s_pivot = _stress(emb_pivot, small_D)
    s_classical = _stress(emb_classical, small_D)
    assert abs(s_pivot - s_classical) / max(s_classical, 1e-12) < 1e-6


# ---------------------------------------------------------------------------
# Smaller k still produces a sensible embedding
# ---------------------------------------------------------------------------

def test_pivot_mds_small_k_better_than_random(small_X, small_D):
    """A Pivot MDS embedding at k=20 should have much lower stress than a
    random Gaussian embedding of the same shape."""
    rng = np.random.RandomState(0)
    pivots = rng.choice(small_X.shape[0], size=20, replace=False)

    emb_pivot, _ = pivot_mds(small_X, pivots, 2)
    stress_pivot = _stress(emb_pivot, small_D)

    # Match the scale of pivot_mds output, so the random baseline is fair.
    scale = np.std(emb_pivot)
    emb_random = rng.standard_normal(emb_pivot.shape) * scale
    stress_random = _stress(emb_random, small_D)

    assert stress_pivot < 0.5 * stress_random, (
        f"PivotMDS stress {stress_pivot:.2f} should be << random {stress_random:.2f}"
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_pivot_mds_k_less_than_components_raises(small_X):
    with pytest.raises(ValueError, match="n_pivots"):
        pivot_mds(small_X, np.array([0, 1]), n_components=3)


def test_pivot_mds_out_of_range_pivot_raises(small_X):
    n = small_X.shape[0]
    with pytest.raises(ValueError, match="out-of-range"):
        pivot_mds(small_X, np.array([0, 1, n + 10]), n_components=2)
    with pytest.raises(ValueError, match="out-of-range"):
        pivot_mds(small_X, np.array([0, -1, 5]), n_components=2)


def test_pivot_mds_bad_metric_raises(small_X):
    with pytest.raises(ValueError, match="metric"):
        pivot_mds(small_X, np.array([0, 1, 2, 3]), 2, metric="cosine")


def test_pivot_mds_non_square_precomputed_raises(small_X):
    with pytest.raises(ValueError, match="square"):
        pivot_mds(small_X, np.array([0, 1, 2, 3]), 2, metric="precomputed")