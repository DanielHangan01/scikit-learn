"""Tests for pivot sampling strategy in SGDMDS."""
import numpy as np
import pytest
from sklearn.datasets import make_s_curve
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.metrics import euclidean_distances


@pytest.fixture(scope="module")
def small_X():
    X, _ = make_s_curve(n_samples=150, random_state=0)
    return X.astype(np.float64)


@pytest.fixture(scope="module")
def precomputed_D(small_X):
    return euclidean_distances(small_X)


# ---------------------------------------------------------------------------
# Basic output shape and finiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dissimilarity", ["euclidean", "lazy"])
def test_output_shape_and_finiteness(small_X, dissimilarity):
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=15,
        dissimilarity=dissimilarity,
        max_iter=5,
        random_state=0,
    ).fit(small_X)

    assert est.embedding_.shape == (150, 2)
    assert np.isfinite(est.embedding_).all()
    assert np.isfinite(est.stress_)
    assert est.stress_ >= 0


def test_output_shape_precomputed(precomputed_D):
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=15,
        dissimilarity="precomputed",
        max_iter=5,
        random_state=0,
    ).fit(precomputed_D)

    assert est.embedding_.shape == (150, 2)
    assert np.isfinite(est.embedding_).all()


# ---------------------------------------------------------------------------
# pivot_indices_ attribute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dissimilarity", ["euclidean", "lazy", "precomputed"])
def test_pivot_indices_valid(small_X, precomputed_D, dissimilarity):
    X = precomputed_D if dissimilarity == "precomputed" else small_X
    n_pivots = 15
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=n_pivots,
        dissimilarity=dissimilarity,
        max_iter=5,
        random_state=42,
    ).fit(X)

    p = est.pivot_indices_
    n = X.shape[0]
    assert p.shape == (n_pivots,), "wrong number of pivots"
    assert p.dtype in (np.int32, np.int64, np.intp)
    assert (p >= 0).all() and (p < n).all(), "pivot out of range"
    assert len(set(p.tolist())) == n_pivots, "pivots must be unique (MaxMin guarantees this)"


# ---------------------------------------------------------------------------
# MaxMin selection produces well-spread pivots
# ---------------------------------------------------------------------------

def test_maxmin_spread():
    """On a 1D grid [0, 1, …, N-1], MaxMin with k pivots should pick
    approximately evenly spaced indices."""
    N, k = 100, 10
    X = np.arange(N, dtype=np.float64).reshape(-1, 1)
    est = SGDMDS(
        n_components=1,
        sampling_strategy="pivot",
        n_pivots=k,
        dissimilarity="lazy",
        max_iter=1,
        random_state=0,
    ).fit(X)
    p = np.sort(est.pivot_indices_)
    gaps = np.diff(p)
    # Gaps should be roughly N/k = 10.  Allow generous slack.
    assert gaps.min() >= 5, f"pivots too clustered: min gap = {gaps.min()}"
    assert gaps.max() <= 30, f"pivots too spread unevenly: max gap = {gaps.max()}"


# ---------------------------------------------------------------------------
# n_pivots="auto" resolves to a valid integer
# ---------------------------------------------------------------------------

def test_auto_n_pivots(small_X):
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots="auto",
        dissimilarity="lazy",
        max_iter=3,
        random_state=0,
    ).fit(small_X)

    k = est.pivot_indices_.shape[0]
    assert k > 2          # more than n_components
    assert k <= small_X.shape[0]
    assert np.isfinite(est.stress_)


# ---------------------------------------------------------------------------
# Non-metric path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dissimilarity", ["lazy", "precomputed"])
def test_non_metric(small_X, precomputed_D, dissimilarity):
    X = precomputed_D if dissimilarity == "precomputed" else small_X
    est = SGDMDS(
        n_components=2,
        metric=False,
        sampling_strategy="pivot",
        n_pivots=15,
        dissimilarity=dissimilarity,
        max_iter=5,
        random_state=0,
    ).fit(X)

    assert est.embedding_.shape == (X.shape[0], 2)
    assert np.isfinite(est.embedding_).all()
    assert hasattr(est, "pivot_indices_")


# ---------------------------------------------------------------------------
# Hybrid sampling strategy (Phase 2b diagnostic)
# ---------------------------------------------------------------------------

def test_hybrid_alpha_one_matches_pure_pivot(small_X):
    """With alpha=1.0 the hybrid kernel mix collapses to pure pivot updates."""
    common = dict(
        n_components=2, dissimilarity="lazy", n_pivots=15,
        max_iter=5, random_state=7, weighting="uniform",
        learning_rate_init=0.5, scheduler="exponential",
    )
    pivot = SGDMDS(sampling_strategy="pivot", **common).fit(small_X)
    hybrid = SGDMDS(
        sampling_strategy="hybrid", hybrid_alpha=1.0, **common
    ).fit(small_X)
    np.testing.assert_allclose(pivot.embedding_, hybrid.embedding_, atol=1e-10)
    assert abs(pivot.stress_ - hybrid.stress_) < 1e-6


def test_hybrid_alpha_zero_skips_pivot_updates(small_X):
    """alpha=0.0 should make every update a random pair (no pivot updates)."""
    # Compare to the same model with alpha=1.0 (different update mix) — outputs
    # should differ measurably.
    common = dict(
        n_components=2, dissimilarity="lazy", n_pivots=15,
        max_iter=5, random_state=7, weighting="uniform",
        learning_rate_init=0.5, scheduler="exponential",
    )
    h_zero = SGDMDS(
        sampling_strategy="hybrid", hybrid_alpha=0.0, **common
    ).fit(small_X)
    h_one = SGDMDS(
        sampling_strategy="hybrid", hybrid_alpha=1.0, **common
    ).fit(small_X)
    # The embeddings must differ — different update streams.
    assert not np.allclose(h_zero.embedding_, h_one.embedding_, atol=1e-6)
    # Both must be finite.
    assert np.isfinite(h_zero.embedding_).all()
    assert np.isfinite(h_zero.stress_)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_hybrid_intermediate_alpha(small_X, alpha):
    est = SGDMDS(
        n_components=2,
        sampling_strategy="hybrid",
        hybrid_alpha=alpha,
        n_pivots=15,
        dissimilarity="lazy",
        max_iter=5,
        random_state=0,
    ).fit(small_X)
    assert est.embedding_.shape == (small_X.shape[0], 2)
    assert np.isfinite(est.embedding_).all()
    assert est.stress_ > 0
    # Pivots are still selected for hybrid mode.
    assert est.pivot_indices_.shape == (15,)


def test_hybrid_requires_lazy(small_X):
    """hybrid mode is lazy-only (mixes the two lazy kernels)."""
    est = SGDMDS(
        n_components=2,
        sampling_strategy="hybrid",
        hybrid_alpha=0.5,
        n_pivots=15,
        dissimilarity="euclidean",
        max_iter=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="hybrid"):
        est.fit(small_X)


def test_hybrid_alpha_out_of_range_raises(small_X):
    est = SGDMDS(
        n_components=2,
        sampling_strategy="hybrid",
        hybrid_alpha=1.5,  # out of [0, 1]
        n_pivots=15,
        dissimilarity="lazy",
        max_iter=1,
        random_state=0,
    )
    with pytest.raises(ValueError):
        est.fit(small_X)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_n_pivots_too_small_raises(small_X):
    """n_pivots <= n_components should raise ValueError."""
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=2,  # equal to n_components, not strictly greater
        dissimilarity="lazy",
        max_iter=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="n_pivots"):
        est.fit(small_X)


def test_n_pivots_too_large_raises(small_X):
    """n_pivots > n_samples should raise ValueError."""
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=small_X.shape[0] + 1,
        dissimilarity="lazy",
        max_iter=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="n_pivots"):
        est.fit(small_X)


# ---------------------------------------------------------------------------
# Precomputed and euclidean modes give equivalent embeddings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Budget knob on sampling_strategy='random' (Experiment I)
# ---------------------------------------------------------------------------

def test_random_budget_default_matches_auto(small_X):
    """Explicit n_updates_per_epoch=N(N-1)/2 must match the 'auto' default."""
    N = small_X.shape[0]
    common = dict(
        n_components=2, dissimilarity="lazy",
        sampling_strategy="random", max_iter=5, random_state=0,
    )
    e_auto = SGDMDS(**common).fit(small_X).embedding_
    e_full = SGDMDS(n_updates_per_epoch=N * (N - 1) // 2, **common).fit(small_X).embedding_
    np.testing.assert_allclose(e_auto, e_full, atol=1e-10)


def test_random_budget_small_runs(small_X):
    """A small budget (N updates/epoch) must still produce finite output."""
    N = small_X.shape[0]
    est = SGDMDS(
        n_components=2,
        sampling_strategy="random",
        dissimilarity="lazy",
        n_updates_per_epoch=N,
        max_iter=5,
        random_state=0,
    ).fit(small_X)
    assert est.embedding_.shape == (N, 2)
    assert np.isfinite(est.embedding_).all()
    assert np.isfinite(est.stress_)
    assert est.stress_ >= 0


@pytest.mark.parametrize("strategy", ["cycle", "pivot", "hybrid"])
def test_random_budget_rejected_for_other_strategies(small_X, strategy):
    """Setting an int n_updates_per_epoch with non-random strategy raises."""
    kwargs = dict(
        n_components=2, dissimilarity="lazy",
        sampling_strategy=strategy, n_updates_per_epoch=100,
        max_iter=1, random_state=0,
    )
    if strategy in ("pivot", "hybrid"):
        kwargs["n_pivots"] = 15
    if strategy == "hybrid":
        kwargs["hybrid_alpha"] = 0.5
    est = SGDMDS(**kwargs)
    with pytest.raises(ValueError, match="n_updates_per_epoch"):
        est.fit(small_X)


def test_random_budget_validation_range(small_X):
    """n_updates_per_epoch=0 must be rejected via _parameter_constraints."""
    est = SGDMDS(
        n_components=2,
        sampling_strategy="random",
        dissimilarity="lazy",
        n_updates_per_epoch=0,
        max_iter=1,
        random_state=0,
    )
    with pytest.raises(ValueError):
        est.fit(small_X)


# ---------------------------------------------------------------------------
# Precomputed and euclidean modes give equivalent embeddings
# ---------------------------------------------------------------------------

def test_precomputed_vs_euclidean_consistency(small_X, precomputed_D):
    """Both modes receive the same distance information; stresses should be
    in the same ballpark (not necessarily identical due to pair ordering)."""
    kwargs = dict(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=15,
        max_iter=10,
        random_state=7,
    )
    stress_euc = SGDMDS(dissimilarity="euclidean", **kwargs).fit(small_X).stress_
    stress_pre = SGDMDS(dissimilarity="precomputed", **kwargs).fit(precomputed_D).stress_

    # Both should produce finite stress; ratio should be reasonable
    assert np.isfinite(stress_euc) and np.isfinite(stress_pre)
    ratio = max(stress_euc, stress_pre) / max(min(stress_euc, stress_pre), 1e-9)
    assert ratio < 10, f"stresses diverged too much: euclidean={stress_euc:.1f}, precomputed={stress_pre:.1f}"
