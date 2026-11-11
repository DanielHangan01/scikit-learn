from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chisquare

from scipy.spatial.distance import pdist, squareform

from sklearn.datasets import make_s_curve
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.manifold._sgd_mds_cython import (
    draw_blocks,
    run_sgd_epoch_blocked,
    run_sgd_epoch_random,
)
from sklearn.metrics import euclidean_distances

GOLDEN_PATH = Path(__file__).parent / "data_sgd_mds_golden.npz"

X_S, _ = make_s_curve(200, random_state=0)
X_G = np.random.RandomState(1).standard_normal((150, 20))
D_G = euclidean_distances(X_G)
# The golden embeddings must not depend on the CPU: BLAS and NumPy's exp round
# the last digit differently with AVX-512, and SGD amplifies that. So the
# cycle goldens take distances from pdist, and no golden runs 5 epochs, whose
# learning-rate schedule hits such an exp.
DATA = {
    "s_curve": X_S,
    "gauss20": X_G,
    "s_curve_pdist": squareform(pdist(X_S)),
    "gauss20_pdist": squareform(pdist(X_G)),
}

# The configurations the golden embeddings were recorded with.
RANDOM = dict(sampling_strategy="random", max_iter=10, n_updates_per_epoch=20 * 150)
EXPONENTIAL = dict(scheduler="exponential", learning_rate_init="auto", max_iter=10)
GOLDEN = {
    "budget_uniform": ("gauss20", dict(RANDOM, block_size=2)),
    "budget_inverse": ("gauss20", dict(RANDOM, block_size=2, weighting="inverse")),
    "budget_nonmetric": (
        "s_curve",
        dict(RANDOM, block_size=2, n_updates_per_epoch=20 * 200, metric_mds=False),
    ),
    "random_auto": (
        "s_curve",
        dict(sampling_strategy="random", block_size=2, max_iter=10),
    ),
    "blocked_uniform": ("gauss20", dict(RANDOM, block_size=16)),
    "blocked_inverse": ("gauss20", dict(RANDOM, block_size=16, weighting="inverse")),
    "blocked_nonmetric": (
        "s_curve",
        dict(RANDOM, block_size=16, n_updates_per_epoch=20 * 200, metric_mds=False),
    ),
    "blocked_m4": ("gauss20", dict(RANDOM, block_size=4)),
    "cycle_exponential": ("gauss20_pdist", dict(EXPONENTIAL, metric="precomputed")),
    "cycle_golden_config": ("gauss20_pdist", dict(metric="precomputed")),
    "cycle_nonmetric": (
        "s_curve_pdist",
        dict(EXPONENTIAL, metric_mds=False, metric="precomputed"),
    ),
}


def _run_blocked(E0, X, n_updates, m, lr=0.4, weighting=0, seed=12345, knots=None):
    x_knots, y_knots = (None, None) if knots is None else knots
    E = E0.copy()
    n_blocks = run_sgd_epoch_blocked(
        E, X, n_updates, m, lr, weighting, seed, x_knots=x_knots, y_knots=y_knots
    )
    return E, n_blocks


@pytest.fixture
def small_problem():
    rng = np.random.RandomState(0)
    return rng.standard_normal((300, 37)), rng.standard_normal((300, 2))


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_embeddings(name):
    """Check that embeddings stay bit-identical across refactorings."""
    data, params = GOLDEN[name]
    embedding = SGDMDS(random_state=7, **params).fit_transform(DATA[data])
    assert_array_equal(embedding, np.load(GOLDEN_PATH)[name])


@pytest.mark.parametrize("params", [dict(), dict(sampling_strategy="random")])
def test_output(params):
    est = SGDMDS(max_iter=5, random_state=0, **params)
    embedding = est.fit_transform(X_S)
    assert embedding is est.embedding_
    assert embedding.shape == (X_S.shape[0], 2)
    assert np.isfinite(embedding).all()
    assert est.stress_ >= 0
    assert est.n_iter_ == 5
    assert_allclose(embedding.mean(axis=0), 0, atol=1e-10)


def test_get_feature_names_out():
    est = SGDMDS(max_iter=2, random_state=0).fit(X_S)
    assert_array_equal(est.get_feature_names_out(), ["sgdmds0", "sgdmds1"])


def test_dissimilarity_matrix_attribute():
    assert_allclose(
        SGDMDS(max_iter=1, random_state=0).fit(X_G).dissimilarity_matrix_, D_G
    )
    est = SGDMDS(sampling_strategy="random", max_iter=1, random_state=0).fit(X_G)
    assert not hasattr(est, "dissimilarity_matrix_")


def test_precomputed_matches_euclidean():
    euclidean = SGDMDS(max_iter=5, random_state=0).fit_transform(X_G)
    precomputed = SGDMDS(max_iter=5, random_state=0, metric="precomputed")
    assert_array_equal(precomputed.fit_transform(D_G), euclidean)


def test_metric_and_metric_params():
    embedding = SGDMDS(
        max_iter=2, random_state=0, metric="minkowski", metric_params={"p": 1}
    ).fit_transform(X_G)
    expected = SGDMDS(max_iter=2, random_state=0, metric="manhattan").fit_transform(X_G)
    assert_allclose(embedding, expected)


def test_compute_stress_does_not_change_embedding():
    params = dict(sampling_strategy="random", max_iter=3, random_state=0)
    est = SGDMDS(compute_stress=False, **params).fit(X_G)
    assert est.stress_ == -1.0
    assert_array_equal(est.embedding_, SGDMDS(**params).fit(X_G).embedding_)


def test_n_init_keeps_the_best_run():
    single = [
        SGDMDS(max_iter=2, random_state=seed).fit(X_G).stress_ for seed in range(3)
    ]
    best = SGDMDS(n_init=8, max_iter=2, random_state=0).fit(X_G).stress_
    assert best <= max(single)


def test_n_updates_auto_is_the_number_of_pairs():
    n_samples = X_G.shape[0]
    params = dict(sampling_strategy="random", max_iter=3, random_state=0)
    assert_array_equal(
        SGDMDS(**params).fit_transform(X_G),
        SGDMDS(
            n_updates_per_epoch=n_samples * (n_samples - 1) // 2, **params
        ).fit_transform(X_G),
    )


def test_block_size_is_capped_at_n_samples():
    X = X_G[:5]
    params = dict(sampling_strategy="random", max_iter=3, random_state=0)
    assert_array_equal(
        SGDMDS(block_size=16, **params).fit_transform(X),
        SGDMDS(block_size=5, **params).fit_transform(X),
    )


def test_block_size_is_ignored_with_cycle():
    assert_array_equal(
        SGDMDS(block_size=4, max_iter=3, random_state=0).fit_transform(X_G),
        SGDMDS(max_iter=3, random_state=0).fit_transform(X_G),
    )


@pytest.mark.parametrize(
    "params", [dict(), dict(sampling_strategy="random", block_size=2)]
)
def test_epoch_hook(params):
    snapshots = []
    est = SGDMDS(max_iter=4, random_state=0, **params)
    est._epoch_hook = lambda epoch, embedding: snapshots.append(
        (epoch, embedding.copy())
    )
    est.fit(X_G)
    assert [epoch for epoch, _ in snapshots] == [0, 1, 2, 3, 4]
    assert not np.array_equal(snapshots[0][1], snapshots[-1][1])
    assert_allclose(snapshots[-1][1] - snapshots[-1][1].mean(axis=0), est.embedding_)


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(n_updates_per_epoch=100), "n_updates_per_epoch"),
        (dict(sampling_strategy="random", metric="manhattan"), "only supports"),
        (dict(sampling_strategy="random", metric="precomputed"), "only supports"),
        (dict(compute_stress=False, n_init=2), "requires n_init=1"),
    ],
)
def test_invalid_combinations(params, match):
    with pytest.raises(ValueError, match=match):
        SGDMDS(max_iter=1, **params).fit(X_G)


@pytest.mark.parametrize(
    "params",
    [
        dict(block_size=1),
        dict(n_updates_per_epoch=0),
        dict(sampling_strategy="pivot"),
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        SGDMDS(max_iter=1, **params).fit(X_G)


def test_precomputed_must_be_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        SGDMDS(metric="precomputed", max_iter=1).fit(D_G + np.eye(150, k=1))


def test_square_input_warns():
    with pytest.warns(UserWarning, match="square matrix"):
        SGDMDS(max_iter=1, random_state=0).fit(D_G)


@pytest.mark.parametrize("weighting", [0, 1])
@pytest.mark.parametrize("non_metric", [False, True])
def test_blocked_kernel_with_two_points_matches_random_kernel(
    small_problem, weighting, non_metric
):
    X, E0 = small_problem
    knots = None
    x_knots = y_knots = None
    if non_metric:
        x_knots = np.ascontiguousarray([0.0, 2.0, 4.0, 6.0, 9.0, 30.0])
        y_knots = np.ascontiguousarray([0.0, 1.5, 4.5, 5.0, 8.0, 25.0])
        knots = (x_knots, y_knots)

    expected = E0.copy()
    run_sgd_epoch_random(
        expected, X, 5000, 0.4, weighting, 12345, x_knots=x_knots, y_knots=y_knots
    )
    got, _ = _run_blocked(E0, X, 5000, 2, weighting=weighting, knots=knots)
    assert_array_equal(got, expected)


def test_blocks_follow_the_random_kernel_stream():
    def stream(seed):
        x = seed
        while True:
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            yield x

    n_samples, seed = 97, 2024
    for m in (2, 5):
        draws = stream(seed)
        expected = [[next(draws) % n_samples for _ in range(m)] for _ in range(50)]
        assert_array_equal(draw_blocks(n_samples, m, 50, seed), expected)


def test_pairs_of_a_block_are_uniform():
    n_samples, m = 6, 4
    blocks = draw_blocks(n_samples, m, 30000, 99)
    for a, b in [(0, 1), (0, 3), (2, 3)]:
        cells = blocks[:, a] * n_samples + blocks[:, b]
        assert chisquare(np.bincount(cells, minlength=n_samples**2)).pvalue > 1e-3


def _sequential_updates(E0, X, blocks, n_updates, lr):
    """NumPy version of the blocked kernel, with uniform weights."""

    def distance(x, y):
        # Sums in order, as the kernel does; NumPy's pairwise sum would not.
        acc = 0.0
        for a, b in zip(x.tolist(), y.tolist()):
            acc += (a - b) * (a - b)
        return float(np.sqrt(acc))

    E = E0.copy()
    m = blocks.shape[1]
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    done = 0
    for block in blocks:
        todo = pairs[: min(len(pairs), n_updates - done)]
        for a, b in todo:
            i, j = block[a], block[b]
            if i == j:
                continue
            delta = distance(X[i], X[j])
            diff = E[i] - E[j]
            dist = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]))
            if dist < 1e-12:
                continue
            step = (min(lr, 1.0) * (dist - delta) / (2.0 * dist)) * diff
            E[i] -= step
            E[j] += step
        done += len(todo)
    return E


@pytest.mark.parametrize("m", [3, 4, 6, 8])
def test_blocked_kernel_matches_numpy(m):
    # Duplicate points within a block are common with 11 samples, and the last
    # block is partial since 103 is not a multiple of any number of pairs here.
    # m covers every distance loop: 3 (remainder), 4 (fused), 6 and 8 (chunks).
    rng = np.random.RandomState(3)
    X = rng.standard_normal((11, 5))
    E0 = rng.standard_normal((11, 2))
    n_updates = 103
    n_blocks = -(-n_updates // (m * (m - 1) // 2))

    blocks = draw_blocks(11, m, n_blocks, 12345)
    got, drawn = _run_blocked(E0, X, n_updates, m)
    assert drawn == n_blocks
    assert_array_equal(got, _sequential_updates(E0, X, blocks, n_updates, lr=0.4))


@pytest.mark.parametrize("m", [4, 8])
def test_blocked_kernel_conserves_the_centroid(small_problem, m):
    X, E0 = small_problem
    got, _ = _run_blocked(E0, X, 200 * (m * (m - 1) // 2), m)
    assert_allclose(got.mean(axis=0), E0.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4, 16])
def test_budget_is_counted_in_pairs(small_problem, m):
    X, E0 = small_problem
    n_pairs = m * (m - 1) // 2
    for n_updates in (1, n_pairs, n_pairs + 1, 1000):
        _, n_blocks = _run_blocked(E0, X, n_updates, m)
        assert n_blocks == -(-n_updates // n_pairs)


@pytest.mark.parametrize("m", [1, 301])
def test_blocked_kernel_invalid_block_size(small_problem, m):
    X, E0 = small_problem
    with pytest.raises(ValueError, match="block_size"):
        _run_blocked(E0, X, 100, m)
