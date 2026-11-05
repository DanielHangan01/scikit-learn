"""Tests for point-blocked sampling in SGDMDS (``block_size`` and friends).

Two guarantees are checked here:

1. NOTHING THAT EXISTED BEFORE CHANGED. ``data_blocked_golden.npz`` holds
   embeddings recorded from commit 7c4301d (before blocking was added) for
   every sampling mode; they must still reproduce bit for bit. On top of
   that, the blocked kernel itself must reproduce the original random-pair
   kernel bit for bit at ``block_size=2`` with random blocks.

2. THE BLOCKED KERNEL DOES WHAT ITS DOCSTRING SAYS. Its sampling is checked
   through ``draw_blocks`` (which runs the kernel's own sampler), and its
   three update rules against small NumPy oracles fed the same blocks.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chisquare

from sklearn.datasets import make_s_curve
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.manifold._sgd_mds_cython import (
    draw_blocks,
    run_sgd_epoch_lazy_blocked,
    run_sgd_epoch_lazy_random_native,
)

GOLDEN_PATH = Path(__file__).parent / "data_blocked_golden.npz"

X_S, _ = make_s_curve(200, random_state=0)
X_G = np.random.RandomState(1).standard_normal((150, 20))
DATA = {"s_curve": X_S, "gauss20": X_G}

# Verbatim copy of the configs the golden file was recorded with. Changing
# any of them invalidates the fixture.
BUDGET = dict(
    dissimilarity="lazy",
    sampling_strategy="random",
    learning_rate_init=0.5,
    scheduler="hybrid",
    max_iter=10,
)
GOLDEN = {
    "budget_uniform": ("gauss20", dict(BUDGET, n_updates_per_epoch=20 * 150)),
    "budget_inverse": (
        "gauss20",
        dict(BUDGET, n_updates_per_epoch=20 * 150, weighting="inverse"),
    ),
    "budget_nonmetric": (
        "s_curve",
        dict(BUDGET, n_updates_per_epoch=20 * 200, metric=False),
    ),
    "random_auto": ("s_curve", dict(BUDGET, max_iter=5)),
    "cycle_euclidean": ("gauss20", dict(dissimilarity="euclidean", max_iter=10)),
    "pivot_lazy": (
        "gauss20",
        dict(
            dissimilarity="lazy",
            sampling_strategy="pivot",
            n_pivots=10,
            max_iter=10,
        ),
    ),
    "pivot_euclidean": (
        "s_curve",
        dict(
            dissimilarity="euclidean",
            sampling_strategy="pivot",
            n_pivots=10,
            max_iter=10,
        ),
    ),
    "hybrid_lazy": (
        "gauss20",
        dict(
            dissimilarity="lazy",
            sampling_strategy="hybrid",
            n_pivots=10,
            hybrid_alpha=0.5,
            max_iter=10,
        ),
    ),
}

UPDATE_CODES = {"sequential": 0, "sum": 1, "mean": 2}


def _pairs(m):
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


def _run_blocked(E0, X, n_updates, m, partition, mode, lr=0.4, weighting=0,
                 seed=12345, fused=True, knots=(None, None)):
    E = E0.copy()
    perm = np.arange(X.shape[0], dtype=np.int32)
    n_blocks = run_sgd_epoch_lazy_blocked(
        E, X, n_updates, m, partition, mode, lr, weighting, seed, perm,
        fused=fused, x_knots=knots[0], y_knots=knots[1],
    )
    return E, n_blocks


@pytest.fixture
def small_problem():
    rs = np.random.RandomState(0)
    return rs.standard_normal((300, 37)), rs.standard_normal((300, 2))


# =============================================================================
# 1. Nothing that existed before changed
# =============================================================================


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_embeddings_reproduce_bitwise(name):
    data, kwargs = GOLDEN[name]
    golden = np.load(GOLDEN_PATH)[name]
    embedding = SGDMDS(random_state=7, **kwargs).fit_transform(DATA[data])
    assert_array_equal(embedding, golden)


def _isotonic_knots():
    # A monotone map with a few kinks, as _fit_monotonic_map would produce.
    x = np.ascontiguousarray([0.0, 2.0, 4.0, 6.0, 9.0, 30.0])
    y = np.ascontiguousarray([0.0, 1.5, 4.5, 5.0, 8.0, 25.0])
    return x, y


@pytest.mark.parametrize("weighting", [0, 1])
@pytest.mark.parametrize("non_metric", [False, True])
@pytest.mark.parametrize("update", ["sequential", "sum", "mean"])
def test_blocked_m2_matches_random_kernel_bitwise(
    small_problem, weighting, non_metric, update
):
    X, E0 = small_problem
    knots = _isotonic_knots() if non_metric else (None, None)

    ref = E0.copy()
    run_sgd_epoch_lazy_random_native(
        ref, X, 5000, 0.4, weighting, 12345, x_knots=knots[0],
        y_knots=knots[1],
    )
    got, _ = _run_blocked(E0, X, 5000, 2, False, UPDATE_CODES[update],
                          weighting=weighting, knots=knots)
    assert_array_equal(got, ref)


@pytest.mark.parametrize("weighting", ["uniform", "inverse"])
@pytest.mark.parametrize("metric", [True, False])
def test_estimator_blocked_path_at_m2_matches_default(weighting, metric):
    # block_update='sum' at m=2 routes through the blocked kernel but is the
    # same algorithm; it must equal the default (original kernel) exactly,
    # including the estimator's seeding, scheduling and isotonic refits.
    kwargs = dict(GOLDEN["budget_uniform"][1], weighting=weighting,
                  metric=metric)
    ref = SGDMDS(random_state=7, **kwargs).fit_transform(X_G)
    got = SGDMDS(random_state=7, block_update="sum", **kwargs).fit_transform(
        X_G
    )
    assert_array_equal(got, ref)


# =============================================================================
# 2. Sampling
# =============================================================================


def _xorshift32_stream(seed):
    x = seed if seed != 0 else 1
    while True:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        yield x


def test_random_blocks_follow_the_random_kernels_stream():
    # Random blocks are m consecutive draws of the random kernel's
    # generator, taken modulo N: at m=2 exactly its (i, j) sequence.
    n, seed = 97, 2024
    for m in (2, 5):
        stream = _xorshift32_stream(seed)
        expected = np.array(
            [[next(stream) % n for _ in range(m)] for _ in range(50)]
        )
        assert_array_equal(draw_blocks(n, m, 50, False, seed), expected)


@pytest.mark.parametrize("n, m", [(103, 4), (100, 4), (50, 8), (40, 2)])
def test_partition_blocks_are_disjoint_and_cover(n, m):
    per_partition = n // m
    blocks = draw_blocks(n, m, 3 * per_partition, True, 5)
    for p in range(3):
        part = blocks[p * per_partition:(p + 1) * per_partition].ravel()
        assert len(np.unique(part)) == per_partition * m
    # consecutive partitions are reshuffled, not repeated
    assert not np.array_equal(blocks[:per_partition],
                              blocks[per_partition:2 * per_partition])


def test_random_block_pair_slots_are_uniform():
    # Every pair of slots is an independent uniform (i, j) draw over N^2,
    # including i == j -- the distribution of the random kernel.
    n, m = 6, 4
    blocks = draw_blocks(n, m, 30000, False, 99)
    for a, b in [(0, 1), (0, 3), (2, 3)]:
        cells = np.bincount(blocks[:, a] * n + blocks[:, b], minlength=n * n)
        assert chisquare(cells).pvalue > 1e-3


# =============================================================================
# 3. Update rules against NumPy oracles
# =============================================================================


def _delta_in_order(x, y):
    """||x - y|| summed over features in order, as the kernels do.

    Python floats are IEEE doubles with no fused multiply-add, so this is
    bit-identical to the kernels' loop (NumPy's pairwise sum is not).
    """
    acc = 0.0
    for a, b in zip(x.tolist(), y.tolist()):
        diff = a - b
        acc += diff * diff
    return float(np.sqrt(acc))


def _oracle(E0, X, blocks, n_updates, mode, lr):
    """Uniform-weight metric step, straight from the docstring.

    Written with the kernel's expressions in the kernel's order, so it is
    expected to agree bit for bit.
    """
    E = E0.copy()
    m = blocks.shape[1]
    pairs = _pairs(m)
    done = 0
    mu = min(lr, 1.0)
    for block in blocks:
        todo = pairs[: min(len(pairs), n_updates - done)]
        snapshot = E.copy()
        acc = np.zeros((m, E.shape[1]))
        count = np.zeros(m)
        for a, b in todo:
            i, j = block[a], block[b]
            count[a] += 1
            count[b] += 1
            if i == j:
                continue
            delta = _delta_in_order(X[i], X[j])
            src = E if mode == "sequential" else snapshot
            diff = src[i] - src[j]
            dist = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]))
            if dist < 1e-12:
                continue
            step = (mu * (dist - delta) / (2.0 * dist)) * diff
            if mode == "sequential":
                E[i] -= step
                E[j] += step
            else:
                acc[a] -= step
                acc[b] += step
        if mode != "sequential":
            for a in range(m):
                if count[a]:
                    E[block[a]] += acc[a] / (count[a] if mode == "mean" else 1)
        done += len(todo)
    return E


@pytest.mark.parametrize("update", ["sequential", "sum", "mean"])
@pytest.mark.parametrize("partition", [False, True])
@pytest.mark.parametrize("m", [3, 4, 6, 8])
def test_update_rules_match_numpy_oracle(update, partition, m):
    # N=11 makes duplicate indices within random blocks common, and 103
    # pair updates is not a multiple of any P used here, so the final block
    # is partial. m covers every distance path: 3 (remainder loop only),
    # 4 (fused), 6 (chunks of four + remainder), 8 (chunks only).
    rs = np.random.RandomState(3)
    X = rs.standard_normal((11, 5))
    E0 = rs.standard_normal((11, 2))
    n_updates = 103
    P = m * (m - 1) // 2
    n_blocks = -(-n_updates // P)

    blocks = draw_blocks(11, m, n_blocks, partition, 12345)
    expected = _oracle(E0, X, blocks, n_updates, update, lr=0.4)
    got, drawn = _run_blocked(E0, X, n_updates, m, partition,
                              UPDATE_CODES[update])
    assert drawn == n_blocks
    assert_array_equal(got, expected)


@pytest.mark.parametrize("update", ["sequential", "sum", "mean"])
@pytest.mark.parametrize("partition", [False, True])
def test_fused_4_point_pass_matches_generic_bitwise(
    small_problem, update, partition
):
    X, E0 = small_problem
    a, _ = _run_blocked(E0, X, 5003, 4, partition, UPDATE_CODES[update],
                        fused=True)
    b, _ = _run_blocked(E0, X, 5003, 4, partition, UPDATE_CODES[update],
                        fused=False)
    assert_array_equal(a, b)


@pytest.mark.parametrize("update", ["sequential", "sum", "mean"])
@pytest.mark.parametrize("partition", [False, True])
@pytest.mark.parametrize("m", [4, 8])
def test_centroid_is_conserved(small_problem, update, partition, m):
    # Every pair moves its two points by equal and opposite amounts. 'mean'
    # rescales per point, which stays balanced only when every point of a
    # block took part in the same number of pairs -- i.e. on full blocks --
    # so it is checked with a budget that is a whole number of blocks.
    X, E0 = small_problem
    P = m * (m - 1) // 2
    got, _ = _run_blocked(E0, X, 200 * P, m, partition, UPDATE_CODES[update])
    assert_allclose(got.mean(axis=0), E0.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4, 16])
def test_budget_is_counted_in_pairs(small_problem, m):
    X, E0 = small_problem
    P = m * (m - 1) // 2
    for n_updates in (1, P, P + 1, 1000):
        _, n_blocks = _run_blocked(E0, X, n_updates, m, True, 0)
        assert n_blocks == -(-n_updates // P)


# =============================================================================
# 4. Estimator
# =============================================================================


@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize("sampling", ["random", "partition"])
@pytest.mark.parametrize("update", ["sequential", "sum", "mean"])
def test_estimator_blocked_modes_run(m, sampling, update):
    kwargs = GOLDEN["budget_uniform"][1]
    ref = SGDMDS(random_state=7, **kwargs).fit(X_G)
    est = SGDMDS(random_state=7, block_size=m, block_sampling=sampling,
                 block_update=update, **kwargs).fit(X_G)
    assert np.all(np.isfinite(est.embedding_))
    assert est.stress_ < 1.5 * ref.stress_


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dissimilarity="euclidean"),
        dict(dissimilarity="lazy", sampling_strategy="pivot"),
        dict(dissimilarity="lazy", sampling_strategy="hybrid"),
    ],
)
def test_block_params_rejected_for_other_strategies(kwargs):
    with pytest.raises(ValueError, match="only configurable"):
        SGDMDS(block_size=4, max_iter=2, **kwargs).fit(X_G)


def test_block_size_larger_than_n_samples_raises():
    with pytest.raises(ValueError, match="exceeds n_samples"):
        SGDMDS(dissimilarity="lazy", sampling_strategy="random",
               block_size=151, max_iter=2).fit(X_G)


def test_block_size_one_is_invalid():
    with pytest.raises(ValueError):
        SGDMDS(dissimilarity="lazy", sampling_strategy="random",
               block_size=1, max_iter=2).fit(X_G)


def test_kernel_partition_requires_perm(small_problem):
    X, E0 = small_problem
    with pytest.raises(ValueError, match="perm buffer"):
        run_sgd_epoch_lazy_blocked(E0.copy(), X, 10, 4, True, 0, 0.4, 0, 1)


def test_force_blocked_kernel_is_bitwise_identical(monkeypatch):
    # The research switch changes the implementation, never the embedding.
    import sklearn.manifold._sgd_mds as sgd_mds_module

    kwargs = GOLDEN["budget_uniform"][1]
    monkeypatch.setattr(sgd_mds_module, "FORCE_BLOCKED_KERNEL", True)
    got = SGDMDS(random_state=7, **kwargs).fit_transform(X_G)
    assert_array_equal(got, np.load(GOLDEN_PATH)["budget_uniform"])
