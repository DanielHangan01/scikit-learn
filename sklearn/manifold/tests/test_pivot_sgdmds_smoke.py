import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.manifold._sgd_mds import SGDMDS
import time

X, _ = make_s_curve(n_samples=500, random_state=0)

N_PIVOTS = 20

for diss in ["euclidean", "lazy"]:
    est = SGDMDS(
        n_components=2,
        sampling_strategy="pivot",
        n_pivots=N_PIVOTS,
        dissimilarity=diss,
        max_iter=20,
        random_state=0,
    ).fit(X)
    assert est.embedding_.shape == (500, 2)
    assert est.pivot_indices_.shape == (N_PIVOTS,)
    assert len(set(est.pivot_indices_.tolist())) == N_PIVOTS  # MaxMin gives unique
    assert np.isfinite(est.stress_)
    print(diss, "stress=", est.stress_)


for strategy in ["random", "pivot"]:
    t0 = time.time()
    est = SGDMDS(
        n_components=2,
        dissimilarity="lazy",
        sampling_strategy=strategy,
        n_pivots=N_PIVOTS,   # ignored for "random"
        max_iter=20,
        random_state=0,
    ).fit(X)
    print(f"{strategy:8s}  stress={est.stress_:.0f}  time={time.time()-t0:.2f}s")