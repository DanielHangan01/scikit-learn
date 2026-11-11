"""
Shared harness for the paper's reporting runs (Experiments O, P and Q).

The protocol is in EXPERIMENT_O_P_SPEC.md section 7. In short:

- Every fit runs under ``threadpool_limits(1)``. Its time excludes stress
  evaluation, except for SMACOF, whose stopping rule needs it.
- SGDMDS (through ``_epoch_hook``) and SQuaD-MDS (through ``callback``) hand
  their embedding to a ``Recorder``, which keeps a copy at chosen steps and
  subtracts its own time from the fit. The copies are scored after the fit.
- SMACOF's per-iteration stress comes from its own ``stress_history_``.
- Scores use the convention of every earlier experiment: raw stress is
  ``sum_{i<j} (d_ij - delta_ij)^2``, and the scaled stress is the same at the
  stress-optimal global scale ``alpha``.
- Results go under ``SGDMDS_RESULTS_ROOT`` (default ``benchmarks/results_paper``),
  never into ``results/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_info, threadpool_limits

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))

from bench_utils import (  # noqa: E402
    DATASET_ROOT,
    stress_and_optimal_scale_chunked_multi,
)

from sklearn.manifold import MDS  # noqa: E402
from sklearn.manifold._sgd_mds import SGDMDS  # noqa: E402
from sklearn.manifold._squad_mds import squad_mds  # noqa: E402

# Q's torus at N=500 has D=500, so its input is square by coincidence: the
# estimators then warn about a possibly precomputed matrix on every fit.
warnings.filterwarnings("ignore", message="The provided input is a square matrix")

RESULTS_ROOT = Path(
    os.environ.get("SGDMDS_RESULTS_ROOT", BENCH_DIR / "results_paper")
)
REFERENCE_ROOT = Path(
    os.environ.get("SGDMDS_REFERENCE_ROOT", BENCH_DIR / "results")
)

COLUMNS = [
    "dataset", "n_samples", "n_features", "algo", "config", "seed",
    "subsample_seed", "k", "block_size", "n_updates_per_epoch", "n_iter", "lr",
    "exaggerate_d", "init", "max_step_frac", "n_iter_run", "time_fit",
    "time_snapshots", "time_scoring", "stress_raw", "stress_scaled",
    "scale_alpha", "est_rel_err", "n_snapshots", "clip_frac", "diverged", "host",
    "t_start", "canary_time", "canary_d8", "canary_d784",
]
CANARY_EVERY_S = 900


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def sgd_config(kind: str, n_samples: int, k: Optional[int] = None) -> Tuple[str, Dict]:
    """(config id, SGDMDS parameters) for one SGD arm. Defaults are the golden
    configuration, so only the sampling differs between arms."""
    if kind == "full":
        return "full", dict(sampling_strategy="cycle")
    if kind == "random-auto":
        return "random-auto-m2", dict(sampling_strategy="random", block_size=2)
    if kind == "budget":
        return f"budget-k{k}", dict(sampling_strategy="random",
                                     n_updates_per_epoch=k * n_samples,
                                     block_size=2)
    if kind == "blocked":
        return f"blocked-k{k}-m16", dict(sampling_strategy="random",
                                          n_updates_per_epoch=k * n_samples,
                                          block_size=16)
    raise ValueError(kind)


def valid_k(n_samples: int, ks: Iterable[int]) -> List[int]:
    """A budget of k*N pairs must stay below the N(N-1)/2 pairs that exist."""
    return [k for k in ks if k < (n_samples - 1) // 2]


def squad_configs(grid: Dict[str, list]) -> List[Tuple[str, Dict]]:
    """(config id, squad_mds parameters) for the full cross of an M grid."""
    out = []
    for n_iter in grid["n_iter"]:
        for lr in grid["lr"]:
            for exa in grid["exaggerate_d"]:
                for init in grid["init"]:
                    for cap in grid["max_step_frac"]:
                        cap_tag = "none" if cap is None else f"{cap:g}"
                        cid = (f"squad-i{n_iter}-lr{lr:g}-exa{int(exa)}-"
                               f"{init}-cap{cap_tag}")
                        out.append((cid, dict(n_iter=int(n_iter), lr=float(lr),
                                              exaggerate_d=bool(exa), init=init,
                                              max_step_frac=cap)))
    return out


def squad_steps(n_iter: int, n_points: int = 20) -> set:
    """Iterations at which SQuaD-MDS is snapshotted: 0, ~20 log-spaced, last."""
    steps = np.unique(np.round(np.geomspace(1, n_iter, n_points)).astype(int))
    return {0, n_iter, *map(int, steps)}


# ---------------------------------------------------------------------------
# Timed fits
# ---------------------------------------------------------------------------

class Recorder:
    """Snapshot hook. Keeps copies of the embedding and times itself.

    ``times`` are seconds since ``start()`` with the recorder's own overhead
    removed, so they line up with ``time_fit``.
    """

    def __init__(self, keep: Optional[set] = None):
        self.keep = keep

    def start(self):
        self.steps, self.times, self.snapshots = [], [], []
        self.overhead = 0.0
        self.t0 = perf_counter()

    def __call__(self, step, embedding):
        t = perf_counter()
        if self.keep is None or step in self.keep:
            self.snapshots.append(np.array(embedding, copy=True))
            self.steps.append(int(step))
            self.times.append(t - self.t0 - self.overhead)
        self.overhead += perf_counter() - t

    def stop(self) -> float:
        return perf_counter() - self.t0 - self.overhead


def fit_sgd(X, params: Dict, seed: int) -> Dict[str, Any]:
    rec = Recorder()
    est = SGDMDS(compute_stress=False, random_state=seed, **params)
    est._epoch_hook = rec
    with threadpool_limits(1):
        rec.start()
        embedding = est.fit_transform(X)
        time_fit = rec.stop()
    return dict(embedding=embedding, time_fit=time_fit,
                time_snapshots=rec.overhead, steps=rec.steps, times=rec.times,
                snapshots=rec.snapshots, n_iter_run=int(est.n_iter_))


def fit_smacof(X, init: str, seed: int, max_iter: int = 300,
               eps: float = 1e-6) -> Dict[str, Any]:
    """SMACOF as shipped, apart from the init. Its stress is part of its
    stopping rule, so it stays inside the timed region."""
    mds = MDS(n_components=2, metric_mds=True, n_init=1, init=init,
              max_iter=max_iter, eps=eps, random_state=seed,
              normalized_stress=False, n_jobs=1)
    with threadpool_limits(1):
        t0 = perf_counter()
        embedding = mds.fit_transform(X)
        time_fit = perf_counter() - t0
    hist = np.asarray(mds.stress_history_, dtype=np.float64)
    # stress_history_ times start at SMACOF's first iteration; shift them so
    # the last one sits at the end of the fit (the setup comes before).
    offset = time_fit - hist[-1, 0]
    return dict(embedding=embedding, time_fit=time_fit, time_snapshots=0.0,
                steps=list(range(1, len(hist) + 1)),
                times=(hist[:, 0] + offset).tolist(),
                history_raw=hist[:, 1].tolist(), snapshots=[],
                n_iter_run=int(mds.n_iter_))


def fit_squad(X, params: Dict, seed: int) -> Dict[str, Any]:
    rec = Recorder(squad_steps(params["n_iter"]))
    with threadpool_limits(1):
        rec.start()
        embedding, info = squad_mds(
            X, n_components=2, metric="euclidean", stop_exaggeration=0.6,
            momentum=0.0, lr_decay="geometric", random_state=seed,
            callback=rec, **params,
        )
        time_fit = rec.stop()
    return dict(embedding=embedding, time_fit=time_fit,
                time_snapshots=rec.overhead, steps=rec.steps, times=rec.times,
                snapshots=rec.snapshots, n_iter_run=int(info["n_iter"]),
                clip_frac=info["n_clipped"] / max(info["n_point_steps"], 1))


def run_fit(algo: str, X, params: Dict, seed: int) -> Tuple[Dict, Dict]:
    """One timed fit of an arm; returns (fit, the row fields describing it)."""
    if algo == "smacof":
        return (fit_smacof(X, params["init"], seed),
                dict(init=params["init"], n_iter=300))
    if algo == "sgdmds":
        sgd_params = {k: v for k, v in params.items() if k != "k"}
        return fit_sgd(X, sgd_params, seed), dict(
            k=params.get("k"), block_size=sgd_params.get("block_size"),
            n_updates_per_epoch=sgd_params.get("n_updates_per_epoch"))
    return fit_squad(X, params, seed), dict(params)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _from_sums(s_ee, s_ed, s_dd, s_rr):
    alpha = s_ed / s_ee if s_ee > 0 else 1.0
    scaled = s_dd - s_ed * s_ed / s_ee if s_ee > 0 else s_rr
    return s_rr, scaled, alpha


class ExactScorer:
    """Exact stress of embeddings of one X. Input distances are computed once."""

    estimated = False

    def __init__(self, X):
        self.delta = pdist(np.asarray(X, dtype=np.float64))
        self.s_dd = float(self.delta @ self.delta)

    def __call__(self, embedding) -> Tuple[float, float, float]:
        if not np.all(np.isfinite(embedding)):
            return np.inf, np.inf, np.nan
        d = pdist(np.asarray(embedding, dtype=np.float64))
        r = d - self.delta
        return _from_sums(float(d @ d), float(d @ self.delta), self.s_dd,
                          float(r @ r))


class PairSampleScorer:
    """Unbiased stress estimate from a fixed sample of pairs (i < j).

    Every embedding of the same X is scored on the same pairs, so their
    sampling errors largely cancel when they are compared.
    """

    estimated = True

    def __init__(self, X, n_pairs: int = 10**7, seed: int = 0,
                 chunk_bytes: int = 256 * 2**20):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        i = rng.randint(0, n, size=n_pairs)
        j = rng.randint(0, n, size=n_pairs)
        mask = i != j
        self.i = i[mask].astype(np.int32)
        self.j = j[mask].astype(np.int32)
        self.scale = (n * (n - 1) / 2) / self.i.shape[0]
        chunk = max(1, chunk_bytes // (16 * X.shape[1]))
        self.delta = np.empty(self.i.shape[0])
        for s in range(0, self.i.shape[0], chunk):
            diff = X[self.i[s:s + chunk]] - X[self.j[s:s + chunk]]
            self.delta[s:s + chunk] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        self.s_dd = float(self.delta @ self.delta)

    def __call__(self, embedding) -> Tuple[float, float, float]:
        if not np.all(np.isfinite(embedding)):
            return np.inf, np.inf, np.nan
        e = np.asarray(embedding, dtype=np.float64)
        diff = e[self.i] - e[self.j]
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        r = d - self.delta
        raw, scaled, alpha = _from_sums(float(d @ d), float(d @ self.delta),
                                        self.s_dd, float(r @ r))
        return self.scale * raw, self.scale * scaled, alpha


def score_history(fit: Dict, scorer) -> Dict[str, Any]:
    """Score a fit's snapshots. SMACOF's history is its own (raw stress)."""
    if "history_raw" in fit:
        return dict(step=fit["steps"], time=fit["times"],
                    stress_raw=fit["history_raw"], stress_scaled=None,
                    estimated=False)
    scores = [scorer(s) for s in fit["snapshots"]]
    return dict(step=fit["steps"], time=fit["times"],
                stress_raw=[float(s[0]) for s in scores],
                stress_scaled=[float(s[1]) for s in scores],
                estimated=bool(scorer.estimated))


def exact_final_scores(X, embeddings: List[np.ndarray]):
    """Exact (raw, scaled, alpha) per embedding, batched over one X.

    Uses all available threads; call it outside the timed region.
    """
    finite = [k for k, e in enumerate(embeddings) if np.all(np.isfinite(e))]
    out = [(np.inf, np.inf, np.nan)] * len(embeddings)
    if finite:
        scored = stress_and_optimal_scale_chunked_multi(
            X, [embeddings[k] for k in finite])
        for k, (scaled, raw, alpha) in zip(finite, scored):
            out[k] = (raw, scaled, alpha)
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class ResultStore:
    """Per-dataset results.csv (one row per fit), histories.jsonl and
    embeddings/. Resumes per fit: a fit is done once its row is on disk."""

    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.csv = out_dir / "results.csv"
        self.hist = out_dir / "histories.jsonl"
        self.emb_dir = out_dir / "embeddings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._done = set()
        if self.csv.exists():
            df = pd.read_csv(self.csv, on_bad_lines="skip")
            self._done = {(int(r.n_samples), str(r.config), int(r.seed))
                          for r in df.itertuples()}

    def done(self, n_samples: int, config: str, seed: int) -> bool:
        return (int(n_samples), config, int(seed)) in self._done

    def __len__(self):
        return len(self._done)

    def add(self, row: Dict[str, Any], history: Dict[str, Any],
            embedding: Optional[np.ndarray] = None):
        row = dict(row, host=socket.gethostname())
        with open(self.hist, "a") as f:
            f.write(json.dumps(dict(n_samples=int(row["n_samples"]),
                                    config=row["config"],
                                    seed=int(row["seed"]), **history)) + "\n")
        if embedding is not None:
            self.emb_dir.mkdir(exist_ok=True)
            np.save(self.emb_dir / f"N{row['n_samples']}-{row['config']}"
                    f"-s{row['seed']}.npy", embedding)
        frame = pd.DataFrame([{c: row.get(c) for c in COLUMNS}], columns=COLUMNS)
        frame.to_csv(self.csv, mode="a", header=not self.csv.exists(),
                     index=False)
        self._done.add((int(row["n_samples"]), row["config"], int(row["seed"])))


def make_row(fit: Dict, history: Dict, final: Tuple[float, float, float],
             **fields) -> Dict[str, Any]:
    raw, scaled, alpha = final
    est_rel_err = None
    if history.get("estimated") and np.isfinite(raw) and raw > 0:
        est_rel_err = abs(history["stress_raw"][-1] - raw) / raw
    return dict(
        fields, n_iter_run=fit["n_iter_run"], time_fit=fit["time_fit"],
        time_snapshots=fit["time_snapshots"], stress_raw=raw,
        stress_scaled=scaled, scale_alpha=alpha, est_rel_err=est_rel_err,
        n_snapshots=len(history["step"]), clip_frac=fit.get("clip_frac"),
        diverged=not np.isfinite(raw),
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=BENCH_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def blas_threads_inside_fit() -> List[int]:
    """Thread counts the BLAS libraries report inside the timed region."""
    with threadpool_limits(1):
        return [d["num_threads"] for d in threadpool_info()]


def canary_ns_per_pair(n_samples: int = 20_000, reps: int = 3) -> Dict[str, float]:
    """Time a fixed kernel workload, to detect slowdowns from other jobs.

    Measured at the start and end of every run (see write_meta). On one machine
    model the values should agree across jobs; a job whose canary is slow ran on
    a contended node. D=8 is compute-bound, D=784 memory-bandwidth-bound.
    """
    from sklearn.manifold._sgd_mds_cython import run_sgd_epoch_random

    out = {}
    rng = np.random.RandomState(0)
    for D in (8, 784):
        X = np.ascontiguousarray(rng.standard_normal((n_samples, D)))
        E0 = np.ascontiguousarray(rng.standard_normal((n_samples, 2)))
        n_updates = 20 * n_samples
        best = np.inf
        with threadpool_limits(1):
            for _ in range(reps):
                E = E0.copy()
                t0 = perf_counter()
                run_sgd_epoch_random(E, X, n_updates, 0.1, 0, 12345)
                best = min(best, perf_counter() - t0)
        out[f"D{D}"] = 1e9 * best / n_updates
    return out


class Canary:
    """Runs the canary between fits, at most every ``every`` seconds.

    It is never inside a timed region: call ``maybe_run()`` before starting a
    fit's timer. Each measurement is appended to ``canary.csv``, and the latest
    one goes into every row, so a fit can be matched to the load around it.
    """

    def __init__(self, out_dir: Path, every: float = CANARY_EVERY_S):
        self.path = out_dir / "canary.csv"
        self.every = every
        self.last = -np.inf
        self.values: Dict[str, float] = {}

    def maybe_run(self) -> Dict[str, Any]:
        if time.time() - self.last >= self.every:
            ns = canary_ns_per_pair()
            self.last = time.time()
            self.values = dict(canary_time=self.last, canary_d8=ns["D8"],
                               canary_d784=ns["D784"])
            frame = pd.DataFrame([dict(self.values, host=socket.gethostname())])
            frame.to_csv(self.path, mode="a", header=not self.path.exists(),
                         index=False)
        return dict(self.values, t_start=time.time())


def machine_fingerprint() -> Dict[str, Any]:
    import Cython
    import scipy

    import sklearn

    return {
        "host": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "n_cpus": os.cpu_count(),
        "cpus_available": len(os.sched_getaffinity(0)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "-uno")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "cython": Cython.__version__,
        "blas": [dict(api=d.get("internal_api"), threads=d.get("num_threads"),
                      arch=d.get("architecture"),
                      lib=Path(d.get("filepath", "")).name)
                 for d in threadpool_info()],
        "blas_threads_inside_fit": blas_threads_inside_fit(),
        "canary_ns_per_pair": canary_ns_per_pair(),
    }


def dataset_checksum(name: str) -> Optional[str]:
    path = DATASET_ROOT / name / "X.npy"
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def provenance() -> Dict[str, Any]:
    """What must stay fixed for results to be mixed in one results file."""
    fp = machine_fingerprint()
    prov = {k: fp[k] for k in ("git_commit", "python", "numpy", "scipy",
                               "sklearn", "cython", "cpu_model")}
    prov["blas"] = [[b["lib"], b["arch"]] for b in fp["blas"]]
    # In JSON form, so it compares equal to the copy read back from meta.json.
    return json.loads(json.dumps(prov))


def check_provenance(out_dir: Path):
    """Refuse to add results made by different code or libraries.

    A dataset's results are tied to the provenance of its first run. A resumed
    run must match it, and no run may start from uncommitted changes. Set
    SGDMDS_ALLOW_MIXED=1 to override (smoke tests on a development machine).
    """
    if os.environ.get("SGDMDS_ALLOW_MIXED") == "1":
        return
    if _git("status", "--porcelain", "-uno"):
        raise SystemExit("Refusing to run: the repository has uncommitted changes "
                         "(git status). Commit them, or run on a clean checkout.")
    path = out_dir / "meta.json"
    if not path.exists():
        return
    first = json.loads(path.read_text()).get("provenance")
    now = provenance()
    if first is not None and first != now:
        diff = {k: (first.get(k), now.get(k)) for k in now if first.get(k) != now.get(k)}
        raise SystemExit(f"Refusing to resume {out_dir}: its results were made with "
                         f"different code or libraries: {diff}")


def write_meta(out_dir: Path, meta: Dict[str, Any]):
    path = out_dir / "meta.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    if "provenance" not in old:
        old["provenance"] = provenance()
    runs = old.get("runs", [])
    runs.append(dict(machine_fingerprint(), peak_rss_gb=peak_rss_gb(),
                     complete=meta.get("complete", False)))
    path.write_text(json.dumps(dict(old, **meta, runs=runs), indent=2,
                               default=str))


def is_complete(out_dir: Path) -> bool:
    path = out_dir / "meta.json"
    return path.exists() and json.loads(path.read_text()).get("complete", False)


def parse_args(argv: List[str], flags: Dict[str, Any]):
    """Minimal ``--flag value`` parser; returns (positional, values)."""
    argv = list(argv)
    values = {}
    for flag, cast in flags.items():
        if flag in argv:
            i = argv.index(flag)
            if cast is bool:
                values[flag] = True
                del argv[i]
            else:
                values[flag] = cast(argv[i + 1])
                del argv[i:i + 2]
    return argv, values


def int_list(s: str) -> List[int]:
    return [int(v) for v in s.split(",") if v]
