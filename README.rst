SGD-MDS: Bridging Graph Drawing and Dimensionality Reduction
=============================================================

This is a fork of `scikit-learn <https://scikit-learn.org>`_ with the implementation and
experiments from the paper:

    **Bridging Graph Drawing and Dimensionality Reduction with Stochastic Stress Optimization**
    *(EUROVIS 2026, GDxDR: Bridging Graph Drawing and Dimensionality Reduction, Short Paper)*

The paper adapts SGD-based stress optimization from graph drawing to metric MDS on vector data.
The result is a scikit-learn-compatible ``SGDMDS`` estimator that converges faster than SMACOF
while reaching comparable or lower stress on standard high-dimensional benchmarks.


Repository Structure
--------------------

This branch holds the code and runs behind the extended paper. Earlier experiments
(A to N), their docs and the Pivot MDS baseline are on the branch ``research/history``.

Implementation
~~~~~~~~~~~~~~

All code lives under ``sklearn/manifold/``:

- ``_sgd_mds.py``, ``_sgd_mds_cython.pyx``: the ``SGDMDS`` estimator and its kernels.
  ``sampling_strategy='cycle'`` sweeps all pairs of a stored distance matrix each
  epoch; ``'random'`` draws ``n_updates_per_epoch`` pairs and computes their
  distances on the fly, in blocks of ``block_size`` points (memory linear in N).
- ``_squad_mds.py``, ``_squad_mds_cython.pyx``: SQuaD-MDS (Lambert et al., 2022),
  a research baseline, not part of the public API.
- ``_mds.py``: scikit-learn's SMACOF, with a ``stress_history_`` attribute added for
  the convergence comparisons.

Experiments
~~~~~~~~~~~

Under ``sklearn/manifold/benchmarks/``:

- ``run_experiment_o.py`` (small suite: 17 datasets), ``run_experiment_p.py`` (large
  suite: 8 sources at N = 10k to 50k) and ``run_experiment_q.py`` (runtime against N),
  sharing ``paper_harness.py`` (timing, snapshots, scoring, provenance) and
  ``paper_config.py`` (suites and grids). Design: ``EXPERIMENT_O_P_SPEC.md``.
- ``analyze_experiment_o_p_q.py``, ``dataset_tables.py``, ``paper_tables.py``: the
  analysis and tables, reported in ``EXPERIMENT_O_P_Q_RESULTS.md``,
  ``EXPERIMENT_O_P_DATASET_TABLES.md`` and ``PAPER_RESULTS.md``.
- ``cluster/``: job scripts for the COMA cluster; ``retime.py`` and
  ``compare_results.py`` check timings and reproduction; ``bench_blocked_kernel.py``
  is the kernel microbenchmark.

Datasets go in ``benchmarks/datasets/<name>/X.npy``; results are written under
``benchmarks/results_paper/`` (set ``SGDMDS_RESULTS_ROOT`` to change it). Neither is
in git.


Installation
------------

The Cython kernels need an in-place build of scikit-learn::

    conda env create -f sklearn/manifold/benchmarks/environment.yml
    conda activate sklearn-dev
    pip install --no-build-isolation -e .

To check the build::

    python -m pytest sklearn/manifold/tests/test_sgd_mds.py


Reproducing the Experiments
---------------------------

From ``sklearn/manifold/benchmarks/``, on a clean checkout (the runners refuse to
run from uncommitted changes)::

    python run_experiment_o.py            # all 17 datasets, or name some
    python run_experiment_p.py            # all 8 sources
    python run_experiment_q.py
    python analyze_experiment_o_p_q.py    # writes results_paper/analysis/
    python paper_tables.py                # the paper's tables

A smoke test: ``python run_experiment_o.py seismic --seeds 1 --quick``.


Quick Usage Example
-------------------

.. code-block:: python

    import numpy as np
    from sklearn.manifold._sgd_mds import SGDMDS

    X = np.random.randn(2000, 50)

    # Full sweep over all pairs (stores the N x N distance matrix)
    embedding = SGDMDS(random_state=42).fit_transform(X)

    # Budgeted, point-blocked sampling: 200 updates per point per epoch,
    # distances computed on the fly (memory linear in N)
    fast = SGDMDS(sampling_strategy="random", n_updates_per_epoch=200 * len(X),
                  random_state=42)
    embedding_fast = fast.fit_transform(X)


Citation
--------

If you use this code, please cite::

    Bridging Graph Drawing and Dimensionality Reduction with Stochastic Stress Optimization.
    EUROVIS 2026, GDxDR: Bridging Graph Drawing and Dimensionality Reduction, Short Paper.
