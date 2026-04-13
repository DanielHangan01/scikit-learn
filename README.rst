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

Core Implementation
~~~~~~~~~~~~~~~~~~~

All new code lives under ``sklearn/manifold/``:

- ``_sgd_mds.py``: the ``SGDMDS`` estimator. Includes the learning rate schedulers
  (exponential, harmonic, hybrid, constant), weight bounds estimation, and the
  ``fit``/``fit_transform`` interface. Supports three dissimilarity modes: ``'euclidean'``,
  ``'precomputed'``, and ``'lazy'`` (distances computed on-the-fly, O(1) auxiliary memory).

- ``_sgd_mds_cython.pyx``: the Cython inner loops. ``run_sgd_epoch`` handles the precomputed
  mode (sampling without replacement), and ``run_sgd_epoch_lazy_random_native`` handles the
  lazy mode (xorshift32 sampling). Also includes ``interpolate_isotonic`` for non-metric MDS.

- ``_mds.py``: the existing SMACOF implementation, extended with per-iteration timing and
  stress history (``stress_history_`` attribute) used in benchmark comparisons.

- ``meson.build``: registers ``_sgd_mds_cython`` as a compiled extension.

Benchmarks
~~~~~~~~~~

All benchmark scripts are under ``sklearn/manifold/benchmarks/``:

- ``bench_utils.py``: shared utilities for loading datasets (synthetic and from disk) and
  the ``run_smacof_benchmark`` / ``run_sgd_benchmark`` wrappers.

- ``run_experiment_a.py``: runs SGD-MDS vs SMACOF on precomputed distance matrices over
  20 random seeds per dataset, saving stress, timing, and convergence histories.

- ``run_experiment_b.py``: scaling study with two protocols in one script. The synthetic
  protocol sweeps N from 500 to 5000 on s_curve, swiss_roll, and torus (lifted to high
  dimensionality). The real-world protocol runs fashion_mnist, spambase, and seismic at
  their native size. Both compare SMACOF, SGD-euclidean, and SGD-lazy.

- ``run_experiment_c.py``: hyperparameter grid search over learning rates, schedulers,
  switch ratios, epsilons, and iteration budgets.

All results are written to ``sklearn/manifold/benchmarks/results/``.

Datasets
~~~~~~~~

Real-world datasets go in ``sklearn/manifold/benchmarks/datasets/<name>/`` as ``X.npy``
and optionally ``y.npy``. The datasets used in the paper come from the Espadoto et al.
survey benchmark collection. Synthetic datasets (``s_curve``, ``swiss_roll``, ``torus``)
are generated on the fly and need no setup.


Installation
------------

The Cython extension ``_sgd_mds_cython`` needs to be compiled before use, so a full
in-place build of scikit-learn is required.

1. Create and activate a virtual environment::

    python -m venv .venv
    source .venv/bin/activate

2. Install build dependencies::

    pip install numpy cython meson-python ninja

3. Build scikit-learn in-place::

    pip install --no-build-isolation -e .

To check the build worked::

    python -c "from sklearn.manifold._sgd_mds import SGDMDS; print(SGDMDS())"


Reproducing the Experiments
----------------------------

All benchmark scripts need to be run from the ``sklearn/manifold/benchmarks/`` directory::

    cd sklearn/manifold/benchmarks/

Dataset Setup
~~~~~~~~~~~~~

Place real-world datasets under ``datasets/<name>/X.npy`` (and optionally ``y.npy``)::

    benchmarks/datasets/
        fashion_mnist/
            X.npy
            y.npy
        coil20/
            X.npy
            y.npy
        epileptic/
            X.npy
            ...

Synthetic datasets are generated automatically.

Experiment A: Convergence and Solution Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs SGD-MDS (30 epochs) vs SMACOF (300 iterations) over 20 random seeds per dataset.
Reproduces Figures 1 and 2 and the stress comparison tables.

.. code-block:: bash

    # All datasets in benchmarks/datasets/ plus s_curve
    python run_experiment_a.py

    # Specific datasets
    python run_experiment_a.py fashion_mnist coil20

Results go to ``results/experiment_a/sgd_30_epochs/<dataset>/``.

Experiment B: Scaling Study (Synthetic + Real-World)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reproduces Figure 3. Measures wall-clock time for SMACOF, SGD-euclidean, and SGD-lazy
as N grows (synthetic) or at native dataset size (real-world).

.. code-block:: bash

    # Both protocols
    python run_experiment_b.py

    # Synthetic sweep only
    python run_experiment_b.py synthetic

    # Real-world only
    python run_experiment_b.py real

    # Specific dataset(s)
    python run_experiment_b.py torus fashion_mnist

Results go to ``results/experiment_b/scaling_mixed_euclid_vs_lazy/<dataset>/``

Experiment C: Hyperparameter Grid Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searches over learning rates, schedulers, switch ratios, and epsilons to find the
configuration used in the other experiments.

.. code-block:: bash

    python run_experiment_c.py

    # Specific datasets
    python run_experiment_c.py fashion_mnist s_curve

Results go to ``results/experiment_c/<dataset>/``.


Quick Usage Example
-------------------

.. code-block:: python

    import numpy as np
    from sklearn.manifold._sgd_mds import SGDMDS

    X = np.random.randn(500, 50)

    # Standard mode (N <= ~20,000): precomputes the full distance matrix
    sgd = SGDMDS(n_components=2, dissimilarity="euclidean",
                 max_iter=30, scheduler="hybrid", random_state=42)
    embedding = sgd.fit_transform(X)

    # Lazy mode (large N): computes distances on-the-fly, O(1) extra memory
    sgd_lazy = SGDMDS(n_components=2, dissimilarity="lazy",
                      max_iter=30, scheduler="hybrid", random_state=42)
    embedding_lazy = sgd_lazy.fit_transform(X)

    print(f"Stress: {sgd.stress_:.4f}, Iterations: {sgd.n_iter_}")


Citation
--------

If you use this code, please cite::

    Bridging Graph Drawing and Dimensionality Reduction with Stochastic Stress Optimization.
    EUROVIS 2026, GDxDR: Bridging Graph Drawing and Dimensionality Reduction, Short Paper.
