py.extension_module(
  '_utils',
  [cython_gen.process('_utils.pyx'), utils_cython_tree],
  subdir: 'sklearn/manifold',
  install: true
)

py.extension_module(
  '_barnes_hut_tsne',
  cython_gen.process('_barnes_hut_tsne.pyx'),
  dependencies: [np_dep, openmp_dep],
  subdir: 'sklearn/manifold',
  install: true
)

py.extension_module(
  '_sgd_mds_cython',
  sources: ['_sgd_mds_cython.pyx'],
  dependencies: [np_dep],
  cython_args: cython_args,
  subdir: 'sklearn/manifold',
  install: true
)
