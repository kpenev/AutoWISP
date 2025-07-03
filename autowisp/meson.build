py_src = [
  '__init__.py',
  'catalog.py',
  'evaluator.py',
  'file_utilities.py',
  'fits_utilities.py',
  'hdf5_file.py',
  'image_smoothing.py',
  'image_utilities.py',
  'iterative_rejection_util.py',
  'miscellaneous.py',
  'multiprocessing_util.py',
  'piecewise_bicubic_psf_map.py',
  'pipeline_exceptions.py',
  'processor.py',
  'source_finder.py',
  'source_finder_util.py',
  'split_sources.py',
  'tess_interface.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp',
    pure: true,
)

subdir('astrometry')
subdir('data_reduction')
subdir('database')
subdir('diagnostics')
subdir('fit_expression')
subdir('hat')
subdir('image_calibration')
subdir('light_curves')
subdir('magnitude_fitting')
subdir('processing_steps')
subdir('resources')
subdir('tests')
