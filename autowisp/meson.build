py_src = [
  '__init__.py',
  'bui_util.py',
  'catalog.py',
  'crash_report.py',
  'error_cli.py',
  'error_context.py',
  'error_persistence.py',
  'error_render.py',
  'evaluator.py',
  'exceptions.py',
  'file_utilities.py',
  'fits_utilities.py',
  'hdf5_file.py',
  'image_smoothing.py',
  'image_utilities.py',
  'iterative_rejection_util.py',
  'miscellaneous.py',
  'multiprocessing_util.py',
  'piecewise_bicubic_psf_map.py',
  'processor.py',
  'run_pipeline.py',
  'source_finder.py',
  'source_finder_util.py',
  'split_sources.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp',
    pure: true,
)

# Record the git version at build time so installed copies (which are not git
# repositories) can report it without needing git at run time. See
# autowisp/_write_version.py and autowisp.miscellaneous.get_code_version_str.
custom_target(
    'autowisp-version',
    output: '_version.py',
    command: [py, files('_write_version.py'), '@SOURCE_ROOT@', '@OUTPUT@'],
    build_by_default: true,
    install: true,
    install_dir: py.get_install_dir() / 'autowisp',
)

subdir('astrometry')
subdir('browser_interface')
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
