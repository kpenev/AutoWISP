py_src = [
  '__init__.py',
  'base.py',
  'condition.py',
  'condition_expression.py',
  'configuration.py',
  'hdf5_attributes.py',
  'hdf5_datasets.py',
  'hdf5_links.py',
  'hdf5_products.py',
  'hdf5_structure_versions.py',
  'image.py',
  'image_type.py',
  'input_master_types.py',
  'light_curve_processing_progress.py',
  'light_curve_status.py',
  'master_files.py',
  'observing_session.py',
  'processing_sequence.py',
  'step_dependencies.py',
  'steps_and_parameters.py',
  'target.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database/data_model',
    pure: true,
)

subdir('provenance')
