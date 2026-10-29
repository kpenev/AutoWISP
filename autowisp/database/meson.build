py_src = [
  '__init__.py',
  'defaults.py',
  'existing_project.py',
  'experiments.py',
  'frozen_row.py',
  'hdf5_file_structure.py',
  'image_processing.py',
  'initialize_data_reduction_structure.py',
  'initialize_database.py',
  'initialize_light_curve_structure.py',
  'interface.py',
  'lightcurve_processing.py',
  'migrate.py',
  'migrate_cli.py',
  'pending_projects.py',
  'photref_selection.py',
  'pipeline_runs.py',
  'processing.py',
  'project_copy.py',
  'project_rows.py',
  'provenance_resolver.py',
  'sessions.py',
  'user_interface.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database',
    pure: true,
)

# Alembic reads this at runtime to locate migrations/; it is data, not a
# module, so it needs its own install_sources call.
py.install_sources(
    ['alembic.ini'],
    subdir: 'autowisp/database',
    pure: true,
)

subdir('data_model')
subdir('migrations')
