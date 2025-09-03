py_src = [
  '__init__.py',
  'admin.py',
  'apps.py',
  'detrending_diagnostics_views.py',
  'display_fits_util.py',
  'log_views.py',
  'models.py',
  'progress_view.py',
  'select_photref_views.py',
  'select_raw_view.py',
  'tune_starfind_views.py',
  'urls.py',
  'views.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/browser_interface/processing',
    pure: true,
)

subdir('migrations')
subdir('static')
subdir('templates')
