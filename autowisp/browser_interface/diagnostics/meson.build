py_src = [
  '__init__.py',
  'admin.py',
  'apps.py',
  'detrending_diagnostics_views.py',
  'expression_data.py',
  'expression_views.py',
  'forms.py',
  'image_diagnostics_views.py',
  'models.py',
  'preview_calibrated.py',
  'tests.py',
  'urls.py',
  'views.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/browser_interface/diagnostics',
    pure: true,
)

subdir('migrations')
subdir('static')
subdir('templates')
