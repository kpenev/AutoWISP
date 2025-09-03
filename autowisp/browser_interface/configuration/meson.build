py_src = [
  '__init__.py',
  'admin.py',
  'apps.py',
  'models.py',
  'urls.py',
  'views.py'
]

py.install_sources(
    py_src,
    subdir: 'autowisp/browser_interface/configuration',
    pure: true,
)

subdir('migrations')
subdir('static')
subdir('templates')
