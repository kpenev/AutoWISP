py_src = [
  '__init__.py',
  'admin.py',
  'apps.py',
  'create_project_view.py',
  'models.py',
  'tests.py',
  'urls.py',
  'views.py'
]

py.install_sources(
    py_src,
    subdir: 'autowisp/browser_interface/home',
    pure: true,
)

subdir('migrations')
subdir('static')
subdir('templates')
