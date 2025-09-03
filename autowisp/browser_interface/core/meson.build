py_src = [
  '__init__.py',
  'admin.py',
  'apps.py',
  'models.py',
  'views.py',
  'walk_fs_view.py'
]

py.install_sources(
    py_src,
    subdir: 'autowisp/browser_interface/core',
    pure: true,
)

subdir('migrations')
subdir('templates')
subdir('templatetags')
