py_src = [
  'env.py',
  'helpers.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database/migrations',
    pure: true,
)

# The revision template is data read by Alembic when generating a new
# revision, not an importable module.
py.install_sources(
    ['script.py.mako'],
    subdir: 'autowisp/database/migrations',
    pure: true,
)

subdir('versions')
