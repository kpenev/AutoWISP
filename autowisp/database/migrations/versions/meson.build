# Every revision must be listed here. One that is missing is absent from an
# installed build while still working from a source checkout, so the chain
# would end early and a project would be stamped at the wrong head. The
# revision-chain test catches this when run against an installed package.
py_src = [
  '0001_baseline.py',
  '0002_image_observing_session_index.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database/migrations/versions',
    pure: true,
)
