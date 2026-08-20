# Every revision must be listed here. One that is missing is absent from an
# installed build while still working from a source checkout, so the chain
# would end early and a project would be stamped at the wrong head. The
# revision-chain test catches this when run against an installed package.
py_src = [
  '0001_baseline.py',
  '0002_image_session_index.py',
  '0003_repair_timestamp_triggers.py',
  '0004_shorten_image_raw_fname.py',
  '0005_shorten_condition_expr.py',
  '0006_drop_step_desc_unique.py',
  '0007_drop_condition_fks.py',
  '0008_widen_lc_status_id.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database/migrations/versions',
    pure: true,
)
