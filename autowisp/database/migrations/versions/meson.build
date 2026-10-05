# Every revision must be listed here. One that is missing is absent from an
# installed build while still working from a source checkout, so the chain
# would end early and a project would be stamped at the wrong head. The
# revision-chain tests compare this list with the directory, which they can
# only do from a checkout: an installed chain that ends early looks valid.
py_src = [
  '0001_baseline.py',
  '0002_image_session_index.py',
  '0003_repair_timestamp_triggers.py',
  '0004_shorten_image_raw_fname.py',
  '0005_shorten_condition_expr.py',
  '0006_drop_step_desc_unique.py',
  '0007_drop_condition_fks.py',
  '0008_widen_lc_status_id.py',
  '0009_float_columns_to_double.py',
  '0010_unique_image_type_name.py',
  '0011_diagnostic_expression.py',
  '0012_qc_included_datasets.py',
  '0013_exclusion_rule_parameters.py',
  '0014_drop_photref_merit_step.py',
  '0015_engine_set_masters.py',
  '0016_diagnostic_definitions.py',
]

py.install_sources(
    py_src,
    subdir: 'autowisp/database/migrations/versions',
    pure: true,
)
