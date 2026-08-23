# Cloud Detection Plan

Tracking the work needed to turn the current cloud-detection prototype (branch
`CloudDetection`) into a robust, configurable, tested feature. Details will be
filled in gradually — for now this captures the current state and the near-term
milestones.

## Current state (as of branch `CloudDetection`)

Diagnostics are computed during normal pipeline steps and stored in the
`ImageDiagnostics` table (registered as `DiagnosticType` rows in
`autowisp/database/initialize_database.py`):

- **`calibrate` step** — via `autowisp/diagnostics/sky.py::get_local_sky_diagnostics`,
  called from `autowisp/image_calibration/calibrator.py`. Splits each color
  channel into an `8x8` grid, suppresses stars by keeping the dimmer pixels in
  each block (`bright_clip=0.8` quantile), and derives:
  - `median_rb_ratio` — median local R/B sky ratio
  - `local_sky_gb_ratio` — median local G/B sky ratio
  - `local_sky_brightness_minmax_frac` — spread of block brightness across frame
- **`find_stars` step** — `autowisp/processing_steps/find_stars.py::_regional_source_balance`
  produces `src_count_min_half_fraction` (smaller/larger source count between
  image halves, worse of horizontal/vertical split).
- **Pre-existing diagnostics reused for flagging**: `num_extracted_src`,
  `matched_fraction`, `srcextract_mag_zeropt`, `pixel_q5`.

The **flagging decision** currently lives entirely in the browser interface:
`autowisp/browser_interface/diagnostics/image_diagnostics_views.py::_get_cloud_detection`.
It is recomputed on every diagnostics page view (nothing persisted), builds
robust per-observing-session baselines (median / MAD / 0.8 quantile), and flags
an image when it shows extreme star loss, or moderate star loss combined with a
supporting sky-color / regional-collapse signal. Sessions with fewer than
`CLOUD_MIN_CONTEXT_IMAGES` fall back to an absolute star-count floor.

All ~15 thresholds are hardcoded module-level constants (`CLOUD_*`) in that
view file.

## Milestone 1 — Proper configurability

Move all tuning knobs out of hardcoded constants into the pipeline's normal
configuration mechanism (command line + configuration/parameter tables), so the
detection can be tuned per instrument/site without editing source.

Open questions / TODO:
- [ ] Decide where cloud flagging belongs. It is currently a browser-view
      side effect recomputed on each render. It should likely be a real
      processing step (or part of an existing step) that persists a per-image
      cloudy flag to the DB, with the browser view only *displaying* the stored
      result.
- [ ] Enumerate every `CLOUD_*` constant in `image_diagnostics_views.py` and the
      hardcoded params in `sky.py` (`grid_size`, `bright_clip`) and map each to a
      configuration parameter with a sensible default.
- [ ] Wire the parameters through the parameter/configuration tables the way
      other steps do (follow the existing processing-step config pattern).
- [ ] Ensure diagnostic computation params (grid size, bright clip) are recorded
      with the stored diagnostics so results remain interpretable if defaults change.

## Milestone 2 — Field validation across instruments & locations

The current metrics were tuned on limited data and likely misfire on benign
conditions. Users need to run detection over observing sessions from a range of
instruments and sites and report back so we can refine the metrics.

Suspected false-positive sources to check specifically:
- Dawn / dusk (rising sky background, changing color balance)
- Light pollution (elevated, colored sky background)
- Low-altitude fields (airmass reddening, gradients, vignetting)
- Moon presence / moonrise-moonset

Collect results here as testing proceeds:
- [ ] (session / instrument / site) → observed behavior, false positives/negatives

TODO:
- [ ] Define a lightweight way for users to review flags vs. truth (the
      diagnostics page already highlights flagged frames — decide whether that
      plus a notes column is enough, or a small export is needed).
- [ ] Once patterns emerge, refine metrics/thresholds and re-validate.

## Milestone 3 — Unit tests for the diagnostic functions

Verify each diagnostic function computes what it claims and stays stable under
future edits. These are pure-function tests (no full pipeline run needed).

- [ ] `get_local_sky_diagnostics` / `_block_sky_medians` (`autowisp/diagnostics/sky.py`):
      synthetic channel arrays with known R/G/B block medians → assert
      `median_rb_ratio`, `local_sky_gb_ratio`, `local_sky_brightness_minmax_frac`;
      cover star suppression (bright pixels rejected), masked pixels, missing
      channels (returns `{}`), and the min-good-pixel guards.
- [ ] `_regional_source_balance` (`find_stars.py`): known source layouts →
      assert `src_count_min_half_fraction` for balanced vs. one-sided fields,
      and horizontal-vs-vertical worst-case selection.
- [ ] Flagging helpers (`_get_cloud_session_baselines`, `_get_star_loss_flags`,
      `_get_sky_change_flag`, `_get_regional_star_collapse_flag`,
      `_is_unusual_fractional_shift`): feed constructed baselines/diagnostics and
      assert flag decisions at, above, and below thresholds.
- [ ] Follow the existing test conventions (`autowisp/tests/`, `unittest`); add
      the new test module to `autowisp/tests/meson.build`.
