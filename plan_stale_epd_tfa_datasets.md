---
name: Fix stale EPD/TFA corrected datasets
description: Plan to fix shape mismatch when LC files grow after EPD/TFA has already run, on branch fix-stale-epd-tfa-datasets
type: project
originSessionId: 011dc283-565d-4673-8535-f1d1da1b0e66
---
# Plan: Fix Stale EPD/TFA Corrected Datasets

**Branch**: `fix-stale-epd-tfa-datasets`

**Why**: When an LC file grows (new images processed via fit_magnitudes) after EPD/TFA has already run, `add_corrected_dataset` in `light_curve_file.py` uses `if_exists="ignore"` and leaves the stale smaller-shaped corrected dataset in place. On the next EPD/TFA run, h5py raises `TypeError: Boolean indexing array has incompatible shape` because `fit_points.shape=(N_new,)` but the corrected dataset has shape `(N_old,)`.

**Root cause confirmed**: LC at `/AperturePhotometry/Aperture000/MagnitudeFitting/Magnitude` had 1630 points, but `/AperturePhotometry/Aperture000/EPD/Magnitude` (the corrected dataset) had stale shape of 408 from a previous run.

## Two Cases to Handle

When the existing corrected dataset is smaller than `corrected_selection.shape`:

**Case 1** – New images match `fit_points_filter_expression`: EPD ensemble is stale. Must re-run: `epd`, `generate_epd_statistics`, `tfa`, `generate_tfa_statistics`.

**Case 2** – New images don't match the filter: Just extend the dataset to make room. No re-processing needed.

**Detection**: Check `existing_dset[fit_points[:existing_size]]` for non-fill values. Non-fill → Case 1. All fill → Case 2.

## Files to Change

1. **`autowisp/light_curves/light_curve_file.py`** ✅ DONE
   - `StaleCorrectionsError(corrected_key, existing_size, required_size)` exception class at module level
   - `add_corrected_dataset` restructured:
     - Capture `newly_created = add_dataset(if_exists="ignore", ...)`; if newly created, initialize all positions with fill value so "never written" is distinguishable from real corrections
     - fill_value: `nan if fill_val is None else float(fill_val)` for float dtype; zero-of-correct-type for non-float
     - If existing dataset is smaller than required:
       - **Detection**: `existing_dset[corrected_selection[:existing_size]]` — isnan-check for unconfigured-float, `==fill_value` otherwise
       - **Case 1** (non-fill found): fill those positions with fill_value (undo stale corrections for this photref only, preserving other photrefs), raise `StaleCorrectionsError` WITHOUT resizing — on re-run the cleared positions look like Case 2
       - **Case 2** (all fill): resize to required_size, fill new slots with fill_value, proceed normally
   - Key design: photref observations are NOT necessarily sequential or starting at index 0 — always use boolean indexing into the existing dataset, never assume tail-only positions

2. **`autowisp/light_curves/correction.py`** (base class `Correction`)
   - Add a `correct_one_dataset` method that calls the child's core implementation
     and handles the common `StaleCorrectionsError` → `StaleLCResult` logic
   - Child classes override the core implementation; the base class owns the error handling

3. **`autowisp/light_curves/epd_correction.py`**
   - Promote the nested `correct_one_dataset` function (currently inside `__call__`, lines 259–343)
     to a proper method on `EPDCorrection` containing EPD-specific logic only
   - Error handling moves to the base class; no duplication with TFA

4. **`autowisp/light_curves/tfa_correction.py`**
   - Same promotion: nested `correct_one_dataset` (lines 974–1123) becomes a proper method
     on `TFACorrection` with TFA-specific logic only

5. **`autowisp/light_curves/apply_correction.py`**
   - Define `StaleLCResult` as a small dataclass/class with a `filename` attribute
   - `apply_parallel_correction`: collect stale LC filenames via `isinstance(result, StaleLCResult)`, return alongside normal results
   - Using a typed object (not a numeric sentinel like `numpy.inf`) ensures no legitimate bad-fit result can accidentally trigger a pipeline reset
   - `StaleLCResult` carries only `filename` — the photref is NOT encoded in it because `fit_points_filter_expression` is what restricts to a photref, not dataset substitutions like `magfit_iteration`. The photref (`single_photref_fname`) is known from context in `lightcurve_processing.__call__`

6. **`autowisp/processing_steps/lc_detrending.py`**
   - Thread stale LC list through `detrend_light_curves` back to caller

7. **`autowisp/database/lightcurve_processing.py`**
   - Add `reset_lc_pipeline_steps(single_photref_fname, step_names=(...))`:
     - Delete `LightCurveProcessingProgress` rows for epd/generate_epd_statistics/tfa/generate_tfa_statistics
     - Delete `MasterFile` rows for `epd_stat` and `tfa_stat` (let generate steps recreate/overwrite)
     - Refresh in-memory `self.pending` via `set_pending(db_session)`
   - In `__call__`: detect stale results from EPD/TFA, call `reset_lc_pipeline_steps`, skip marking step as final (pipeline will re-run)

## Key Design Notes

- Case 2 is handled silently inside `add_corrected_dataset` with no pipeline impact
- Case 1 resets the **entire photref's ensemble** (not per-LC) — EPD/TFA are ensemble corrections
- `replace_nonfinite` sentinel is ~-1.7e38 (not NaN) due to HDF5 scaleoffset filter constraints
- Exception-across-Pool pickling avoided by returning sentinel values, not raising
- Pipeline reset granularity: full photref reset (all 4 downstream steps)

## Open Question

After `reset_lc_pipeline_steps` marks EPD/TFA as not done, should the current pipeline run immediately re-queue and re-execute those steps, or simply leave them pending for the next invocation? Deferred — to be decided during implementation.

## Status

Plan updated 2026-05-04. Ready for implementation.
