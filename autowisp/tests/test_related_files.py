"""Unit tests for the per-step ``related_files`` classifiers (item 9).

Each parallel/step site turns its work item (plus any batch-constant
auxiliary inputs) into :class:`RelatedFile` entries, so an error carries
the artifact(s) it was about. Every test here drives the **real** step or
call site rather than a helper in isolation, in one of two ways matching
how the site works:

- main-process steps scope the context themselves, so the first call
  inside the scope is stubbed and reports what the ambient context holds;
- ``run_pool`` sites hand a classifier to the pool for the workers to
  apply, so ``run_pool`` is intercepted and the classifier it was given is
  applied to a sample item.

The generic scoping machinery (worker transport, crash promotion, dedup)
is covered in ``test_error_context.py``.
"""

import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy

from autowisp.error_context import get_error_context
from autowisp.exceptions import FileKind
from autowisp.processing_steps import add_images_to_db as add_images_to_db_step
from autowisp.processing_steps import calculate_photref_merit as merit_step
from autowisp.processing_steps import calibrate as calibrate_step
from autowisp.processing_steps import (
    fit_source_extracted_psf_map as psf_map_step,
)
from autowisp.processing_steps import fit_star_shape as fit_star_shape_step
from autowisp.processing_steps import stack_to_master as stack_step
from autowisp.processing_steps import stack_to_master_flat as flat_step
from autowisp.light_curves import apply_correction as apply_correction_step

# ``magnitude_fitting/__init__`` re-exports the ``iterative_refit``
# *function* under the submodule's own name, so a plain import binds the
# function and cannot be patched against.
iterative_refit_step = import_module(
    "autowisp.magnitude_fitting.iterative_refit"
)


def _pairs(related):
    """(kind, posix path) tuples for order-independent assertions."""

    return [(rf.kind, rf.path.as_posix()) for rf in related]


class _StubRaise(Exception):
    """Raised by a stubbed per-item call to stop the step right there."""


class TestMainProcessScopes(unittest.TestCase):
    """The main-process steps, exercised through their real dispatch.

    These build their ``RelatedFile`` entries at the scope rather than in a
    classifier, so rather than testing a helper in isolation each test runs
    the actual step and stubs the first call *inside* the scope. The stub
    records what the ambient context holds at that moment -- exactly what
    ``_stamp`` copies onto an error raised anywhere below -- and then
    raises, stopping the step before it reaches any database, FITS, or HDF5
    access.
    """

    def _record_scope_and_raise(self, seen):
        """Return a stub that appends the ambient files, then raises."""

        # Signature-agnostic: the stubbed calls differ per step and none of
        # their arguments matter, only the context in force when they run.
        def stub(*_args, **_kwargs):
            """Stand in for the first call inside the scope."""

            seen.append(_pairs(get_error_context().related_files))
            raise _StubRaise

        return stub

    def test_calibrate_scopes_the_raw_image_and_masters(self):
        """Channel-keyed master dicts contribute one entry per channel.

        ``Calibrator`` is built before the loop and its constructor opens
        the masters, so it is stubbed out; everything from the scope down
        is the real step.
        """

        seen = []
        config = {
            "master_bias": {"R": "/M/bias_R.fits", "G": "/M/bias_G.fits"},
            "master_dark": None,  # not applied -> skipped
            "master_flat": "/M/flat.fits",  # bare filename also accepted
        }
        with (
            patch.object(calibrate_step, "Calibrator", lambda **_kwargs: None),
            self.assertRaises(_StubRaise),
        ):
            calibrate_step.calibrate(
                ["/RAW/img.fits"],
                None,
                config,
                self._record_scope_and_raise(seen),
                None,
            )

        self.assertEqual(len(seen), 1)
        pairs = seen[0]
        self.assertEqual(pairs[0], (FileKind.RAW_IMAGE, "/RAW/img.fits"))
        self.assertIn((FileKind.MASTER_BIAS, "/M/bias_R.fits"), pairs)
        self.assertIn((FileKind.MASTER_BIAS, "/M/bias_G.fits"), pairs)
        self.assertIn((FileKind.MASTER_FLAT, "/M/flat.fits"), pairs)
        # The un-applied dark contributes nothing.
        self.assertFalse(any(kind is FileKind.MASTER_DARK for kind, _ in pairs))

    def test_calibrate_without_masters_scopes_only_the_image(self):
        """With no masters configured, only the raw image is attached."""

        seen = []
        with (
            patch.object(calibrate_step, "Calibrator", lambda **_kwargs: None),
            self.assertRaises(_StubRaise),
        ):
            calibrate_step.calibrate(
                ["/RAW/img.fits"],
                None,
                {},
                self._record_scope_and_raise(seen),
                None,
            )

        self.assertEqual(seen, [[(FileKind.RAW_IMAGE, "/RAW/img.fits")]])

    def test_add_images_to_db_scopes_the_raw_image(self):
        """The raw image is in scope before any DB work begins.

        ``Evaluator`` is the first call inside the scope and runs *before*
        ``start_db_session``, so stubbing it keeps the test away from the
        database entirely.
        """

        seen = []
        with (
            patch.object(
                add_images_to_db_step,
                "Evaluator",
                self._record_scope_and_raise(seen),
            ),
            self.assertRaises(_StubRaise),
        ):
            add_images_to_db_step.add_images_to_db(["/RAW/a.fits"], {})

        self.assertEqual(seen, [[(FileKind.RAW_IMAGE, "/RAW/a.fits")]])

    def test_psf_map_scopes_the_dr_file(self):
        """The PSF-map fit scopes the DR file it is smoothing."""

        seen = []
        config = dict.fromkeys(
            [
                "srcextract_psf_params",
                "srcextract_psfmap_terms",
                "srcextract_psfmap_weights",
                "srcextract_psfmap_error_avg",
                "srcextract_psfmap_rej_level",
                "srcextract_psfmap_max_rej_iter",
            ]
        )
        with (
            patch.object(
                psf_map_step,
                "smooth_srcextract_psf",
                self._record_scope_and_raise(seen),
            ),
            patch.object(
                psf_map_step, "get_dr_substitutions", lambda _config: {}
            ),
            self.assertRaises(_StubRaise),
        ):
            psf_map_step.fit_source_extracted_psf_map(
                ["/DR/a.h5"], None, config, None, None
            )

        self.assertEqual(seen, [[(FileKind.DR_FILE, "/DR/a.h5")]])

    def test_photref_merit_scopes_the_dr_file(self):
        """Merit calculation scopes the DR file it is scoring."""

        seen = []
        config = {
            what + "_version": 0
            for what in [
                "srcextract",
                "catalogue",
                "skytoframe",
                "background",
                "srcproj",
            ]
        }
        with (
            patch.object(
                merit_step, "get_typical_star", lambda *args, **kwargs: None
            ),
            patch.object(
                merit_step,
                "get_frame_merit_info",
                self._record_scope_and_raise(seen),
            ),
            self.assertRaises(_StubRaise),
        ):
            merit_step.calculate_photref_merit(["/DR/a.h5"], config)

        self.assertEqual(seen, [[(FileKind.DR_FILE, "/DR/a.h5")]])

    def test_stack_to_master_scopes_frames_and_master(self):
        """Stacking scopes every input frame plus the master it writes.

        ``mark_start`` is injected by the manager and is the first call
        inside the scope, so recording through it needs no patching of the
        step itself -- only ``get_master_fname``, which would otherwise
        open the first frame to build the name.
        """

        seen = []
        with (
            patch.object(
                stack_step, "get_master_fname", lambda *args: "/M/master.fits"
            ),
            self.assertRaises(_StubRaise),
        ):
            stack_step.stack_to_master(
                ["/CAL/a.fits", "/CAL/b.fits"],
                None,
                {},
                self._record_scope_and_raise(seen),
                None,
            )

        self.assertEqual(
            seen,
            [
                [
                    (FileKind.CALIBRATED_IMAGE, "/CAL/a.fits"),
                    (FileKind.CALIBRATED_IMAGE, "/CAL/b.fits"),
                    (FileKind.OUTPUT, "/M/master.fits"),
                ]
            ],
        )

    def test_stack_to_master_flat_scopes_both_masters(self):
        """The flat stacker attaches the high *and* low masters it writes.

        The step consumes a large, interdependent configuration, so this
        uses the step's own parser defaults rather than a hand-built dict.
        """

        seen = []
        masters = {"high": "/M/high.fits", "low": "/M/low.fits"}
        with (
            patch.object(flat_step, "get_master_fnames", lambda *args: masters),
            self.assertRaises(_StubRaise),
        ):
            flat_step.stack_to_master_flat(
                ["/CAL/f.fits"],
                None,
                flat_step.parse_command_line([]),
                self._record_scope_and_raise(seen),
                None,
            )

        self.assertEqual(
            seen,
            [
                [
                    (FileKind.CALIBRATED_IMAGE, "/CAL/f.fits"),
                    (FileKind.OUTPUT, "/M/high.fits"),
                    (FileKind.OUTPUT, "/M/low.fits"),
                ]
            ],
        )


class TestWorkerPoolClassifiers(unittest.TestCase):
    """The ``run_pool`` sites, checked through the call that wires them up.

    These steps do not scope the context themselves: they hand ``run_pool``
    a classifier and ``_WorkerEntry`` applies it inside each worker. That
    mechanism -- including transport back to the parent, crash promotion
    and dedup -- is covered in ``test_error_context.py``, so what is left
    to check per step is *which* classifier it passes and what that
    classifier makes of an item. Each test therefore runs the real call
    site with ``run_pool`` intercepted, then applies the classifier it was
    handed.
    """

    def _captured_classifier(self, module, run, pool_result=()):
        """Run ``run``, returning the ``related_files`` its pool was given."""

        captured = {}

        def fake_run_pool(_worker, _items, **kwargs):
            """Stand in for ``run_pool``, recording the classifier."""

            captured["related_files"] = kwargs.get("related_files")
            return pool_result

        with patch.object(module, "run_pool", fake_run_pool):
            run()
        return captured["related_files"]

    def test_fit_star_shape_passes_the_frame_set(self):
        """Every frame of a simultaneous-fit set is attached, not just one."""

        config = {
            "data_reduction_fname": "{RAWFNAME}.h5",
            "num_simultaneous": 2,
            "num_parallel_processes": 2,
        }
        classifier = self._captured_classifier(
            fit_star_shape_step,
            lambda: fit_star_shape_step.fit_star_shape(
                ["/CAL/a.fits", "/CAL/b.fits"], None, config, None, None
            ),
        )

        self.assertEqual(
            _pairs(classifier(["/CAL/a.fits", "/CAL/b.fits"])),
            [
                (FileKind.CALIBRATED_IMAGE, "/CAL/a.fits"),
                (FileKind.CALIBRATED_IMAGE, "/CAL/b.fits"),
            ],
        )

    def _detrending_classifier(self, **config):
        """The classifier ``apply_parallel_correction`` passes for a run."""

        return self._captured_classifier(
            apply_correction_step,
            lambda: apply_correction_step.apply_parallel_correction(
                ["/LC/src.h5"], None, 2, **config
            ),
            # The real return is concatenated, so hand back something empty
            # rather than a bare list.
            pool_result=[numpy.array([])],
        )

    def test_detrending_passes_lightcurve_and_photref(self):
        """epd/tfa attach the light curve plus the single photref."""

        with patch.object(apply_correction_step, "get_db_engine", MagicMock()):
            classifier = self._detrending_classifier(
                single_photref_dr_fname="/DR/sp.h5"
            )

        self.assertEqual(
            _pairs(classifier("/LC/src.h5")),
            [
                (FileKind.LIGHTCURVE, "/LC/src.h5"),
                (FileKind.DR_FILE, "/DR/sp.h5"),
            ],
        )

    def test_detrending_without_photref_passes_only_the_lightcurve(self):
        """With no single photref configured, only the LC is attached."""

        with patch.object(apply_correction_step, "get_db_engine", MagicMock()):
            classifier = self._detrending_classifier()

        self.assertEqual(
            _pairs(classifier("/LC/src.h5")),
            [(FileKind.LIGHTCURVE, "/LC/src.h5")],
        )

    def _magfit_classifier(self, master_photref_fname):
        """The classifier ``single_iteration`` passes for a magfit run."""

        configuration = SimpleNamespace(
            source_name_format="{0:d}",
            master_photref_fname=master_photref_fname,
            single_photref_dr_fname="/DR/sp.h5",
            num_parallel_processes=2,
        )
        with patch.object(
            iterative_refit_step, "LinearMagnitudeFit", MagicMock()
        ):
            return self._captured_classifier(
                iterative_refit_step,
                lambda: iterative_refit_step.single_iteration(
                    ["/DR/x.h5"],
                    photref=None,
                    configuration=configuration,
                    path_substitutions={"magfit_iteration": 0},
                    # Never called, but the step wraps both in
                    # ``functools.partial``, which rejects ``None``.
                    mark_start=MagicMock(),
                    mark_end=MagicMock(),
                ),
            )

    def test_magfit_passes_dr_and_both_photrefs(self):
        """fit_magnitudes attaches the DR plus the single/master photref."""

        classifier = self._magfit_classifier("/M/mp.fits")

        self.assertEqual(
            _pairs(classifier("/DR/x.h5")),
            [
                (FileKind.DR_FILE, "/DR/x.h5"),
                (FileKind.DR_FILE, "/DR/sp.h5"),
                (FileKind.MASTER_PHOTREF, "/M/mp.fits"),
            ],
        )

    def test_magfit_without_master_photref(self):
        """Before a master photref exists it is simply omitted."""

        classifier = self._magfit_classifier(None)

        self.assertEqual(
            _pairs(classifier("/DR/x.h5")),
            [(FileKind.DR_FILE, "/DR/x.h5"), (FileKind.DR_FILE, "/DR/sp.h5")],
        )


if __name__ == "__main__":
    unittest.main()
