"""Unit tests for the per-step ``related_files`` classifiers (item 9).

Each parallel/step site turns its work item (plus any batch-constant
auxiliary inputs) into :class:`RelatedFile` entries, so an error carries
the artifact(s) it was about. Every test here drives the **real** step or
call site rather than a helper in isolation, in one of two ways matching
how the site works:

- main-process steps scope the context themselves, so the first call
  inside the scope is stubbed to *raise* and the assertion is on the
  related files the capture boundary stamps onto the escaping error --
  never on the ambient context read from inside the scope, which proves
  only that the scope exists, not that anything reaches the error;
- ``run_pool`` sites hand a classifier to the pool for the workers to
  apply, so ``run_pool`` is intercepted and the classifier it was given is
  applied to a sample item.

The generic scoping machinery (worker transport, crash promotion, dedup)
is covered in ``test_error_context.py``.
"""

import os
import unittest
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.interface import set_project_home
from autowisp.error_context import capture_errors
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.exceptions import AutoWISPError, Component, FileKind
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
    """The ``(kind, posix path)`` entries, for comparison in a test.

    Always compared with ``assertCountEqual``: the order in which related
    files appear carries no meaning (nothing downstream depends on it --
    the renderer just lists them and the artifact-FK lookup is an SQL
    ``IN``), so pinning it would over-constrain the implementation.
    Counting rather than a set still catches a missing entry, an
    unexpected extra one, and an accidental duplicate.
    """

    return [(rf.kind, rf.path.as_posix()) for rf in related]


class _StubRaise(Exception):
    """Raised by a stubbed per-item call to stop the step right there."""


# A mixin for the test classes below, so it deliberately has no tests of
# its own.
# pylint: disable=too-few-public-methods
class _StampedFilesMixin:
    """Run something behind a capture boundary and inspect the error."""

    def _stamped_files(self, run):
        """Return the related files on the error escaping ``run``.

        Wraps ``run`` in the same capture boundary the manager applies to
        a step, so this reproduces the real path: the code raises, its
        scopes unwind, and the boundary stamps whatever survived.
        """

        @capture_errors(component=Component.STEP)
        def boundary():
            """Stand in for ``ProcessingManager._run_step``."""

            run()

        with self.assertRaises(AutoWISPError) as caught:
            boundary()
        return _pairs(caught.exception.related_files)

    @staticmethod
    def _raise_stub(*_args, **_kwargs):
        """Stand in for the first call inside a scope; always raises."""

        raise _StubRaise

    def _stamped_step_files(self, step, collection, config=None, **patches):
        """Fail ``step`` inside its scope; return the files it reports.

        Covers the steps sharing the manager's calling convention
        ``(collection, start_status, configuration, mark_start, mark_end)``:
        ``mark_start`` is the first call inside the scope, so passing the
        raising stub for it fails the step exactly there, with no need to
        patch the step itself.

        Args:
            step(callable):    The step function to run.

            collection:    The images / DR files to hand it.

            config(dict or None):    Its configuration; empty if omitted.

            patches:    Attributes of the step's *own* module to replace
                for the duration -- the objects built before the loop that
                would otherwise reach real files (a ``Calibrator``, a
                master-filename resolver).

        Returns:
            list:    ``(kind, path)`` pairs from the stamped error.
        """

        module = import_module(step.__module__)
        with ExitStack() as patched:
            for attribute, replacement in patches.items():
                patched.enter_context(
                    patch.object(module, attribute, replacement)
                )
            return self._stamped_files(
                lambda: step(
                    collection, None, config or {}, self._raise_stub, None
                )
            )


# pylint: enable=too-few-public-methods


class TestMainProcessScopes(_StampedFilesMixin, unittest.TestCase):
    """The main-process steps, exercised through their real dispatch.

    These build their ``RelatedFile`` entries at the scope rather than in a
    classifier, so each test runs the actual step with the first call
    *inside* the scope stubbed to raise -- which also stops the step before
    it reaches any database, FITS or HDF5 access.

    The assertion is on the **stamped exception**, not on the live ambient
    context. That distinction is the whole point: the scope's ``with``
    unwinds (resetting the ContextVar) before any enclosing ``except``
    runs, so a test that reads the context from inside the scope passes
    even when nothing ever reaches the error. What the pipeline actually
    depends on is what ``capture_errors`` -- the boundary at
    ``ProcessingManager._run_step`` -- ends up putting on the error.
    """

    def test_calibrate_scopes_the_raw_image_and_masters(self):
        """Channel-keyed master dicts contribute one entry per channel.

        ``Calibrator`` is built before the loop and its constructor opens
        the masters, so it is stubbed out; everything from the scope down
        is the real step.
        """

        config = {
            "master_bias": {"R": "/M/bias_R.fits", "G": "/M/bias_G.fits"},
            "master_dark": None,  # not applied -> skipped
            "master_flat": "/M/flat.fits",  # bare filename also accepted
        }
        pairs = self._stamped_step_files(
            calibrate_step.calibrate,
            ["/RAW/img.fits"],
            config,
            Calibrator=lambda **_kwargs: None,
        )

        # Exhaustive, so the un-applied dark failing to be skipped shows up
        # as an unexpected extra entry.
        self.assertCountEqual(
            pairs,
            [
                (FileKind.RAW_IMAGE, "/RAW/img.fits"),
                (FileKind.MASTER_BIAS, "/M/bias_R.fits"),
                (FileKind.MASTER_BIAS, "/M/bias_G.fits"),
                (FileKind.MASTER_FLAT, "/M/flat.fits"),
            ],
        )

    def test_calibrate_without_masters_scopes_only_the_image(self):
        """With no masters configured, only the raw image is attached."""

        pairs = self._stamped_step_files(
            calibrate_step.calibrate,
            ["/RAW/img.fits"],
            Calibrator=lambda **_kwargs: None,
        )

        self.assertCountEqual(pairs, [(FileKind.RAW_IMAGE, "/RAW/img.fits")])

    def test_add_images_to_db_scopes_the_raw_image(self):
        """The raw image is in scope before any DB work begins.

        The odd one out: this step takes no progress callbacks, so it
        cannot use ``_stamped_step_files`` and fails through a patch
        instead. ``Evaluator`` is the first call inside the scope and runs
        *before* ``start_db_session``, so stubbing it also keeps the test
        away from the database entirely.
        """

        with patch.object(add_images_to_db_step, "Evaluator", self._raise_stub):
            pairs = self._stamped_files(
                lambda: add_images_to_db_step.add_images_to_db(
                    ["/RAW/a.fits"], {}
                )
            )

        self.assertCountEqual(pairs, [(FileKind.RAW_IMAGE, "/RAW/a.fits")])

    def test_stack_to_master_scopes_frames_and_master(self):
        """Stacking scopes every input frame plus the master it writes.

        ``mark_start`` is injected by the manager and is the first call
        inside the scope, so failing through it needs no patching of the
        step itself -- only ``get_master_fname``, which would otherwise
        open the first frame to build the name.
        """

        pairs = self._stamped_step_files(
            stack_step.stack_to_master,
            ["/CAL/a.fits", "/CAL/b.fits"],
            get_master_fname=lambda *args: "/M/master.fits",
        )

        self.assertCountEqual(
            pairs,
            [
                (FileKind.CALIBRATED_IMAGE, "/CAL/a.fits"),
                (FileKind.CALIBRATED_IMAGE, "/CAL/b.fits"),
                (FileKind.OUTPUT, "/M/master.fits"),
            ],
        )

    def test_stack_to_master_flat_scopes_both_masters(self):
        """The flat stacker attaches the high *and* low masters it writes.

        The step consumes a large, interdependent configuration, so this
        uses the step's own parser defaults rather than a hand-built dict.
        """

        masters = {"high": "/M/high.fits", "low": "/M/low.fits"}
        pairs = self._stamped_step_files(
            flat_step.stack_to_master_flat,
            ["/CAL/f.fits"],
            flat_step.parse_command_line([]),
            get_master_fnames=lambda *args: masters,
        )

        self.assertCountEqual(
            pairs,
            [
                (FileKind.CALIBRATED_IMAGE, "/CAL/f.fits"),
                (FileKind.OUTPUT, "/M/high.fits"),
                (FileKind.OUTPUT, "/M/low.fits"),
            ],
        )


class TestHDF5ProductsAttachThemselves(_StampedFilesMixin, unittest.TestCase):
    """DR and lightcurve files scope themselves while open.

    This is what replaced the per-step DR/LC scopes: every pipeline HDF5
    product opened through a ``with`` block attaches itself, so a step that
    simply opens one needs no scoping code of its own.

    Unlike the rest of the module these use real products, which needs a
    throwaway project home for the layout lookup.
    """

    @classmethod
    def setUpClass(cls):
        """Create a project home holding one (empty) DR file."""

        # Lives for the whole class and is released in tearDownClass, so a
        # ``with`` is not an option here.
        # pylint: disable=consider-using-with
        cls._home = TemporaryDirectory()
        # pylint: enable=consider-using-with
        set_project_home(cls._home.name)
        cls.dr_fname = os.path.join(cls._home.name, "frame.h5")
        with DataReductionFile(cls.dr_fname, "a"):
            pass

    @classmethod
    def tearDownClass(cls):
        """Remove the throwaway project home."""

        cls._home.cleanup()

    def _dr_pair(self):
        """The expected entry for the fixture DR file."""

        return (FileKind.DR_FILE, Path(self.dr_fname).as_posix())

    def test_open_for_reading_attaches_as_input(self):
        """A DR opened ``"r"`` is reported, as an input."""

        def read_and_fail():
            """Fail with the DR file open."""

            with DataReductionFile(self.dr_fname, "r"):
                raise _StubRaise

        self.assertCountEqual(
            self._stamped_files(read_and_fail), [self._dr_pair()]
        )

    def test_nested_products_all_attach(self):
        """A lightcurve opened inside a DR block adds to it, not replaces."""

        lc_fname = os.path.join(self._home.name, "src.h5")

        def write_and_fail():
            """Fail with both a DR and a lightcurve open."""

            with (
                DataReductionFile(self.dr_fname, "r"),
                LightCurveFile(lc_fname, "a", source_ids={"Gaia DR3": "123"}),
            ):
                raise _StubRaise

        self.assertCountEqual(
            self._stamped_files(write_and_fail),
            [
                self._dr_pair(),
                (FileKind.LIGHTCURVE, Path(lc_fname).as_posix()),
            ],
        )

    def test_in_memory_product_attaches_nothing(self):
        """The nameless in-memory products have no file to report."""

        def fail_in_memory():
            """Fail with only an in-memory product open."""

            with DataReductionFile():
                raise _StubRaise

        self.assertCountEqual(self._stamped_files(fail_in_memory), [])

    def test_psf_map_step_names_the_dr_it_failed_on(self):
        """``fit_source_extracted_psf_map`` reports the DR it was reading.

        Nothing is stubbed: the step is handed a DR that lacks the
        datasets it needs and fails on its own *inside* the block that
        opened it. That is what makes this catch a later refactor which
        narrows the ``with`` so the work no longer happens while the file
        is open -- a regression the hook's own tests cannot see.
        """

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
        config.update(
            (component + "_version", 0)
            for component in ["srcextract", "catalogue", "skytoframe"]
        )

        self.assertCountEqual(
            self._stamped_files(
                lambda: psf_map_step.fit_source_extracted_psf_map(
                    [self.dr_fname], None, config, MagicMock(), MagicMock()
                )
            ),
            [self._dr_pair()],
        )

    def test_merit_step_names_the_dr_it_failed_on(self):
        """``calculate_photref_merit`` reports the DR it was scoring."""

        config = {
            component + "_version": 0
            for component in [
                "srcextract",
                "catalogue",
                "skytoframe",
                "background",
                "srcproj",
            ]
        }

        self.assertCountEqual(
            self._stamped_files(
                lambda: merit_step.calculate_photref_merit(
                    [self.dr_fname], config
                )
            ),
            [self._dr_pair()],
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

        self.assertCountEqual(
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

        self.assertCountEqual(
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

        self.assertCountEqual(
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

        self.assertCountEqual(
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

        self.assertCountEqual(
            _pairs(classifier("/DR/x.h5")),
            [(FileKind.DR_FILE, "/DR/x.h5"), (FileKind.DR_FILE, "/DR/sp.h5")],
        )


if __name__ == "__main__":
    unittest.main()
