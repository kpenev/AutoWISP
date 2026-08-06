"""Autowisp unit-test init."""

from collections.abc import Sequence
from os import path, makedirs
from subprocess import run, PIPE, STDOUT
from shutil import copytree, copy, move, rmtree
from glob import glob
import logging

from asteval import Interpreter
from astrowisp.tests.utilities import FloatTestCase

from autowisp.database.interface import (
    set_project_home,
    initialize_cmdline_database,
)
from autowisp.database.user_interface import import_json_to_survey
from autowisp.database.initialize_database import initialize_database


class AutoWISPTestCase(FloatTestCase):
    """Base class for AutoWISP tests."""

    #: Set False by a test that does not want its processing directory
    #: kept even when it fails (nothing does at present). Failure is
    #: detected from the test result, so a passing test never has to say
    #: anything -- the previous arrangement, where each test opted in by
    #: setting ``successful_test``, silently preserved the directory of
    #: every test that forgot to.
    preserve_failed_processing = True
    _logger = logging.getLogger(__name__)

    # Stage the cached Gaia catalog FITS (``test_data/MASTERS/Gaia``) into the
    # processing directory in setUp so steps that call ``ensure_catalog`` reuse
    # them instead of hitting the live Gaia archive. Catalog tests, which are
    # meant to exercise the live query, set this False.
    stage_catalog_cache = True

    # Header keys whose presence and value must NOT be required to
    # match between the two files:
    #
    # - ``EXTEND`` -- structural, not data.
    # - ``CALGITID`` -- git commit of the calibration code.
    # - ``COMMENT`` -- free-form text.
    # - ``M{BIAS,DARK,FLAT}SHA`` -- hash of master files.
    # - ``TARGETID``, ``TELSCPID``, ``CAMERAID``, ``OBSSSNID``,
    #   ``OBSERVER`` -- DB row identifiers written by the pipeline's
    #   ``add_images_to_db`` step but absent from the per-step CLI
    #   flow used to produce the test_data fixtures.
    _ignore_header_keys = {
        "DATASUM",
        "CHECKSUM",
        "CALGITID",
        "COMMENT",
        "EXTEND",
        "TARGETID",
        "TELSCPID",
        "CAMERAID",
        "OBSSSNID",
        "OBSERVER",
        "PROJHOME",
    } | {f"M{master.upper()}SHA" for master in ("bias", "dark", "flat")}

    # Keys whose stored value must be evaluated with ``asteval`` before
    # the comparison. Subclasses can extend this set.
    _evaluated_header_keys = {"OUTLTHRS"}

    def _compare_headers(self, fname1, fname2, header1, header2):
        """Assert that the two FITS headers match.

        Args:
            fname1, fname2(str):    Names of the files (for messages).

            header1, header2:    The ``astropy.io.fits.Header`` objects.
        """

        keys = [
            set(h.keys()) - self._ignore_header_keys for h in (header1, header2)
        ]
        self.assertTrue(
            keys[0] == keys[1],
            f"Headers of {fname1} and {fname2} do not have the same keys!\n"
            f"    Only in {fname1}: {keys[0] - keys[1]}\n"
            f"    Only in {fname2}: {keys[1] - keys[0]}",
        )

        master_fname_keys = {
            f"M{tp.upper()}FNM" for tp in ("bias", "dark", "flat")
        }
        original_files = [set(), set()]
        aeval = Interpreter()

        for key, value in header1.items():
            if key.strip() == "" or key in self._ignore_header_keys:
                continue
            other = header2[key]
            if key in master_fname_keys:
                self.assertEqual(
                    path.basename(other),
                    path.basename(value),
                    f"Master {key[1:-3].lower()} does not match between "
                    f"{fname1} and {fname2}: "
                    f"{path.basename(value)!r} vs {path.basename(other)!r}.",
                )
            elif key.startswith("ORIGF"):
                original_files[0].add(path.basename(value))
                original_files[1].add(path.basename(other))
            elif key in self._evaluated_header_keys:
                evaluated = [aeval(str(v)) for v in (value, other)]
                msg = (
                    f"Evaluated value for key {key!r} does not match between "
                    f"{fname1} and {fname2}: {value!r} -> {evaluated[0]!r} vs "
                    f"{other!r} -> {evaluated[1]!r}."
                )
                if all(
                    isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                    for v in evaluated
                ):
                    self.assertEqual(
                        list(evaluated[0]), list(evaluated[1]), msg
                    )
                else:
                    self.assertEqual(evaluated[0], evaluated[1], msg)
            else:
                self.assertEqual(
                    other,
                    value,
                    f"Value for key {key!r} does not match between "
                    f"{fname1} and {fname2}: {value!r} vs {other!r}.",
                )

        self.assertEqual(
            *original_files,
            f"Original input files in {fname1} and {fname2} do not match!\n"
            f"    Only in {fname1}: "
            f"{sorted(original_files[0] - original_files[1])}\n"
            f"    Only in {fname2}: "
            f"{sorted(original_files[1] - original_files[0])}",
        )

    def get_inputs(self, inputs):
        """Get the input files for the test step and return what to clean up."""

        for product in inputs:
            for source in glob(path.join(self.test_directory, product)):
                destination = source.replace(
                    self.test_directory,
                    self.processing_directory,
                )
                assert path.exists(source)
                self._logger.debug("Copying %r to %r", source, destination)
                if path.isdir(source):
                    copytree(source, destination)
                else:
                    assert path.isfile(source)
                    destination = path.dirname(destination)
                    makedirs(destination, exist_ok=True)
                    copy(source, destination)

    preserve_processing_dir = None

    @classmethod
    def set_test_directory(
        cls,
        test_dirname,
        processing_dirname,
        failed_test_dirname,
        preserve_processing_dir=None,
    ):
        """Set the directory where data to test against is located."""

        cls.test_directory = test_dirname
        cls.processing_directory = processing_dirname
        cls.failed_test_directory = failed_test_dirname
        cls.preserve_processing_dir = preserve_processing_dir

    def setUp(self):
        """Make sure the data to compare against is defined."""

        print(f"Setting up processing in {self.processing_directory!r}")
        self.assertTrue(
            hasattr(self, "test_directory"), "No test data directory defined!"
        )
        self.assertTrue(
            hasattr(self, "processing_directory"),
            "No processing directory defined!",
        )
        self.assertTrue(
            path.exists(self.test_directory),
            f"Test directory {self.test_directory} does not exist!",
        )
        makedirs(self.processing_directory, exist_ok=False)
        gaia_cache = path.join(self.test_directory, "MASTERS", "Gaia")
        if self.stage_catalog_cache and path.isdir(gaia_cache):
            makedirs(
                path.join(self.processing_directory, "MASTERS"), exist_ok=True
            )
            copytree(
                gaia_cache,
                path.join(self.processing_directory, "MASTERS", "Gaia"),
            )
        copy(
            path.join(self.test_directory, "test.cfg"),
            path.join(self.processing_directory, "test.cfg"),
        )
        set_project_home(self.processing_directory)
        with open(
            path.join(self.test_directory, "survey_instruments.json"),
            "r",
            encoding="utf-8",
        ) as survey_json:
            import_json_to_survey(survey_json)

    def _test_failed(self):
        """Whether *this* test has just failed or errored.

        Read off the result rather than a flag the test sets, so nothing
        has to be remembered at the end of every test method. Two things
        make the obvious shortcuts wrong: ``_outcome.success`` is still
        True here (it is reset per test *part*, and ``tearDown`` is its
        own part), and ``result.errors`` / ``result.failures`` accumulate
        over the whole run -- so the entries have to be matched against
        this test rather than merely counted.

        Returns:
            bool:    True if this test recorded a failure or an error.
        """

        result = getattr(getattr(self, "_outcome", None), "result", None)
        if result is None:
            return False
        return any(
            test is self
            for group in ("errors", "failures")
            for test, _ in getattr(result, group, ())
        )

    def tearDown(self):
        """Remove the processing directory."""

        print(f"Tearing down processing in {self.processing_directory!r}")
        if self.preserve_failed_processing and self._test_failed():
            # Preserve every failed test in its own subdirectory (keyed by
            # class + method) so a run with several failures keeps all of them
            # for post-mortem, rather than each failure overwriting the last.
            destination = path.join(
                self.failed_test_directory,
                f"{type(self).__name__}_{self._testMethodName}",
            )
            if path.exists(destination):
                rmtree(destination, ignore_errors=False)
            copytree(self.processing_directory, destination)
        if self.preserve_processing_dir is not None:
            destination = path.join(
                self.preserve_processing_dir,
                self._testMethodName + "_processing",
            )
            if path.exists(destination):
                rmtree(destination)
            move(self.processing_directory, destination)
        else:
            rmtree(self.processing_directory)

    def run_step(self, command):
        """Run a calibration step and check the return code."""

        calib_process = run(
            command,
            cwd=self.processing_directory,
            check=False,
            stdout=PIPE,
            stderr=STDOUT,
            timeout=3600,
        )
        self.assertTrue(
            calib_process.returncode == 0,
            f"AutoWISP step command:\n{command!r}\n"
            f"Started from {self.processing_directory!r} "
            f"failed:\n{calib_process.stdout.decode('utf-8')}",
        )
