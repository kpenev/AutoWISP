#!/usr/bin/env python3

"""Run the AutoWISP test suite."""

import argparse
import os
import sys
from contextlib import nullcontext
from glob import glob
from os import makedirs, path
from tempfile import TemporaryDirectory

import unittest

from autowisp.tests import AutoWISPTestCase
from autowisp.tests.get_test_data import get_test_data

# Automatically used by pytest
# pylint: disable=unused-import
from autowisp.tests.test_calibrate import TestCalibrate
from autowisp.tests.test_stack_to_master import TestStackToMaster
from autowisp.tests.test_find_stars import (
    TestFindStars,
    TestStarlessFrameHandling,
)
from autowisp.tests.test_source_finder import (
    TestExtremeSourceCounts,
    TestQuantileBrightnessThreshold,
)
from autowisp.tests.test_solve_astrometry import TestSolveAstrometry
from autowisp.tests.test_fit_star_shape import TestFitStarShape
from autowisp.tests.test_measure_aperture_photometry import (
    TestMeasureAperturePhotometry,
)
from autowisp.tests.test_fit_source_extracted_psf_map import (
    TestFitSourceExtractedPSFMap,
)
from autowisp.tests.test_fit_magnitudes import TestFitMagnitudes
from autowisp.tests.test_create_lightcurves import TestCreateLightcurves
from autowisp.tests.test_epd import TestEPD
from autowisp.tests.test_tfa import TestTFA
from autowisp.tests.test_tfa_num_templates import (
    TestTemplateSourceIdsVlen,
    TestTemplateSelectionDiagnostics,
)
from autowisp.tests.test_detrending_stat import TestDetrendingStat
from autowisp.tests.test_catalog import TestCatalog
from autowisp.tests.test_lc_filter import (
    TestCatalogSourceListFilter,
    TestLCFilter,
)
from autowisp.tests.test_provenance_resolver import TestProvenanceResolver
from autowisp.tests.test_parse_config_overwrites import (
    TestParseConfigOverwrites,
)
from autowisp.tests.test_parameter_description_length import (
    TestParameterDescriptionLength,
)
from autowisp.tests.test_database_migration import (
    TestAdditiveMigrations,
    TestRevisionChain,
    TestMigrateProject,
    TestCheckProjectSchema,
    TestSqliteMigrationLock,
    TestConcurrentMigration,
    TestSchemaDrift,
)
from autowisp.tests.test_project_creation_guard import TestNewProjectGuard
from autowisp.tests.test_error_persistence import (
    TestPersistError,
    TestRunPipelineHandler,
    TestCleanupErrors,
    TestParseDuration,
)
from autowisp.tests.test_error_render import (
    TestErrorSummary,
    TestErrorDetail,
    TestFormatDetailText,
    TestErrorListRows,
    TestErrorCounts,
)
from autowisp.tests.test_error_cli import (
    TestExitCodeFor,
    TestReportError,
    TestCliEntryPoint,
)
from autowisp.tests.test_error_capture_middleware import (
    TestErrorCaptureMiddleware,
)
from autowisp.tests.test_crash_report import (
    TestScrubText,
    TestScrubMapping,
    TestScrubConfigValues,
    TestFindErrorProgress,
    TestCollectProvenance,
    TestBuildCrashReport,
    TestCrashReportCli,
)
from autowisp.tests.test_full_pipeline import TestFullPipeline
from autowisp.tests.test_exception_hierarchy import (
    TestExceptionHierarchy,
    TestMigratedExceptions,
    TestFrozenRow,
    TestSnapshotRow,
    TestToDetailDict,
)
from autowisp.tests.test_error_context import (
    TestErrorContextDataclass,
    TestFromConfig,
    TestAmbientAccessors,
    TestErrorContextManager,
    TestCaptureErrors,
    TestWorkerEntry,
    TestPoolPropagation,
    TestProcessQueuePropagation,
    TestNestingGuard,
)
from autowisp.tests.test_diagnostics_views import (
    TestQuantileSeriesExpansion,
    TestSharedTimeOffset,
    TestSeriesGrouping,
)
from autowisp.tests.test_diagnostic_expressions import (
    TestNameSpace,
    TestStoredFields,
)
from autowisp.tests.test_bui_db_config import (
    TestDiscovery,
    TestDefaultDatabase,
    TestUrlTranslation,
    TestMysqlDriver,
)
from autowisp.tests.test_bui_models import (
    TestModelBase,
    TestModifiedIsMaintained,
    TestMigrationCarriesExistingRows,
)

# pylint: enable=unused-import


class _IntegrationLastLoader(unittest.TestLoader):
    """Run ``TestFullPipeline`` after every other test class.

    ``unittest`` discovers classes via ``dir(module)`` which is
    alphabetical, so without this hook the long end-to-end test runs
    between ``TestFitStarShape`` and ``TestMeasureAperturePhotometry``
    -- an integration failure can then eclipse later per-step failures
    under ``failfast=True``. Sorting the resulting top-level suite so
    that the ``TestFullPipeline``-bearing entry is last keeps per-step
    failures visible first.
    """

    def loadTestsFromModule(self, *args, **kwargs):
        suite = super().loadTestsFromModule(*args, **kwargs)

        def is_full_pipeline(case_suite):
            return any(
                type(test).__name__ == "TestFullPipeline" for test in case_suite
            )

        return unittest.TestSuite(sorted(suite, key=is_full_pipeline))


def _parse_test_args(argv):
    """Split our own CLI flags from the ones passed through to ``unittest``.

    Returns ``(parsed, unittest_argv)``. ``parsed`` exposes
    ``failed_test_dir`` and ``data_dir`` (the latter is ``None`` unless
    the user passed ``--data-dir``).
    """

    parser = argparse.ArgumentParser(
        prog="python -m autowisp.tests",
        description=(
            "Run the AutoWISP test suite. Unknown options after the "
            "positional are forwarded to ``unittest.main`` (e.g. -v, -k)."
        ),
    )
    parser.add_argument(
        "failed_test_dir",
        help=(
            "Directory under which each failed test's processing directory "
            "is copied (into a ``<Class>_<method>`` subdirectory) for "
            "post-mortem inspection. All failures in a run are kept."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Use this directory for the unzipped test data instead of a "
            "self-cleaning temporary directory. Useful when the suite "
            "crashes -- the test data, the per-test processing dirs, "
            "and any leftover artifacts are then preserved for "
            "inspection. The directory is created if it does not "
            "exist; the test data is (re-)extracted into it on every "
            "run. If a leftover ``processing/`` directory from a "
            "previous crash is found inside, the run aborts with a "
            "message asking you to remove it."
        ),
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help=(
            "Use a local copy of the test data instead of downloading "
            "from Zenodo. Accepts either a path to a ``test_data.zip`` "
            "file (extracted into the data directory) or a path to an "
            "already-unzipped directory whose contents (CAL/, DR/, "
            "RAW/, ...) are copied into the data directory."
        ),
    )
    parser.add_argument(
        "--test-log",
        default=None,
        help=(
            "Send unittest framework output (test progress, pass/fail, "
            "tracebacks) to this file instead of stdout. Useful because "
            "the pipeline run inside each test redirects stdout/stderr "
            "to its own log file via ``setup_process_map``, taking the "
            "framework output with it. Default: a duplicate of stdout "
            "that survives the pipeline's redirect."
        ),
    )
    parser.add_argument(
        "--preserve-processing",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help=(
            "Preserve each test's processing directory for debugging. "
            "Each directory is moved to "
            "``<DIR>/<test_method_name>_processing`` after the test "
            "tears down (overwriting any prior copy). DIR defaults to "
            "the current working directory if no value is given."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="TestClass",
        help=(
            "Test case class names to skip. Since unittest ``-k`` has no "
            "negation, the runner expands this into ``-k`` selectors for every "
            "other test class, so e.g. ``--exclude TestCatalog "
            "TestFullPipeline`` runs everything but those two. Used by CI to "
            "run a subset per grid cell."
        ),
    )
    return parser.parse_known_args(argv)


def _exclude_to_selectors(exclude):
    """Return ``-k`` args selecting every harness test class except ``exclude``.

    ``unittest -k`` cannot express "everything but X", so enumerate the wanted
    classes instead. The runnable classes are the ``unittest.TestCase``
    subclasses imported into this module (the same ones the harness runs).
    """

    known = {
        name
        for name, obj in globals().items()
        if isinstance(obj, type)
        and issubclass(obj, unittest.TestCase)
        and name.startswith("Test")
    }
    unknown = set(exclude) - known
    if unknown:
        print(f"WARNING: --exclude names not found: {sorted(unknown)}")

    selectors = []
    for name in sorted(known - set(exclude)):
        selectors += ["-k", name]
    return selectors


def _open_test_stream(test_log):
    """Return a file object the unittest runner can use for its output.

    ``ImageProcessingManager.__init__`` calls ``setup_process``, which
    closes ``sys.stdout`` / ``sys.stderr`` before redirecting to its own
    log file. The runner's stream therefore has to live on an
    independent file descriptor; otherwise the framework's progress
    output (and any traceback on a failure) is silently swallowed when
    the first pipeline step runs.

    If ``test_log`` is given, open it for writing. Otherwise duplicate
    stdout's file descriptor so the runner ends up writing to wherever
    stdout was pointing before the test suite started.
    """

    if test_log:
        return open(  # pylint: disable=consider-using-with
            path.abspath(test_log), "w", encoding="utf-8", buffering=1
        )
    return os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)


def main():
    """Parse CLI args, prepare the test data dir, and run the suite."""

    args, unittest_argv = _parse_test_args(sys.argv[1:])

    if args.exclude:
        unittest_argv = _exclude_to_selectors(args.exclude) + unittest_argv

    print("Starting tests")
    if args.data_dir is not None:
        data_dir_path = path.abspath(args.data_dir)
        makedirs(data_dir_path, exist_ok=True)
        leftover_processing = path.join(data_dir_path, "processing")
        if path.exists(leftover_processing):
            sys.exit(
                f"ERROR: leftover processing directory "
                f"{leftover_processing!r} (likely from a previous crash). "
                "Remove it -- or move it somewhere safe to inspect -- "
                "before re-running."
            )
        data_dir_cm = nullcontext(data_dir_path)
    else:
        data_dir_cm = TemporaryDirectory()

    test_stream = _open_test_stream(args.test_log)
    try:
        with data_dir_cm as test_dir:
            get_test_data(test_dir, local_source=args.test_data)
            # Point solve_astrometry at the astrometry.net indices bundled in
            # the test data (env overrides the config path). Environments with
            # no local solve-field (e.g. Windows without ANSVR) fall back to the
            # web solver via astrometry.local_solver_available().
            anet_indices = path.join(test_dir, "anet_indices")
            if path.isdir(anet_indices):
                os.environ["AUTOWISP_ANET_INDICES"] = (
                    f"[{anet_indices}, {anet_indices}]"
                )
            processing_dir = path.join(test_dir, "processing")
            print(f"Test data directory: {test_dir!r}")
            print(f"Test data contents: {glob(test_dir + '/*')}")
            preserve_processing_dir = (
                path.abspath(args.preserve_processing)
                if args.preserve_processing is not None
                else None
            )
            if preserve_processing_dir is not None:
                makedirs(preserve_processing_dir, exist_ok=True)
            AutoWISPTestCase.set_test_directory(
                test_dir,
                processing_dir,
                args.failed_test_dir,
                preserve_processing_dir=preserve_processing_dir,
            )
            unittest.main(
                argv=[sys.argv[0]] + unittest_argv,
                failfast=True,
                testLoader=_IntegrationLastLoader(),
                testRunner=unittest.TextTestRunner(
                    stream=test_stream, verbosity=2
                ),
            )
    finally:
        test_stream.close()


if __name__ == "__main__":
    main()
