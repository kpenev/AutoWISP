#!/usr/bin/env python3

"""Run the AutoWISP test suite."""

import argparse
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
from autowisp.tests.test_find_stars import TestFindStars
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
from autowisp.tests.test_detrending_stat import TestDetrendingStat
from autowisp.tests.test_catalog import TestCatalog
from autowisp.tests.test_provenance_resolver import TestProvenanceResolver
from autowisp.tests.test_parse_config_overwrites import (
    TestParseConfigOverwrites,
)
from autowisp.tests.test_full_pipeline import TestFullPipeline

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
            "Directory to which the processing directory of a failed "
            "test is copied for post-mortem inspection."
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
    return parser.parse_known_args(argv)


def main():
    """Parse CLI args, prepare the test data dir, and run the suite."""

    args, unittest_argv = _parse_test_args(sys.argv[1:])

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

    with data_dir_cm as test_dir:
        get_test_data(test_dir)
        processing_dir = path.join(test_dir, "processing")
        print(f"Test data directory: {test_dir!r}")
        print(f"Test data contents: {glob(test_dir + '/*')}")
        AutoWISPTestCase.set_test_directory(
            test_dir, processing_dir, args.failed_test_dir
        )
        unittest.main(
            argv=[sys.argv[0]] + unittest_argv,
            failfast=True,
            testLoader=_IntegrationLastLoader(),
        )


if __name__ == "__main__":
    main()
