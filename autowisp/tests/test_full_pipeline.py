"""End-to-end pipeline test driven by :mod:`autowisp.run_pipeline`.

Mirrors what the BUI does when a user creates a new project: builds the
project's SQLite database from ``test_data/test.cfg`` and
``test_data/master_config.json`` via the same helpers the BUI uses
(``parse_config_overwrites`` + ``master_config_json_to_settings`` +
``apply_master_config`` + ``initialize_database``), imports the survey
JSON, registers every raw image, runs the full pipeline (calibration
through TFA / detrending statistics), and compares the generated
artifacts to the expected outputs in ``test_data/``.
"""

import json
from argparse import Namespace
from glob import glob
from os import path

import pandas

from autowisp import run_pipeline
from autowisp.database.initialize_database import initialize_database
from autowisp.database.user_interface import (
    apply_master_config,
    import_json_to_survey,
    master_config_json_to_settings,
    parse_config_overwrites,
)
from autowisp.tests.fits_test_case import FITSTestCase
from autowisp.tests.h5_test_case import H5TestCase


class TestFullPipeline(H5TestCase, FITSTestCase):
    """Run the full pipeline end-to-end and check every output."""

    def setUp(self):
        """Bring up a project the same way the BUI does on creation.

        ``AutoWISPTestCase.setUp`` already handles the processing-dir
        creation, ``test.cfg`` copy, ``set_project_home``, and survey
        import (against a minimal schema). On top of that we run a full
        ``initialize_database(drop_all_tables=True)`` -- mirroring the
        BUI's project-creation flow -- and re-import the survey since
        that drops the provenance rows the base class just imported.
        """

        super().setUp()

        with open(
            path.join(self.test_directory, "test.cfg"),
            "r",
            encoding="utf-8",
        ) as cfg_file:
            overwrite_default_config = parse_config_overwrites(cfg_file)
        with open(
            path.join(self.test_directory, "master_config.json"),
            "r",
            encoding="utf-8",
        ) as master_json:
            master_settings = master_config_json_to_settings(
                json.load(master_json)
            )
        step_dependencies, master_info = apply_master_config(master_settings)

        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True),
            step_dependencies,
            master_info,
            overwrite_default_config,
        )

        with open(
            path.join(self.test_directory, "survey_instruments.json"),
            "r",
            encoding="utf-8",
        ) as survey_json:
            import_json_to_survey(survey_json)

    def test_full_pipeline(self):
        """Run the full pipeline and compare every output to expected."""

        config = Namespace(
            project_home=self.processing_directory,
            add_raw_images=[
                path.join(self.test_directory, "RAW", subdir)
                for subdir in ("zero", "dark", "flat", "object")
            ],
            steps=None,
            step_imtypes=[],
            detached=False,
        )
        run_pipeline.main(config)

        for imtype in ("zero", "dark", "flat", "object"):
            self._assert_dir_fits_match(path.join("CAL", imtype))

        for master_basename in (
            "zero_R.fits.fz",
            "dark_R.fits.fz",
            "flat_R.fits.fz",
        ):
            self.assert_fits_match(
                path.join(self.test_directory, "MASTERS", master_basename),
                path.join(
                    self.processing_directory, "MASTERS", master_basename
                ),
            )

        for stats_basename in ("epd_statistics.txt", "tfa_statistics.txt"):
            self._assert_detrending_stats_match(stats_basename)

        self._assert_h5_dir_match("DR")
        self._assert_h5_dir_match("LC")

        self.successful_test = True

    def _assert_dir_fits_match(self, relative_dir):
        """Assert every ``*.fits*`` in ``relative_dir`` matches expected."""

        generated = sorted(
            glob(path.join(self.processing_directory, relative_dir, "*.fits*"))
        )
        expected = sorted(
            glob(path.join(self.test_directory, relative_dir, "*.fits*"))
        )
        self.assertEqual(
            [path.basename(fname) for fname in generated],
            [path.basename(fname) for fname in expected],
            f"FITS file list mismatch under {relative_dir!r}.",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_fits_match(exp_fname, gen_fname)

    def _assert_detrending_stats_match(self, basename):
        """Compare an ``epd`` / ``tfa`` statistics file via pandas.

        Matches the comparison rule used by
        :class:`autowisp.tests.test_detrending_stat.TestDetrendingStat`.
        """

        generated, expected = (
            pandas.read_csv(
                path.join(dirname, "MASTERS", basename),
                sep=r"\s+",
                index_col="ID",
            ).sort_values(by="ID")
            for dirname in (self.processing_directory, self.test_directory)
        )
        self.assertApproxPandas(expected, generated, basename)

    def _assert_h5_dir_match(self, relative_dir):
        """Assert every ``*.h5`` file under ``relative_dir`` matches."""

        generated = sorted(
            glob(path.join(self.processing_directory, relative_dir, "*.h5"))
        )
        expected = sorted(
            glob(path.join(self.test_directory, relative_dir, "*.h5"))
        )
        self.assertEqual(
            [path.basename(fname) for fname in generated],
            [path.basename(fname) for fname in expected],
            f"HDF5 file list mismatch under {relative_dir!r}.",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_groups_match(gen_fname, exp_fname, "/", ignore=None)
            self.assert_groups_match(exp_fname, gen_fname, "/", ignore=None)
