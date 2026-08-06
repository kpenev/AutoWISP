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
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.initialize_database import initialize_database
from autowisp.processing_steps import stack_to_master, stack_to_master_flat
from autowisp.database.interface import start_db_session
from autowisp.database.photref_selection import (
    bind_images_to_photref,
    compute_photref_candidates,
)
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
        # The per-step CLI uses different ``--outlier-threshold``
        # defaults for ``wisp-stack-to-master`` and
        # ``wisp-stack-to-master-flat``, but the pipeline registers a
        # single ``outlier-threshold`` parameter so the per-step
        # asymmetry is lost by default. Replicate it with two
        # mutually-exclusive conditional entries pulled straight from
        # each step's ``parse_command_line`` default -- so any change to
        # those defaults flows into the pipeline test automatically.
        # The ``(None, ...)`` row is dropped by
        # :func:`_overwrite_default_config`, which is what we want here:
        # every image type matches exactly one of the two conditions,
        # so there is no ambiguity.
        for param_name in ["outlier-threshold", "min-valid-values", "max-iter"]:
            non_flat_value = stack_to_master.parse_command_line([])[
                "argument_defaults"
            ][param_name]
            flat_value = stack_to_master_flat.parse_command_line([])[
                "argument_defaults"
            ][param_name]
            overwrite_default_config[param_name] = [
                (('IMAGETYP.strip() != "flat"',), str(non_flat_value)),
                (('IMAGETYP.strip() == "flat"',), str(flat_value)),
            ]
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

        # The DR file the test will register as the single photometric
        # reference comes from test.cfg, so adjusting the test or adding
        # new fixtures only requires editing the config file.
        self._photref_dr_path = path.join(
            self.processing_directory,
            overwrite_default_config["single-photref-dr-fname"][0][1],
        )

    def test_full_pipeline(self):
        """Run the full pipeline and compare every output to expected.

        The pipeline runs in two passes. On the first pass it does as
        much as it can; ``fit_magnitudes`` and the downstream lightcurve
        steps stay pending because no ``single_photref`` master has been
        registered yet. Between the passes the test registers the
        single photometric reference DR file via the same helpers the
        BUI uses, after which the second pass picks up
        ``fit_magnitudes`` and everything beyond.
        """

        raw_image_dirs = [
            path.join(self.test_directory, "RAW", subdir)
            for subdir in ("zero", "dark", "flat", "object")
        ]
        for add_raw_images in (raw_image_dirs, []):
            run_pipeline.main(
                Namespace(
                    project_home=self.processing_directory,
                    add_raw_images=add_raw_images,
                    steps=None,
                    step_imtypes=[],
                    detached=False,
                )
            )
            if add_raw_images:
                # Register the photref between the two passes -- the
                # second pass will then resume from ``fit_magnitudes``.
                self._register_photref()

        for imtype in ("zero", "dark", "flat", "object"):
            self._assert_dir_fits_match(path.join("CAL", imtype))

        for master_basename in (
            "zero_R.fits",
            "dark_R.fits",
            "flat_R.fits",
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

    def _register_photref(self):
        """Mimic the BUI's ``record_photref_selection`` for the test photref.

        Calls :func:`compute_photref_candidates` to obtain the same
        per-condition batches the BUI would surface to the user, finds
        the batch containing the test's chosen photref DR file (set in
        ``setUp`` from ``test.cfg``), registers the file as a
        ``single_photref`` master, and binds every eligible image in
        the batch via :func:`bind_images_to_photref`.
        """

        processing = ImageProcessingManager(pipeline_run_id=None)
        with start_db_session() as db_session:
            result = compute_photref_candidates(processing, db_session)

        photref_batch = None
        for candidate in result["candidates"]:
            for _, _, batch in candidate["groups"]:
                if any(entry[1] == self._photref_dr_path for entry in batch):
                    photref_batch = batch
                    break
            if photref_batch is not None:
                break
        self.assertIsNotNone(
            photref_batch,
            f"No candidate batch contains photref "
            f"{self._photref_dr_path!r}.",
        )

        processing.add_masters(
            {
                "type": "single_photref",
                "filename": self._photref_dr_path,
                "preference_order": None,
                "disable": False,
            }
        )
        bind_images_to_photref(self._photref_dr_path, photref_batch)

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

        def ignore(name):
            for ending in [
                "/EPD/FitProperties/Filter",
                "/TFA/FitProperties/PointsFilterExpression",
            ]:
                if name.endswith(ending):
                    return True
            return False

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
            self.assert_groups_match(gen_fname, exp_fname, "/", ignore=ignore)
            self.assert_groups_match(exp_fname, gen_fname, "/", ignore=ignore)
