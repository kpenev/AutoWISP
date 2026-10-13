"""End-to-end pipeline tests driven by :mod:`autowisp.run_pipeline`.

Mirrors what the BUI does when a user creates a new project: builds the
project's SQLite database from ``test_data/test.cfg`` and
``test_data/master_config.json`` via the same helpers the BUI uses
(``parse_config_overwrites`` + ``master_config_json_to_settings`` +
``apply_master_config`` + ``initialize_database``), imports the survey
JSON, registers every raw image, runs the full pipeline (calibration
through TFA / detrending statistics), and compares the generated
artifacts to the expected outputs in ``test_data/``.

The pipeline runs either all at once, as a user starts it, or one step at
a time with the project moved after each step, which shows that a moved
project carries on wherever it is moved.
"""

import json
from argparse import Namespace
from glob import glob
from os import path
from shutil import move

import pandas
from sqlalchemy import func, select

from autowisp import run_pipeline
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.initialize_database import initialize_database
from autowisp.processing_steps import stack_to_master, stack_to_master_flat
from autowisp.project_paths import to_stored_path
from autowisp.database.interface import (
    get_db_engine,
    set_project_home,
    start_db_session,
)
from autowisp.database.photref_selection import (
    compute_photref_candidates,
    record_single_photref,
)
from autowisp.database.user_interface import (
    apply_master_config,
    get_processing_sequence,
    import_json_to_survey,
    master_config_json_to_settings,
    parse_config_overwrites,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    ImageProcessingProgress,
    ImageType,
    LightCurveProcessingProgress,
    Step,
)

# pylint: enable=no-name-in-module
from autowisp.tests.fits_test_case import FITSTestCase
from autowisp.tests.h5_test_case import H5TestCase
from autowisp.tests.magfit_test_case import MagfitMastersTestCase
from autowisp.tests.unstored_paths import find_unstored_paths


class _EngineProjectTestCase(H5TestCase, FITSTestCase, MagfitMastersTestCase):
    """A project created as the BUI does, processed through the engine."""

    # The engine writes the observing session's identifiers, so they are
    # compared like any other keyword.
    _ignore_header_keys = (
        H5TestCase._ignore_header_keys - H5TestCase._engine_header_keys
    )

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
        # The DR file the test will register as the single photometric
        # reference comes from test.cfg, so adjusting the test or adding
        # new fixtures only requires editing the config file. The engine
        # sets the option itself, so a project does not store it. In stored
        # form, as the processing directory is the project home.
        self._photref_dr_path = overwrite_default_config.pop(
            "single-photref-dr-fname"
        )[0][1]
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

    def _get_raw_image_dirs(self):
        """Return the directories of the raw images to process."""

        return [
            path.join(self.test_directory, "RAW", subdir)
            for subdir in ("zero", "dark", "flat", "object")
        ]

    def _assert_outputs_match(self):
        """Compare every output to expected, and check the paths stored."""

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
        self._assert_dir_fits_match("MASTERS", "lc_catalog_*.fits")
        self._assert_masters_as_expected()

        for stats_basename in ("epd_statistics.txt", "tfa_statistics.txt"):
            self._assert_detrending_stats_match(stats_basename)

        self._assert_h5_dir_match("DR")
        self._assert_h5_dir_match("LC")

        self._assert_paths_stored()

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

        # The batches name DR files in stored form, as the BUI shows them.
        photref_fname = to_stored_path(self._photref_dr_path)
        photref_batch = None
        for candidate in result["candidates"]:
            for _, batch in candidate["groups"]:
                if any(entry[1] == photref_fname for entry in batch):
                    photref_batch = batch
                    break
            if photref_batch is not None:
                break
        self.assertIsNotNone(
            photref_batch,
            f"No candidate batch contains photref "
            f"{self._photref_dr_path!r}.",
        )

        record_single_photref(photref_fname, photref_batch)

    def _assert_dir_fits_match(self, relative_dir, pattern="*.fits*"):
        """Assert every ``pattern`` in ``relative_dir`` matches expected."""

        generated = sorted(
            glob(path.join(self.processing_directory, relative_dir, pattern))
        )
        expected = sorted(
            glob(path.join(self.test_directory, relative_dir, pattern))
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

    def _assert_paths_stored(self):
        """Assert the project holds every path in stored form.

        Over everything the pipeline wrote, the database included, so that
        a product, column or attribute added later is covered without being
        registered anywhere.
        """

        unstored = find_unstored_paths(
            self.processing_directory, get_db_engine()
        )
        self.assertFalse(
            unstored,
            "Paths not in stored form:\n"
            + "\n".join(
                f"{found.source}: {found.location}: {found.value!r} "
                f"({found.kind})"
                for found in unstored
            ),
        )


class TestFullPipeline(_EngineProjectTestCase):
    """Run the full pipeline end-to-end and check every output."""

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

        for add_raw_images in (self._get_raw_image_dirs(), []):
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

        self._assert_outputs_match()


class TestRelocatedPipeline(_EngineProjectTestCase):
    """Move the project before every step, and check that it carries on."""

    # pylint: disable=invalid-name
    def setUp(self):
        """Start with the project where it was created."""

        super().setUp()
        self._left_behind = []

    # pylint: enable=invalid-name

    def test_moved_before_every_step(self):
        """
        Run each step of the processing sequence where the project was moved.

        The raw images are added on their own, and every step then runs
        right after the project was moved, with everything before it written
        elsewhere, and is the only one to run. The single photometric
        reference is registered before magnitude fitting, with a move on
        each side. A step reading a path recorded by absolute path fails,
        the locations left behind being gone, and one writing to it
        recreates a location left behind, which is checked after each step.
        """

        self._run_pipeline(add_raw_images=self._get_raw_image_dirs(), steps=[])
        with start_db_session() as db_session:
            sequence = [
                (step.name, image_type.name)
                for step, image_type in get_processing_sequence(db_session)
            ]
        photref_registered = False
        for step_name, image_type_name in sequence:
            self._move_project()
            if step_name == "fit_magnitudes" and not photref_registered:
                self._register_photref()
                photref_registered = True
                self._move_project()
            progress_before = self._get_last_progress_ids()
            self._run_pipeline(
                steps=[step_name],
                step_imtypes=[f"{step_name}:{image_type_name}"],
            )
            self._assert_only_step_progressed(
                step_name, image_type_name, progress_before
            )
            self.assertEqual(
                [
                    location
                    for location in self._left_behind
                    if path.exists(location)
                ],
                [],
                f"{step_name} of {image_type_name} frames wrote where the "
                "project was before it was moved.",
            )

        self._assert_outputs_match()

    def _run_pipeline(self, *, add_raw_images=(), steps, step_imtypes=()):
        """Run the pipeline in the project, wherever it is now."""

        run_pipeline.main(
            Namespace(
                project_home=self.processing_directory,
                add_raw_images=list(add_raw_images),
                steps=steps,
                step_imtypes=list(step_imtypes),
                detached=False,
            )
        )

    def _move_project(self):
        """Move the project to a new location, and open it there."""

        self._left_behind.append(self.processing_directory)
        # A sibling, so that the move is a rename.
        destination = (
            f"{self._left_behind[0]}.moved{len(self._left_behind):02d}"
        )
        move(self.processing_directory, destination)
        # Followed by tearDown, which keeps or removes the project.
        self.processing_directory = destination
        set_project_home(destination)

    @staticmethod
    def _get_last_progress_ids():
        """Return the last ids of image and lightcurve processing progress."""

        with start_db_session() as db_session:
            return tuple(
                db_session.scalar(select(func.max(table.id))) or 0
                for table in (
                    ImageProcessingProgress,
                    LightCurveProcessingProgress,
                )
            )

    def _assert_only_step_progressed(self, step_name, image_type_name, since):
        """Assert that a run recorded progress of the given step alone."""

        image_since, light_curve_since = since
        with start_db_session() as db_session:
            image_progressed = {
                tuple(row)
                for row in db_session.execute(
                    select(Step.name, ImageType.name)
                    .select_from(ImageProcessingProgress)
                    .join(Step, ImageProcessingProgress.step_id == Step.id)
                    .join(
                        ImageType,
                        ImageProcessingProgress.image_type_id == ImageType.id,
                    )
                    .where(ImageProcessingProgress.id > image_since)
                )
            }
            light_curve_progressed = set(
                db_session.scalars(
                    select(Step.name)
                    .select_from(LightCurveProcessingProgress)
                    .join(Step, LightCurveProcessingProgress.step_id == Step.id)
                    .where(LightCurveProcessingProgress.id > light_curve_since)
                )
            )
        self.assertTrue(
            image_progressed or light_curve_progressed,
            f"{step_name} of {image_type_name} frames made no progress.",
        )
        self.assertLessEqual(image_progressed, {(step_name, image_type_name)})
        self.assertLessEqual(light_curve_progressed, {step_name})
