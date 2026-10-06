"""Tests for fitting new images against the master their photref built.

The engine gives each ``fit_magnitudes`` batch the master photometric
reference built from the batch's single photometric reference, found through
the progress of the batch that built it: the images of that progress are
bound to the reference. A reference is not registered if its masters would
be named like those of another.

The project is the one of ``test_photref_binding``, with more single
photometric references next to its ``ref``.
"""

import unittest
from datetime import datetime
from os import path

from astropy.io import fits
from sqlalchemy import func, select, update

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.interface import start_db_session
from autowisp.database.photref_selection import (
    check_photref_fnames,
    record_single_photref,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Image,
    ImageMasterSelection,
    ImageProcessingProgress,
    ImageType,
    MasterFile,
    MasterType,
    PipelineRun,
    ProcessedImages,
    Step,
)

# pylint: enable=no-name-in-module
from autowisp.exceptions import ConfigurationError
from autowisp.tests.test_photref_binding import PhotrefBindingProject


class MasterPhotrefProject(PhotrefBindingProject):
    """Adds writing and registering masters, holding no tests itself."""

    @classmethod
    def _write_photref(cls, name, channel, exposure, fnum):
        """Write the DR file of a single photometric reference."""

        fname = path.join(cls._tmp.name, "DR", f"{name}_{channel}.h5")
        with DataReductionFile(fname, "w") as dr_file:
            dr_file.initialize(
                fits.Header(
                    {
                        "RAWFNAME": name,
                        "CLRCHNL": channel,
                        "TARGETID": "field",
                        "EXPTIME": exposure,
                        "FNUM": fnum,
                        "PROJHOME": cls._tmp.name,
                    }
                )
            )
        return fname

    @staticmethod
    def _register(db_session, master_type, fname, **columns):
        """Register a master of the given type, return its ID."""

        # pylint: disable=not-callable
        master = MasterFile(
            type_id=db_session.scalar(
                select(MasterType.id).filter_by(name=master_type)
            ),
            filename=fname,
            **{"enabled": True, **columns},
        )
        # pylint: enable=not-callable
        db_session.add(master)
        db_session.flush()
        return master.id


class TestBuiltMaster(MasterPhotrefProject):
    """Each batch is given the master its own single photref built, if any.

    ``ref1`` built a master twice, the first of which was disabled, along
    with a statistics file. ``ref2`` built one. ``ref3`` built none, though
    its batch was processed, so has a progress. A master registered outside
    of any batch was built from no reference the engine knows of.
    """

    #: The image bound to each single photref, in channel R.
    _bound = {"ref1": "near", "ref2": "far", "ref3": "longer"}

    @classmethod
    def _add_progress(cls, db_session, image_name):
        """Record a fit_magnitudes batch of the image in R, return its ID."""

        # pylint: disable=not-callable
        progress = ImageProcessingProgress(
            run_id=cls._run_id,
            step_id=db_session.scalar(
                select(Step.id).filter_by(name="fit_magnitudes")
            ),
            image_type_id=db_session.scalar(
                select(ImageType.id).filter_by(name="object")
            ),
            configuration_version=0,
            started=datetime(2023, 3, 2),
        )
        db_session.add(progress)
        db_session.flush()
        db_session.add(
            ProcessedImages(
                image_id=cls._image_ids[image_name],
                channel="R",
                progress_id=progress.id,
                status=1,
                final=True,
            )
        )
        # pylint: enable=not-callable
        return progress.id

    @classmethod
    def _fill_database(cls):
        """Add the references, the batches and the masters they built."""

        super()._fill_database()
        masters_dir = path.join(cls._tmp.name, "MASTERS")
        cls._photrefs = {
            "ref1": cls._photref_fname,
            "ref2": cls._write_photref("ref2", "R", 30.0, 200),
            "ref3": cls._write_photref("ref3", "R", 30.0, 300),
        }
        cls._masters = {
            "ref1": path.join(masters_dir, "mphotref_ref1.fits"),
            "ref1_old": path.join(masters_dir, "mphotref_ref1_old.fits"),
            "ref2": path.join(masters_dir, "mphotref_ref2.fits"),
        }
        with start_db_session() as db_session:
            # pylint: disable=not-callable
            run = PipelineRun(host="test", process_id=1, started=datetime.now())
            # pylint: enable=not-callable
            db_session.add(run)
            db_session.flush()
            cls._run_id = run.id

            for name in ("ref2", "ref3"):
                cls._register(db_session, "single_photref", cls._photrefs[name])

            cls._old_master_id = cls._register(
                db_session,
                "master_photref",
                cls._masters["ref1_old"],
                progress_id=cls._add_progress(db_session, "near"),
                enabled=False,
            )
            ref1_progress = cls._add_progress(db_session, "near")
            cls._register(
                db_session,
                "master_photref",
                cls._masters["ref1"],
                progress_id=ref1_progress,
            )
            cls._register(
                db_session,
                "magfit_stat",
                path.join(masters_dir, "mfit_stat_ref1.txt"),
                progress_id=ref1_progress,
            )
            cls._register(
                db_session,
                "master_photref",
                cls._masters["ref2"],
                progress_id=cls._add_progress(db_session, "far"),
            )
            cls._add_progress(db_session, "longer")
            cls._register(
                db_session,
                "master_photref",
                path.join(masters_dir, "mphotref_standalone.fits"),
            )

    def setUp(self):
        """Bind each image to its reference; the base removes bindings."""

        super().setUp()
        with start_db_session() as db_session:
            photref_type_id = db_session.scalar(
                select(MasterType.id).filter_by(name="single_photref")
            )
            for photref, image_name in self._bound.items():
                db_session.add(
                    # pylint: disable=not-callable
                    ImageMasterSelection(
                        image_id=self._image_ids[image_name],
                        channel="R",
                        master_type_id=photref_type_id,
                        master_file_id=db_session.scalar(
                            select(MasterFile.id).filter_by(
                                filename=self._photrefs[photref]
                            )
                        ),
                    )
                    # pylint: enable=not-callable
                )

    def given_masters(self):
        """Return ``{photref: master}`` the batch of each reference is given."""

        result = {}
        with start_db_session() as db_session:
            step = db_session.scalar(
                select(Step).filter_by(name="fit_magnitudes")
            )
            for photref, image_name in self._bound.items():
                image = db_session.get(Image, self._image_ids[image_name])
                self._processing.evaluate_expressions_image(image, db_session)
                # pylint: disable-next=protected-access
                configured = self._processing._get_batch_config(
                    [(image, "R", None)], (), step, db_session
                )
                self.assertEqual(len(configured), 1, photref)
                ((config, _),) = configured.values()
                self.assertEqual(
                    config["single_photref_dr_fname"], self._photrefs[photref]
                )
                result[photref] = config["master_photref_fname"]
        return result

    def test_each_batch_gets_the_master_its_photref_built(self):
        """Not the disabled one, the statistics, or the stand-alone master."""

        self.assertEqual(
            self.given_masters(),
            {
                "ref1": self._masters["ref1"],
                "ref2": self._masters["ref2"],
                "ref3": None,
            },
        )

    def test_two_enabled_masters_of_one_photref_are_refused(self):
        """Rather than either being picked."""

        with start_db_session() as db_session:
            db_session.execute(
                update(MasterFile)
                .where(MasterFile.id == self._old_master_id)
                .values(enabled=True)
            )
        try:
            with self.assertRaises(ConfigurationError):
                self.given_masters()
        finally:
            with start_db_session() as db_session:
                db_session.execute(
                    update(MasterFile)
                    .where(MasterFile.id == self._old_master_id)
                    .values(enabled=False)
                )

    def test_default_names_tell_photrefs_apart(self):
        """References of one target, channel and exposure time."""

        for photref, fname in self._photrefs.items():
            with self.subTest(photref=photref):
                check_photref_fnames(self._processing, fname)


class TestPhotrefNameClash(MasterPhotrefProject):
    """Under formats naming masters by target, channel and exposure time.

    Of the candidates, only the one matching the registered ``ref`` in all
    three would have its masters named like ``ref``'s.
    """

    #: The defaults before the frame number was added.
    _old_formats = {
        "master-photref-fname-format": (
            "{PROJHOME}/MASTERS/mphotref_"
            "{TARGETID}_{CLRCHNL}_{EXPTIME}sec_iter{magfit_iteration:03d}.fits"
        ),
        "magfit-stat-fname-format": (
            "{PROJHOME}/MASTERS/mfit_stat_{TARGETID}_{CLRCHNL}_{EXPTIME}sec_"
            "iter{magfit_iteration:03d}.txt"
        ),
    }

    @classmethod
    def _config_overwrites(cls):
        """Add the old formats."""

        return {
            **super()._config_overwrites(),
            **{
                option: [(None, value)]
                for option, value in cls._old_formats.items()
            },
        }

    @classmethod
    def _fill_database(cls):
        """Write the candidates, registering none of them."""

        super()._fill_database()
        cls._candidates = {
            "cand_same": cls._write_photref("cand_same", "R", 30.0, 200),
            "cand_g": cls._write_photref("cand_g", "G", 30.0, 201),
            "cand_60s": cls._write_photref("cand_60s", "R", 60.0, 202),
        }

    def test_only_the_photref_named_alike_is_refused(self):
        """The others differ in channel or exposure time."""

        refused = set()
        for name, fname in self._candidates.items():
            try:
                check_photref_fnames(self._processing, fname)
            except ConfigurationError:
                refused.add(name)
        self.assertEqual(refused, {"cand_same"})

    def test_a_refused_photref_is_not_registered(self):
        """Nor are any images bound to it."""

        def count(model):
            # pylint: disable-next=not-callable
            return db_session.scalar(select(func.count()).select_from(model))

        with start_db_session() as db_session:
            before = (count(MasterFile), count(ImageMasterSelection))
        with self.assertRaises(ConfigurationError):
            record_single_photref(
                self._candidates["cand_same"],
                [(None, None, self._image_ids["near"], "R")],
            )
        with start_db_session() as db_session:
            self.assertEqual(
                (count(MasterFile), count(ImageMasterSelection)), before
            )


if __name__ == "__main__":
    unittest.main()
