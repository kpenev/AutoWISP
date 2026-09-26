"""Tests for binding images pending magnitude fitting to a photometric ref.

The engine records, in ``ImageMasterSelection``, which single photometric
reference every image/channel ``fit_magnitudes`` sees is fit against. With a
finite ``max_photref_separation`` it picks the reference by separation; with an
unlimited one the condition expressions select it, as for any other master, and
that choice is recorded too. Either way the row must agree with the reference
magfit is actually given, since everything grouping images by reference reads
it rather than re-deciding.

These drive the manager the way ``_prepare_processing`` does -- evaluating each
image's expressions from its raw header, then binding -- on a throwaway project
holding one two-channel camera and a single reference registered in channel
``R`` only.
"""

import logging
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from os import makedirs, path

import numpy
from astropy.io import fits
from sqlalchemy import delete, select

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageMasterSelection,
    ImageType,
    MasterFile,
    MasterType,
    ObservingSession,
    Step,
    Target,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraChannel,
    CameraType,
    Observer,
    Telescope,
    TelescopeType,
)

# pylint: enable=no-name-in-module


class PhotrefBindingProject(unittest.TestCase):
    """A project with one registered single photref, holding no tests itself.

    Each test gets a fresh manager, as ``_evaluated_expressions`` caches the
    master chosen for each image when it is first evaluated, and starts with
    no bindings.
    """

    #: ``max-photref-separation`` to configure, or None for the default.
    _max_photref_separation = None

    #: Center and diagonal field of view of the reference, in degrees. Under
    #: the default ``max_photref_separation`` of 0.2, images within 1 degree
    #: of the center are close enough to bind to it.
    _photref_center = (150.0, 20.0)
    _photref_fov = 5.0

    @classmethod
    def setUpClass(cls):
        # Closed in tearDownClass rather than by a context manager, which a
        # fixture spanning every test of the class cannot use.
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        set_project_home(cls._tmp.name)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True),
            overwrite_default_config=(
                None
                if cls._max_photref_separation is None
                else {
                    "max-photref-separation": [
                        (None, str(cls._max_photref_separation))
                    ]
                }
            ),
        )
        DataReductionFile.get_file_structure()
        cls._fill_database()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        # A manager without a pipeline run switches logging off entirely.
        self._logging_disabled = logging.root.manager.disable
        self._processing = ImageProcessingManager(pipeline_run_id=None)

    def tearDown(self):
        logging.disable(self._logging_disabled)
        with start_db_session() as db_session:
            db_session.execute(delete(ImageMasterSelection))

    @classmethod
    def _raw_fname(cls, name):
        """Return the raw frame filename of the image with the given name."""

        return path.join(cls._tmp.name, "RAW", f"{name}.fits")

    @classmethod
    def _add_raw_image(cls, db_session, name, session_id, exposure):
        """Write a raw frame, add its image to the database, return the id."""

        header = fits.Header()
        header["DATE-OBS"] = "2023-03-01"
        header["TIME-OBS"] = "21:00:00"
        header["EXPTIME"] = exposure
        fits.PrimaryHDU(
            data=numpy.zeros((4, 4), dtype=numpy.float32), header=header
        ).writeto(cls._raw_fname(name))

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        image = Image(
            raw_fname=cls._raw_fname(name),
            image_type_id=db_session.scalar(
                select(ImageType.id).filter_by(name="object")
            ),
            observing_session_id=session_id,
            jd=2460005.375,
        )
        # pylint: enable=not-callable
        db_session.add(image)
        db_session.flush()
        return image.id

    @staticmethod
    def _add_diagnostics(db_session, image_id, channel, values):
        """Record the given ``{name: value}`` astrometry diagnostics."""

        for name, value in values.items():
            db_session.add(
                # pylint: disable=not-callable
                ImageDiagnostics(
                    image_id=image_id,
                    channel=channel,
                    diagnostic_id=db_session.scalar(
                        select(DiagnosticType.id).filter_by(name=name)
                    ),
                    value=value,
                )
                # pylint: enable=not-callable
            )

    @classmethod
    def _add_session(cls, db_session):
        """Add the equipment and the observing session, return its id."""

        # pylint: disable=not-callable
        camera_type = CameraType(
            make="Test",
            model="2-channel",
            version="1",
            x_resolution=4,
            y_resolution=4,
            pixel_size=1.0,
            notes="",
        )
        telescope_type = TelescopeType(
            make="Test", model="lens", version="1", f_ratio=2.0, focal_length=50
        )
        observer = Observer(name="observer")
        target = Target(name="field")
        db_session.add_all([camera_type, telescope_type, observer, target])
        db_session.flush()
        for offset, name in enumerate(("R", "G")):
            db_session.add(
                CameraChannel(
                    camera_type_id=camera_type.id,
                    name=name,
                    x_offset=offset,
                    y_offset=0,
                    x_step=2,
                    y_step=1,
                )
            )
        camera = Camera(
            camera_type_id=camera_type.id, serial_number="camera", notes=""
        )
        telescope = Telescope(
            telescope_type_id=telescope_type.id,
            serial_number="telescope",
            notes="",
        )
        db_session.add_all([camera, telescope])
        db_session.flush()

        # Mount and observatory are never read, and SQLite does not enforce
        # the foreign keys left dangling.
        session = ObservingSession(
            observer_id=observer.id,
            camera_id=camera.id,
            telescope_id=telescope.id,
            mount_id=1,
            observatory_id=1,
            target_id=target.id,
            label="night",
            start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
            end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
        )
        # pylint: enable=not-callable
        db_session.add(session)
        db_session.flush()
        return session.id

    @classmethod
    def _add_photref(cls, db_session, session_id):
        """Register the single photref: a DR file of image ``ref``, in R."""

        image_id = cls._add_raw_image(db_session, "ref", session_id, 30.0)
        cls._add_diagnostics(
            db_session,
            image_id,
            "R",
            {
                "ra_center": cls._photref_center[0],
                "dec_center": cls._photref_center[1],
                "diagonal_fov": cls._photref_fov,
            },
        )

        cls._photref_fname = path.join(cls._tmp.name, "DR", "ref_R.h5")
        with DataReductionFile(cls._photref_fname, "w") as dr_file:
            dr_file.initialize(
                fits.Header(
                    {
                        "RAWFNAME": "ref",
                        "CLRCHNL": "R",
                        "TARGETID": "field",
                        "EXPTIME": 30.0,
                    }
                )
            )

        # pylint: disable=not-callable
        photref = MasterFile(
            type_id=db_session.scalar(
                select(MasterType.id).filter_by(name="single_photref")
            ),
            filename=cls._photref_fname,
            enabled=True,
        )
        # pylint: enable=not-callable
        db_session.add(photref)
        db_session.flush()
        cls._photref_id = photref.id

    @classmethod
    def _fill_database(cls):
        """Add the session, the reference and the pending images."""

        # ``{name: (ra_center, dec_center, EXPTIME)}`` of the images pending
        # magnitude fitting. The single_photref must match ``TARGETID``,
        # ``CLRCHNL`` and ``EXPTIME``, so the reference, exposed for 30 s,
        # serves only 30 s images, in ``R``.
        pending = {
            "near": (150.3, 20.0, 30.0),
            "far": (155.0, 20.0, 30.0),
            "longer": (150.3, 20.0, 60.0),
        }

        makedirs(path.join(cls._tmp.name, "RAW"))
        makedirs(path.join(cls._tmp.name, "DR"))
        with start_db_session() as db_session:
            session_id = cls._add_session(db_session)
            cls._add_photref(db_session, session_id)
            cls._image_ids = {}
            for name, (ra_center, dec_center, exposure) in pending.items():
                cls._image_ids[name] = cls._add_raw_image(
                    db_session, name, session_id, exposure
                )
                for channel in ("R", "G"):
                    cls._add_diagnostics(
                        db_session,
                        cls._image_ids[name],
                        channel,
                        {"ra_center": ra_center, "dec_center": dec_center},
                    )

    def bind_pending(self, db_session):
        """Bind every pending image in both channels, as magfit preparation.

        Returns:
            [(str, str)]:
                The ``(image name, channel)`` entries handed on to
                ``fit_magnitudes``, sorted.
        """

        name_of = {image_id: name for name, image_id in self._image_ids.items()}
        pending = []
        for image_id in self._image_ids.values():
            image = db_session.get(Image, image_id)
            self._processing.evaluate_expressions_image(image, db_session)
            pending.extend((image, channel, None) for channel in ("R", "G"))

        # pylint: disable=protected-access
        returned = self._processing._bind_photref_for_pending(
            pending,
            db_session.scalar(select(Step).filter_by(name="fit_magnitudes")),
            db_session,
        )
        # pylint: enable=protected-access
        return sorted(
            (name_of[image.id], channel) for image, channel, _ in returned
        )

    def bindings(self, db_session):
        """Return ``{(image name, channel): MasterFile ID}`` of every binding.

        The reference's own image is never pending, so is never bound.
        """

        name_of = {image_id: name for name, image_id in self._image_ids.items()}
        return {
            (name_of[image_id], channel): master_file_id
            for image_id, channel, master_file_id in db_session.execute(
                select(
                    ImageMasterSelection.image_id,
                    ImageMasterSelection.channel,
                    ImageMasterSelection.master_file_id,
                )
            ).all()
        }

    def all_pending(self):
        """Return every ``(image name, channel)`` pending, sorted."""

        return sorted(
            (name, channel)
            for name in self._image_ids
            for channel in ("R", "G")
        )

    def bind_in_advance(self, name, channel):
        """Bind the named image/channel to the reference before magfit."""

        with start_db_session() as db_session:
            db_session.add(
                # pylint: disable=not-callable
                ImageMasterSelection(
                    image_id=self._image_ids[name],
                    channel=channel,
                    master_type_id=db_session.scalar(
                        select(MasterType.id).filter_by(name="single_photref")
                    ),
                    master_file_id=self._photref_id,
                )
                # pylint: enable=not-callable
            )


class TestSeparationBinding(PhotrefBindingProject):
    """A finite ``max_photref_separation``: bound by distance to the photref."""

    def test_only_images_near_the_photref_are_bound(self):
        """Only the near image is fit against the photref.

        The far one is out of range, the 60 s one does not match the
        photref's conditions, and channel ``G`` has no photref at all.
        """

        with start_db_session() as db_session:
            returned = self.bind_pending(db_session)
            bindings = self.bindings(db_session)

        self.assertEqual(returned, [("near", "R")])
        self.assertEqual(bindings, {("near", "R"): self._photref_id})

    def test_an_image_bound_in_advance_is_kept(self):
        """A binding made in the BUI stands, however far the image is."""

        self.bind_in_advance("far", "R")
        with start_db_session() as db_session:
            returned = self.bind_pending(db_session)
            bindings = self.bindings(db_session)

        self.assertEqual(returned, [("far", "R"), ("near", "R")])
        self.assertEqual(
            bindings,
            {
                ("far", "R"): self._photref_id,
                ("near", "R"): self._photref_id,
            },
        )


class TestConditionBinding(PhotrefBindingProject):
    """An unlimited ``max_photref_separation``: the conditions choose."""

    _max_photref_separation = float("inf")

    def test_the_selected_photref_is_recorded(self):
        """Every image is handed on, those the conditions select are bound.

        Distance no longer matters, so the far image is bound too. Channel
        ``G`` and the 60 s image have no photref, so are not bound, but are
        handed on all the same: dropping them is left to the batch
        configuration, as for any missing master.
        """

        with start_db_session() as db_session:
            returned = self.bind_pending(db_session)
            bindings = self.bindings(db_session)

        self.assertEqual(returned, self.all_pending())
        self.assertEqual(
            bindings,
            {
                ("far", "R"): self._photref_id,
                ("near", "R"): self._photref_id,
            },
        )

    def test_the_binding_is_what_magfit_is_given(self):
        """Each row names the master its batch is configured with."""

        with start_db_session() as db_session:
            returned = self.bind_pending(db_session)
            magfit_given = {
                (name, channel): self._processing.get_master_fname(
                    self._image_ids[name], channel, "single_photref"
                )
                for name, channel in returned
            }
            bindings = self.bindings(db_session)

        self.assertEqual(
            set(magfit_given.values()), {None, self._photref_fname}
        )
        self.assertEqual(
            bindings,
            {
                entry: self._photref_id
                for entry, photref_fname in magfit_given.items()
                if photref_fname is not None
            },
        )

    def test_an_image_bound_in_advance_is_kept(self):
        """A binding the conditions would not make stands, and is used."""

        self.bind_in_advance("longer", "R")
        with start_db_session() as db_session:
            returned = self.bind_pending(db_session)
            bindings = self.bindings(db_session)
            magfit_given = self._processing.get_master_fname(
                self._image_ids["longer"], "R", "single_photref"
            )

        self.assertEqual(returned, self.all_pending())
        self.assertEqual(bindings[("longer", "R")], self._photref_id)
        self.assertEqual(magfit_given, self._photref_fname)


if __name__ == "__main__":
    unittest.main()
