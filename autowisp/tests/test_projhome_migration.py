"""Tests of revision 0018, which stores a project's paths as ``{PROJHOME}/...``.

The upgrade tests run the revision over projects created by every release,
covering what it does to the database. Those projects have no products, so
these cover the files: each kind on its own, and a project holding every
kind, migrated whole and after an interruption.
"""

import os
import tempfile
import unittest
import warnings
from argparse import Namespace
from datetime import datetime
from functools import cache
from pathlib import Path

from alembic.script import ScriptDirectory
from astropy.io import fits
from astropy.table import Table
import h5py
import numpy
from sqlalchemy import select

from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import (
    get_db_engine,
    set_project_home,
    start_db_session,
)
from autowisp.database.migrate import (
    _alembic_config,
    _stamp,
    get_head_revision,
    get_project_revision,
    migrate_project,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    Image,
    ImageType,
    MasterFile,
    MasterType,
    ObservingSession,
    Parameter,
    ProcessedImages,
    Target,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraType,
    Mount,
    MountType,
    Observatory,
    Observer,
    Telescope,
    TelescopeType,
)

# pylint: enable=no-name-in-module
from autowisp.tests import SERVER_URL_ENV, empty_server_database
from autowisp.tests.test_database_migration import on_server

# Resources here are released via addCleanup rather than `with`, since they
# are created in setUp and must outlive it.
# pylint: disable=consider-using-with

_REVISION = "0018_projhome_paths"
_PREVIOUS_REVISION = "0017_master_mask_description"
_SPHOTREF_NAME = "SinglePhotometricReference"


@cache
def _load_revision():
    """Return the module of the revision under test."""

    return (
        ScriptDirectory.from_config(_alembic_config())
        .get_revision(_REVISION)
        .module
    )


def _get_path_cases(project_home):
    """
    Return paths a project on this system may record, and what each becomes.

    The project home must hold ``sub/f.fits``. Windows adds the other ways it
    accepts to write a path, and paths only it can have.
    """

    converted = "{PROJHOME}/sub/f.fits"
    outside = os.path.join(os.path.dirname(project_home), "elsewhere.fits")
    cases = {
        os.path.join(project_home, "sub", "f.fits"): converted,
        "sub/f.fits": converted,
        # A relative path is inside only if it names a file there.
        "sub/missing.fits": "sub/missing.fits",
        outside: outside,
        converted: converted,
    }
    if os.name == "nt":
        drive, rooted_home = os.path.splitdrive(project_home)
        other_drive = "Z:" if drive.upper() != "Z:" else "Y:"
        on_other_drive = other_drive + rooted_home + "\\sub\\f.fits"
        cases.update(
            {
                project_home.replace("\\", "/") + "/sub/f.fits": converted,
                project_home + "/sub\\f.fits": converted,
                # Windows compares paths regardless of case.
                project_home.upper() + "\\sub\\f.fits": converted,
                "sub\\f.fits": converted,
                # Rooted without a drive: on the drive of the home.
                rooted_home + "\\sub\\f.fits": converted,
                "\\elsewhere\\f.fits": "\\elsewhere\\f.fits",
                on_other_drive: on_other_drive,
            }
        )
    return cases


def _touch(fname):
    """Create an empty file, and the directories it is in."""

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb"):
        pass


def _write_fits(fname, cards, *, compressed=True, checksum=False):
    """
    Write an image whose header holds the given cards.

    Compressed, the image is the second HDU, after an empty primary one, as
    calibrated images are; otherwise it is the primary HDU itself.
    """

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    image = numpy.random.default_rng(0).normal(size=(32, 32)).astype("float32")
    header = fits.Header(cards)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.CompImageHDU(image, header)]
        if compressed
        else [fits.PrimaryHDU(image, header)]
    ).writeto(fname, checksum=checksum, overwrite=True)


def _get_hdu_bytes(fname):
    """Return the header and the data bytes of each HDU of a FITS file."""

    with fits.open(fname) as hdu_list:
        layout = [hdu_list.fileinfo(index) for index in range(len(hdu_list))]
    with open(fname, "rb") as fits_file:
        content = fits_file.read()
    return [
        (
            content[info["hdrLoc"] : info["datLoc"]],
            content[info["datLoc"] : info["datLoc"] + info["datSpan"]],
        )
        for info in layout
    ]


def _parse_header(header_bytes):
    """Return the header the given bytes of a FITS file hold."""

    return fits.Header.fromstring(header_bytes.decode("ascii"))


def _count_cards(header_bytes):
    """Return the number of cards a header uses, its END card included."""

    count = 0
    for start in range(0, len(header_bytes), 80):
        count += 1
        if header_bytes[start : start + 80].startswith(b"END     "):
            return count
    raise ValueError("No END card in the header.")


def _write_dr(fname, sphotref, header_cards):
    """Write a DR file recording a photometric reference and a header."""

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with h5py.File(fname, "w") as dr_file:
        dr_file.attrs[_SPHOTREF_NAME] = sphotref
        dr_file.create_dataset(
            "FITSHeader",
            data=numpy.frombuffer(
                b"".join(
                    card.image.encode("ascii")
                    for card in fits.Header(header_cards).cards
                ),
                dtype="i1",
            ),
        )


def _read_dr_header(dr_file):
    """Return the copy of the frame header an open DR file holds."""

    return fits.Header.fromstring(
        dr_file["FITSHeader"][()].tobytes().decode("ascii")
    )


class _ProjectHomeMixin:
    """Give each test a project home holding ``sub/f.fits``."""

    # Mixed into unittest.TestCase subclasses, so setUp is an override
    # despite this class not deriving from TestCase itself.
    # pylint: disable=invalid-name
    def setUp(self):
        """Create the project home."""

        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._project_home = tmp.name
        _touch(self._get_abs_path("sub", "f.fits"))

    # pylint: enable=invalid-name

    def _get_abs_path(self, *parts):
        """Return the absolute path of a file in the project."""

        return os.path.join(self._project_home, *parts)

    def _assert_unchanged(self, fname, migrate):
        """
        Check that converting a file with nothing to convert does not change it.

        Not that it is not written: opening an HDF5 file for writing updates
        its modification time, even if nothing is written.
        """

        with open(fname, "rb") as original:
            content = original.read()

        self.assertFalse(migrate(fname, self._project_home))

        with open(fname, "rb") as migrated:
            self.assertEqual(migrated.read(), content)


class TestProjhomeFitsMigration(_ProjectHomeMixin, unittest.TestCase):
    """The headers of FITS files are converted, and the data copied as is."""

    def test_converts_the_file_keywords_in_place(self):
        """Each path of a file keyword is converted as the cases say."""

        path_cases = _get_path_cases(self._project_home)
        inside = self._get_abs_path("sub", "f.fits")
        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                image_hdu = 1 if compressed else 0
                fname = self._get_abs_path(f"image{image_hdu}.fits")
                _write_fits(
                    fname,
                    [
                        (f"ORIGF{index:03d}", recorded)
                        for index, recorded in enumerate(path_cases)
                    ]
                    + [
                        ("MBIASFNM", inside),
                        ("PROJHOME", self._project_home),
                        # Not a keyword naming a file.
                        ("OBJECT", inside),
                    ],
                    compressed=compressed,
                    checksum=True,
                )
                original = _get_hdu_bytes(fname)
                original_stat = os.stat(fname)
                with fits.open(fname) as hdu_list:
                    original_image = hdu_list[image_hdu].data.copy()

                self.assertTrue(
                    _load_revision()._migrate_fits(fname, self._project_home)
                )

                migrated = _get_hdu_bytes(fname)
                migrated_stat = os.stat(fname)
                # No value outgrew its card, so the header kept its blocks
                # and the file was written in place.
                self.assertEqual(
                    (migrated_stat.st_size, migrated_stat.st_ino),
                    (original_stat.st_size, original_stat.st_ino),
                )
                self.assertEqual(
                    [data for _, data in migrated],
                    [data for _, data in original],
                )
                if compressed:
                    # Nothing to convert there, so its CHECKSUM stays.
                    self.assertEqual(migrated[0][0], original[0][0])

                header = _parse_header(migrated[image_hdu][0])
                for index, (recorded, expected) in enumerate(
                    path_cases.items()
                ):
                    with self.subTest(recorded=recorded):
                        self.assertEqual(header[f"ORIGF{index:03d}"], expected)
                self.assertEqual(header["MBIASFNM"], "{PROJHOME}/sub/f.fits")
                self.assertEqual(header["OBJECT"], inside)
                self.assertNotIn("PROJHOME", header)
                self.assertNotIn("CHECKSUM", header)
                self.assertEqual(
                    header["DATASUM"],
                    _parse_header(original[image_hdu][0])["DATASUM"],
                )

                # A checksum that no longer matches is reported as a
                # warning.
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with fits.open(fname, checksum=True) as hdu_list:
                        numpy.testing.assert_array_equal(
                            hdu_list[image_hdu].data, original_image
                        )

    def test_moves_the_data_when_a_header_outgrows_its_blocks(self):
        """A header needing another block is rewritten, data and all."""

        # Converted, a relative path gets longer, and past 68 characters a
        # value takes a second card.
        recorded = "sub/" + "f" * 55 + ".fits"
        _touch(self._get_abs_path(recorded))
        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                image_hdu = 1 if compressed else 0
                fname = self._get_abs_path(f"image{image_hdu}.fits")
                _write_fits(
                    fname, [("ORIGF001", recorded)], compressed=compressed
                )
                cards_used = _count_cards(_get_hdu_bytes(fname)[image_hdu][0])
                # Fill the last block of the header, so that one more card
                # needs another.
                _write_fits(
                    fname,
                    [("ORIGF001", recorded)]
                    + [("HISTORY", "padding")] * ((-cards_used) % 36),
                    compressed=compressed,
                )
                original = _get_hdu_bytes(fname)
                self.assertEqual(_count_cards(original[image_hdu][0]) % 36, 0)
                with fits.open(fname) as hdu_list:
                    original_image = hdu_list[image_hdu].data.copy()

                self.assertTrue(
                    _load_revision()._migrate_fits(fname, self._project_home)
                )

                migrated = _get_hdu_bytes(fname)
                self.assertEqual(
                    len(migrated[image_hdu][0]),
                    len(original[image_hdu][0]) + 2880,
                )
                self.assertEqual(
                    [data for _, data in migrated],
                    [data for _, data in original],
                )
                self.assertEqual(
                    _parse_header(migrated[image_hdu][0])["ORIGF001"],
                    "{PROJHOME}/" + recorded,
                )
                with fits.open(fname) as hdu_list:
                    numpy.testing.assert_array_equal(
                        hdu_list[image_hdu].data, original_image
                    )

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted, or never holding paths, is not written."""

        fname = self._get_abs_path("image.fits.fz")
        _write_fits(
            fname,
            [
                ("ORIGF001", "{PROJHOME}/sub/f.fits"),
                ("OBJECT", self._get_abs_path("sub", "f.fits")),
            ],
            checksum=True,
        )
        self._assert_unchanged(fname, _load_revision()._migrate_fits)


class TestProjhomeDRMigration(_ProjectHomeMixin, unittest.TestCase):
    """DR files have their photometric references and header converted."""

    def test_converts_the_references_and_the_header(self):
        """Every reference attribute, wherever it is, and the header."""

        path_cases = _get_path_cases(self._project_home)
        inside = self._get_abs_path("sub", "f.fits")
        fname = self._get_abs_path("frame.h5")
        _write_dr(
            fname,
            inside,
            [
                (f"ORIGF{index:03d}", recorded)
                for index, recorded in enumerate(path_cases)
            ]
            + [
                ("MBIASFNM", inside),
                ("PROJHOME", self._project_home),
                ("OBJECT", inside),
            ],
        )
        with h5py.File(fname, "a") as dr_file:
            for index, recorded in enumerate(path_cases):
                dr_file.create_group(f"group{index:02d}").attrs[
                    _SPHOTREF_NAME
                ] = recorded
            dr_file.create_dataset("group00/data", data=[1, 2]).attrs[
                _SPHOTREF_NAME
            ] = inside
            dr_file.create_group("other").attrs["OtherPath"] = inside

        self.assertTrue(_load_revision()._migrate_dr(fname, self._project_home))

        with h5py.File(fname, "r") as dr_file:
            self.assertEqual(
                dr_file.attrs[_SPHOTREF_NAME], "{PROJHOME}/sub/f.fits"
            )
            for index, (recorded, expected) in enumerate(path_cases.items()):
                with self.subTest(recorded=recorded):
                    self.assertEqual(
                        dr_file[f"group{index:02d}"].attrs[_SPHOTREF_NAME],
                        expected,
                    )
            self.assertEqual(
                dr_file["group00/data"].attrs[_SPHOTREF_NAME],
                "{PROJHOME}/sub/f.fits",
            )
            self.assertEqual(dr_file["other"].attrs["OtherPath"], inside)

            header = _read_dr_header(dr_file)
        for index, (recorded, expected) in enumerate(path_cases.items()):
            with self.subTest(header_recorded=recorded):
                self.assertEqual(header[f"ORIGF{index:03d}"], expected)
        self.assertEqual(header["MBIASFNM"], "{PROJHOME}/sub/f.fits")
        self.assertEqual(header["OBJECT"], inside)
        self.assertNotIn("PROJHOME", header)

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted, or never holding paths, is not written."""

        fname = self._get_abs_path("frame.h5")
        _write_dr(
            fname,
            "{PROJHOME}/sub/f.fits",
            [("OBJECT", self._get_abs_path("sub", "f.fits"))],
        )
        self._assert_unchanged(fname, _load_revision()._migrate_dr)


class TestProjhomeLCMigration(_ProjectHomeMixin, unittest.TestCase):
    """Lightcurves have their photometric references converted."""

    def test_converts_the_references(self):
        """Fixed and variable length datasets, widened where needed."""

        path_cases = _get_path_cases(self._project_home)
        recorded = [fname.encode("ascii") for fname in path_cases]
        fname = self._get_abs_path("lc.h5")
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                f"fixed/{_SPHOTREF_NAME}", data=numpy.array(recorded)
            )
            light_curve.create_dataset(
                f"variable/{_SPHOTREF_NAME}",
                data=recorded,
                dtype=h5py.string_dtype(),
            )
            # Converted, the relative path no longer fits.
            short = light_curve.create_dataset(
                f"short/{_SPHOTREF_NAME}",
                data=numpy.array([b"sub/f.fits"]),
                compression="gzip",
            )
            short.attrs["kept"] = 1
            light_curve.create_dataset(
                "other/OtherPath", data=numpy.array(recorded)
            )

        self.assertTrue(_load_revision()._migrate_lc(fname, self._project_home))

        with h5py.File(fname, "r") as light_curve:
            for name in ("fixed", "variable"):
                with self.subTest(dataset=name):
                    self.assertEqual(
                        [
                            value.decode("ascii")
                            for value in light_curve[
                                f"{name}/{_SPHOTREF_NAME}"
                            ][()]
                        ],
                        list(path_cases.values()),
                    )
            short = light_curve[f"short/{_SPHOTREF_NAME}"]
            self.assertEqual(short[()].tolist(), [b"{PROJHOME}/sub/f.fits"])
            self.assertEqual(short.compression, "gzip")
            self.assertEqual(short.attrs["kept"], 1)
            self.assertEqual(
                light_curve["other/OtherPath"][()].tolist(), recorded
            )

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted is not written."""

        fname = self._get_abs_path("lc.h5")
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                f"fixed/{_SPHOTREF_NAME}",
                data=numpy.array([b"{PROJHOME}/sub/f.fits"] * 2),
            )
        self._assert_unchanged(fname, _load_revision()._migrate_lc)


class TestProjhomeProjectMigration(_ProjectHomeMixin, unittest.TestCase):
    """
    A project holding every kind of product is migrated whole.

    The project has frames ``frame1`` (inside the project, processed in
    channels ``R`` and ``G``), ``frame2`` (outside, in ``B``) and ``frame3``
    (inside, in ``R``, but missing). ``calibrated-fname`` takes two values,
    the second, under another condition, as an absolute path.
    """

    # pylint: disable=invalid-name
    def setUp(self):
        """Create the project at the previous revision, with its products."""

        super().setUp()
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        self._outside_raw = os.path.join(outside.name, "frame2.fits")
        self._create_project()
        self._write_products()
        self._add_rows()

    # pylint: enable=invalid-name

    def _create_project(self):
        """Create the project database, at the revision before 0018."""

        if on_server():
            empty_server_database(os.environ[SERVER_URL_ENV])
            set_project_home(
                self._project_home, db_url=os.environ[SERVER_URL_ENV]
            )
        else:
            set_project_home(self._project_home)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True),
            overwrite_default_config={
                "calibrated-fname": [
                    (None, "{PROJHOME}/CAL/{RAWFNAME}_{CLRCHNL}.fits.fz"),
                    (
                        ["EXPTIME > 100"],
                        self._get_abs_path("CAL2", "{RAWFNAME}_{CLRCHNL}.fits"),
                    ),
                ]
            },
        )
        with get_db_engine().begin() as connection:
            _stamp(connection, _PREVIOUS_REVISION)

    def _write_products(self):
        """Write the raw frames and every product, recording the paths."""

        raw = {
            "frame1": self._get_abs_path("RAW", "frame1.fits"),
            "frame2": self._outside_raw,
        }
        for raw_fname in raw.values():
            os.makedirs(os.path.dirname(raw_fname), exist_ok=True)
            fits.PrimaryHDU(numpy.zeros((4, 4), dtype="float32")).writeto(
                raw_fname
            )
        self._expected_raw = {
            "frame1": "{PROJHOME}/RAW/frame1.fits",
            "frame2": self._outside_raw,
        }
        bias = self._get_abs_path("MASTERS", "bias.fits")
        self._sphotref = self._get_abs_path("DR", "frame1_R.h5")
        common_cards = [
            ("MBIASFNM", bias),
            ("PROJHOME", self._project_home),
        ]

        # The calibrated images, with the raw frame each was made from.
        self._calibrated = {
            self._get_abs_path("CAL", "frame1_R.fits.fz"): "frame1",
            self._get_abs_path("CAL", "frame1_G.fits.fz"): "frame1",
            self._get_abs_path("CAL2", "frame2_B.fits"): "frame2",
        }
        for fname, frame in self._calibrated.items():
            _write_fits(fname, [("ORIGF001", raw[frame])] + common_cards)
        # Named by a template, but for a channel never processed.
        self._unprocessed = self._get_abs_path("CAL", "frame1_B.fits.fz")
        _write_fits(self._unprocessed, [("ORIGF001", raw["frame1"])])

        self._dr = {
            self._sphotref: "frame1",
            self._get_abs_path("DR", "frame1_G.h5"): "frame1",
            self._get_abs_path("DR", "frame2_B.h5"): "frame2",
        }
        for fname, frame in self._dr.items():
            _write_dr(
                fname, self._sphotref, [("ORIGF001", raw[frame])] + common_cards
            )

        _write_fits(
            bias,
            [
                ("ORIGF001", raw["frame1"]),
                ("PROJHOME", self._project_home),
            ],
        )
        self._lc_catalog = self._get_abs_path("MASTERS", "lc_catalog.fits")
        Table(
            {"source_id": numpy.array([1, 2], dtype="uint64")},
            meta={"PROJHOME": self._project_home},
        ).write(self._lc_catalog)
        self._masters = {"zero": bias, "lightcurve_catalog": self._lc_catalog}

        self._lightcurves = [
            self._get_abs_path("LC", f"GDR3_{source_id}.h5")
            for source_id in (1, 2)
        ]
        for fname in self._lightcurves:
            self._write_lightcurve(fname)

    def _write_lightcurve(self, fname):
        """Write a lightcurve recording the single photometric reference."""

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                f"AperturePhotometry/{_SPHOTREF_NAME}",
                data=numpy.array([self._sphotref.encode("ascii")] * 3),
            )

    def _add_rows(self):
        """Add the observing session, its images and the masters."""

        with start_db_session() as db_session:
            # False positive: the declarative models are callable.
            # pylint: disable=not-callable
            camera_type = CameraType(
                make="Test",
                model="camera",
                version="1",
                x_resolution=4,
                y_resolution=4,
                pixel_size=1.0,
            )
            telescope_type = TelescopeType(
                make="Test",
                model="lens",
                version="1",
                f_ratio=2.0,
                focal_length=50,
            )
            mount_type = MountType(make="Test", model="mount", version="1")
            observer = Observer(name="observer")
            observatory = Observatory(
                name="observatory", latitude=0.0, longitude=0.0, altitude=0.0
            )
            target = Target(name="field")
            db_session.add_all(
                [
                    camera_type,
                    telescope_type,
                    mount_type,
                    observer,
                    observatory,
                    target,
                ]
            )
            db_session.flush()
            camera = Camera(
                camera_type_id=camera_type.id, serial_number="camera", notes=""
            )
            telescope = Telescope(
                telescope_type_id=telescope_type.id,
                serial_number="telescope",
                notes="",
            )
            mount = Mount(
                mount_type_id=mount_type.id, serial_number="mount", notes=""
            )
            db_session.add_all([camera, telescope, mount])
            db_session.flush()
            session = ObservingSession(
                observer_id=observer.id,
                camera_id=camera.id,
                telescope_id=telescope.id,
                mount_id=mount.id,
                observatory_id=observatory.id,
                target_id=target.id,
                label="night",
                start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
                end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
            )
            db_session.add(session)
            db_session.flush()

            object_type_id = db_session.scalar(
                select(ImageType.id).filter_by(name="object")
            )
            for raw_fname, channels in (
                (self._get_abs_path("RAW", "frame1.fits"), "RG"),
                (self._outside_raw, "B"),
                (self._get_abs_path("RAW", "frame3.fits"), "R"),
            ):
                image = Image(
                    raw_fname=raw_fname,
                    image_type_id=object_type_id,
                    observing_session_id=session.id,
                )
                db_session.add(image)
                db_session.flush()
                for channel in channels:
                    db_session.add(
                        ProcessedImages(
                            image_id=image.id,
                            channel=channel,
                            status=1,
                            final=True,
                        )
                    )
            for type_name, fname in (
                *self._masters.items(),
                ("single_photref", self._sphotref),
            ):
                db_session.add(
                    MasterFile(
                        type_id=db_session.scalar(
                            select(MasterType.id).filter_by(name=type_name)
                        ),
                        filename=fname,
                        enabled=True,
                    )
                )
            # pylint: enable=not-callable

    def _migrate(self):
        """Migrate the project to the current head."""

        return migrate_project(
            get_db_engine(), self._project_home, assume_backed_up=on_server()
        )

    def _assert_project_converted(self):
        """Check the database and every product hold only stored paths."""

        self.assertEqual(
            get_project_revision(get_db_engine()), get_head_revision()
        )
        with start_db_session() as db_session:
            self.assertEqual(
                set(db_session.scalars(select(MasterFile.filename))),
                {
                    "{PROJHOME}/MASTERS/bias.fits",
                    "{PROJHOME}/MASTERS/lc_catalog.fits",
                    "{PROJHOME}/DR/frame1_R.h5",
                },
            )
            self.assertEqual(
                set(db_session.scalars(select(Image.raw_fname))),
                {
                    "{PROJHOME}/RAW/frame1.fits",
                    self._outside_raw,
                    # Missing, but inside the project.
                    "{PROJHOME}/RAW/frame3.fits",
                },
            )
            self.assertEqual(
                set(
                    db_session.scalars(
                        select(Configuration.value)
                        .join(
                            Parameter,
                            Configuration.parameter_id == Parameter.id,
                        )
                        .where(Parameter.name == "calibrated-fname")
                    )
                ),
                {
                    "{PROJHOME}/CAL/{RAWFNAME}_{CLRCHNL}.fits.fz",
                    "{PROJHOME}/CAL2/{RAWFNAME}_{CLRCHNL}.fits",
                },
            )

        for fname, frame in self._calibrated.items():
            with self.subTest(calibrated=fname):
                header = fits.getheader(fname, 1)
                self.assertEqual(header["ORIGF001"], self._expected_raw[frame])
                self.assertEqual(
                    header["MBIASFNM"], "{PROJHOME}/MASTERS/bias.fits"
                )
                self.assertNotIn("PROJHOME", header)
        self.assertEqual(
            fits.getheader(self._unprocessed, 1)["ORIGF001"],
            self._get_abs_path("RAW", "frame1.fits"),
        )

        for fname, frame in self._dr.items():
            with self.subTest(dr=fname), h5py.File(fname, "r") as dr_file:
                self.assertEqual(
                    dr_file.attrs[_SPHOTREF_NAME], "{PROJHOME}/DR/frame1_R.h5"
                )
                header = _read_dr_header(dr_file)
                self.assertEqual(header["ORIGF001"], self._expected_raw[frame])
                self.assertNotIn("PROJHOME", header)

        bias_header = fits.getheader(self._masters["zero"], 1)
        self.assertEqual(bias_header["ORIGF001"], "{PROJHOME}/RAW/frame1.fits")
        self.assertNotIn("PROJHOME", bias_header)
        self.assertNotIn("PROJHOME", fits.getheader(self._lc_catalog, 1))

        for fname in self._lightcurves:
            with (
                self.subTest(lightcurve=fname),
                h5py.File(fname, "r") as light_curve,
            ):
                self.assertEqual(
                    light_curve[f"AperturePhotometry/{_SPHOTREF_NAME}"][
                        ()
                    ].tolist(),
                    [b"{PROJHOME}/DR/frame1_R.h5"] * 3,
                )

    def test_converts_the_database_and_every_product(self):
        """One migration converts all of it, despite the missing frame."""

        self._migrate()
        self._assert_project_converted()

    def test_resumes_after_an_interruption(self):
        """A failed migration leaves the database; the rerun finishes it."""

        with open(self._lightcurves[1], "wb") as broken:
            broken.write(b"not HDF5")

        with self.assertRaises(OSError):
            self._migrate()

        self.assertEqual(
            get_project_revision(get_db_engine()), _PREVIOUS_REVISION
        )
        with start_db_session() as db_session:
            self.assertIn(
                self._masters["zero"],
                set(db_session.scalars(select(MasterFile.filename))),
            )
        # Converted before the failure, and not converted back.
        self.assertEqual(
            fits.getheader(next(iter(self._calibrated)), 1)["MBIASFNM"],
            "{PROJHOME}/MASTERS/bias.fits",
        )

        self._write_lightcurve(self._lightcurves[1])
        converted = (
            list(self._calibrated)
            + list(self._dr)
            + list(self._masters.values())
            + self._lightcurves[:1]
        )
        contents = {fname: Path(fname).read_bytes() for fname in converted}

        self._migrate()

        self._assert_project_converted()
        self.assertEqual(
            {fname: Path(fname).read_bytes() for fname in converted}, contents
        )
