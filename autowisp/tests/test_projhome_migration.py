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
    Parameter,
    ProcessedImages,
)

# pylint: enable=no-name-in-module
from autowisp.tests import SERVER_URL_ENV, empty_server_database
from autowisp.tests.database_fixtures import add_cameras, add_observing_session
from autowisp.tests.projhome_fixtures import (
    SPHOTREF_NAME,
    ProjectHomeMixin,
    count_cards,
    get_hdu_bytes,
    get_path_cases,
    parse_header,
    read_dr_header,
    touch,
    write_dr,
    write_fits,
    write_master_photref,
)
from autowisp.tests.test_database_migration import on_server

# Resources here are released via addCleanup rather than `with`, since they
# are created in setUp and must outlive it.
# pylint: disable=consider-using-with

# The revision's converters are private to it, and tested here directly.
# pylint: disable=protected-access

# pylint: disable=invalid-name
_REVISION = "0018_projhome_paths"
_PREVIOUS_REVISION = "0017_master_mask_description"
# pylint: enable=invalid-name


@cache
def _load_revision():
    """Return the module of the revision under test."""

    return (
        ScriptDirectory.from_config(_alembic_config())
        .get_revision(_REVISION)
        .module
    )


class TestProjhomeFitsMigration(ProjectHomeMixin, unittest.TestCase):
    """The headers of FITS files are converted, and the data copied as is."""

    def test_converts_the_file_keywords_in_place(self):
        """Each path of a file keyword is converted as the cases say."""

        path_cases = get_path_cases(self._project_home)
        inside = self._get_abs_path("sub", "f.fits")
        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                image_hdu = 1 if compressed else 0
                fname = self._get_abs_path(f"image{image_hdu}.fits")
                write_fits(
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
                original = get_hdu_bytes(fname)
                original_stat = os.stat(fname)
                original_image = fits.getdata(fname, image_hdu)

                self.assertTrue(
                    _load_revision()._migrate_fits(fname, self._project_home)
                )

                migrated = get_hdu_bytes(fname)
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

                header = parse_header(migrated[image_hdu][0])
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
                    parse_header(original[image_hdu][0])["DATASUM"],
                )

                # A checksum that no longer matches is reported as a
                # warning.
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    numpy.testing.assert_array_equal(
                        fits.getdata(fname, image_hdu, checksum=True),
                        original_image,
                    )

    def test_moves_the_data_when_a_header_outgrows_its_blocks(self):
        """A header needing another block is rewritten, data and all."""

        # Converted, a relative path gets longer, and past 68 characters a
        # value takes a second card.
        recorded = "sub/" + "f" * 55 + ".fits"
        touch(self._get_abs_path(recorded))
        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                image_hdu = 1 if compressed else 0
                fname = self._get_abs_path(f"image{image_hdu}.fits")
                write_fits(
                    fname, [("ORIGF001", recorded)], compressed=compressed
                )
                cards_used = count_cards(get_hdu_bytes(fname)[image_hdu][0])
                # Fill the last block of the header, so that one more card
                # needs another.
                write_fits(
                    fname,
                    [("ORIGF001", recorded)]
                    + [("HISTORY", "padding")] * ((-cards_used) % 36),
                    compressed=compressed,
                )
                original = get_hdu_bytes(fname)
                self.assertEqual(count_cards(original[image_hdu][0]) % 36, 0)
                with fits.open(fname) as hdu_list:
                    original_image = hdu_list[image_hdu].data.copy()

                self.assertTrue(
                    _load_revision()._migrate_fits(fname, self._project_home)
                )

                migrated = get_hdu_bytes(fname)
                self.assertEqual(
                    len(migrated[image_hdu][0]),
                    len(original[image_hdu][0]) + 2880,
                )
                self.assertEqual(
                    [data for _, data in migrated],
                    [data for _, data in original],
                )
                self.assertEqual(
                    parse_header(migrated[image_hdu][0])["ORIGF001"],
                    "{PROJHOME}/" + recorded,
                )
                with fits.open(fname) as hdu_list:
                    numpy.testing.assert_array_equal(
                        hdu_list[image_hdu].data, original_image
                    )

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted, or never holding paths, is not written."""

        fname = self._get_abs_path("image.fits.fz")
        write_fits(
            fname,
            [
                ("ORIGF001", "{PROJHOME}/sub/f.fits"),
                ("OBJECT", self._get_abs_path("sub", "f.fits")),
            ],
            checksum=True,
        )
        self._assert_unchanged(fname, _load_revision()._migrate_fits)


class TestProjhomeMasterPhotrefMigration(ProjectHomeMixin, unittest.TestCase):
    """A master photometric reference has its population converted too."""

    def test_converts_the_population_and_the_header(self):
        """Each DR file listed is converted as the cases say, nothing else.

        Every case is listed in the one table, flagged in turn as included
        and not, so that a flag moved to another row shows.
        """

        path_cases = get_path_cases(self._project_home)
        population = {
            recorded: index % 2 == 0
            for index, recorded in enumerate(path_cases)
        }
        fname = self._get_abs_path("MASTERS", "mphotref.fits")
        write_master_photref(
            fname,
            [("MBIASFNM", self._get_abs_path("sub", "f.fits"))],
            population,
        )
        with fits.open(fname) as hdu_list:
            original_values = hdu_list["MPHOTREF"].data.copy()

        self.assertTrue(
            _load_revision()._migrate_master(fname, self._project_home)
        )

        with fits.open(fname) as hdu_list:
            self.assertEqual(
                [hdu.name for hdu in hdu_list],
                ["PRIMARY", "MPHOTREF", "INMASTER"],
            )
            self.assertEqual(
                hdu_list[0].header["MBIASFNM"], "{PROJHOME}/sub/f.fits"
            )
            numpy.testing.assert_array_equal(
                hdu_list["MPHOTREF"].data, original_values
            )
            # Row by row: several cases convert to the same name.
            migrated = hdu_list["INMASTER"].data
            self.assertEqual(
                list(zip(migrated["dr_fname"], migrated["included"].tolist())),
                [
                    (path_cases[recorded], included)
                    for recorded, included in population.items()
                ],
            )

    def test_leaves_a_converted_master_alone(self):
        """A master listing only stored names is not written."""

        fname = self._get_abs_path("MASTERS", "mphotref.fits")
        write_master_photref(
            fname,
            [("MBIASFNM", "{PROJHOME}/sub/f.fits")],
            {"{PROJHOME}/DR/a.h5": True, "{PROJHOME}/DR/b.h5": False},
        )
        self._assert_unchanged(fname, _load_revision()._migrate_master)


class TestProjhomeDRMigration(ProjectHomeMixin, unittest.TestCase):
    """DR files have their photometric references and header converted."""

    def test_converts_the_references_and_the_header(self):
        """Every reference attribute, wherever it is, and the header."""

        path_cases = get_path_cases(self._project_home)
        inside = self._get_abs_path("sub", "f.fits")
        fname = self._get_abs_path("frame.h5")
        write_dr(
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
                    SPHOTREF_NAME
                ] = recorded
            dr_file.create_dataset("group00/data", data=[1, 2]).attrs[
                SPHOTREF_NAME
            ] = inside
            dr_file.create_group("other").attrs["OtherPath"] = inside

        self.assertTrue(_load_revision()._migrate_dr(fname, self._project_home))

        with h5py.File(fname, "r") as dr_file:
            self.assertEqual(
                dr_file.attrs[SPHOTREF_NAME], "{PROJHOME}/sub/f.fits"
            )
            for index, (recorded, expected) in enumerate(path_cases.items()):
                with self.subTest(recorded=recorded):
                    self.assertEqual(
                        dr_file[f"group{index:02d}"].attrs[SPHOTREF_NAME],
                        expected,
                    )
            self.assertEqual(
                dr_file["group00/data"].attrs[SPHOTREF_NAME],
                "{PROJHOME}/sub/f.fits",
            )
            self.assertEqual(dr_file["other"].attrs["OtherPath"], inside)

            header = read_dr_header(dr_file)
        for index, (recorded, expected) in enumerate(path_cases.items()):
            with self.subTest(header_recorded=recorded):
                self.assertEqual(header[f"ORIGF{index:03d}"], expected)
        self.assertEqual(header["MBIASFNM"], "{PROJHOME}/sub/f.fits")
        self.assertEqual(header["OBJECT"], inside)
        self.assertNotIn("PROJHOME", header)

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted, or never holding paths, is not written."""

        fname = self._get_abs_path("frame.h5")
        write_dr(
            fname,
            "{PROJHOME}/sub/f.fits",
            [("OBJECT", self._get_abs_path("sub", "f.fits"))],
        )
        self._assert_unchanged(fname, _load_revision()._migrate_dr)


class TestProjhomeLCMigration(ProjectHomeMixin, unittest.TestCase):
    """Lightcurves have their photometric references converted."""

    def test_converts_the_references(self):
        """Fixed and variable length datasets, widened where needed."""

        path_cases = get_path_cases(self._project_home)
        recorded = [fname.encode("ascii") for fname in path_cases]
        fname = self._get_abs_path("lc.h5")
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                f"fixed/{SPHOTREF_NAME}", data=numpy.array(recorded)
            )
            light_curve.create_dataset(
                f"variable/{SPHOTREF_NAME}",
                data=recorded,
                dtype=h5py.string_dtype(),
            )
            # Converted, the relative path no longer fits.
            short = light_curve.create_dataset(
                f"short/{SPHOTREF_NAME}",
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
                            for value in light_curve[f"{name}/{SPHOTREF_NAME}"][
                                ()
                            ]
                        ],
                        list(path_cases.values()),
                    )
            short = light_curve[f"short/{SPHOTREF_NAME}"]
            self.assertEqual(short[()].tolist(), [b"{PROJHOME}/sub/f.fits"])
            # False positive: pylint takes indexing a file to give a Group.
            # pylint: disable-next=no-member
            self.assertEqual(short.compression, "gzip")
            self.assertEqual(short.attrs["kept"], 1)
            self.assertEqual(
                light_curve["other/OtherPath"][()].tolist(), recorded
            )

    def test_converts_the_reference_in_points_filters(self):
        """The reference the engine adds to the filters, as it writes it."""

        def match(fname):
            """Return the comparison the engine adds for a reference."""

            return f"sphotref == {fname.encode('ascii')!r}"

        inside = self._get_abs_path("DR", "ref.h5")
        outside = os.path.join(os.path.dirname(self._project_home), "ref.h5")
        converted = match("{PROJHOME}/DR/ref.h5")
        fname = self._get_abs_path("lc.h5")
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                "EPD/FitProperties/Filter",
                data=[match(inside), match(outside)],
                dtype=h5py.string_dtype(),
            )
            # After a configured filter, and of fixed length.
            light_curve.create_dataset(
                "TFA/FitProperties/PointsFilterExpression",
                data=numpy.array(
                    [f"(mag < 10) and {match(inside)}".encode("ascii")]
                ),
            )

        self.assertTrue(_load_revision()._migrate_lc(fname, self._project_home))

        with h5py.File(fname, "r") as light_curve:
            for name, expected in (
                ("EPD/FitProperties/Filter", [converted, match(outside)]),
                (
                    "TFA/FitProperties/PointsFilterExpression",
                    [f"(mag < 10) and {converted}"],
                ),
            ):
                with self.subTest(dataset=name):
                    self.assertEqual(
                        light_curve[name][()].tolist(),
                        [value.encode("ascii") for value in expected],
                    )

    def test_leaves_a_file_without_paths_to_convert_alone(self):
        """A file already converted is not written."""

        fname = self._get_abs_path("lc.h5")
        with h5py.File(fname, "w") as light_curve:
            light_curve.create_dataset(
                f"fixed/{SPHOTREF_NAME}",
                data=numpy.array([b"{PROJHOME}/sub/f.fits"] * 2),
            )
        self._assert_unchanged(fname, _load_revision()._migrate_lc)


# The fixture's data, written in setUp and checked by the tests.
# pylint: disable-next=too-many-instance-attributes
class TestProjhomeProjectMigration(ProjectHomeMixin, unittest.TestCase):
    """
    A project holding every kind of product is migrated whole.

    The project has frames ``frame1`` (inside the project, processed in
    channels ``R`` and ``G``), ``frame2`` (outside, in ``B``) and ``frame3``
    (inside, in ``R``, but missing). ``calibrated-fname`` takes two values,
    the second, under another condition, as an absolute path.
    ``master-photref-fname-format`` takes two values too: the first, as an
    absolute path, names the masters magnitude fitting left; the second,
    under another condition, needs a keyword the reference's header lacks.
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
                ],
                "master-photref-fname-format": [
                    (
                        None,
                        self._get_abs_path(
                            "MASTERS",
                            "mphotref_{OBJECT}_iter{magfit_iteration:03d}.fits",
                        ),
                    ),
                    (
                        ["EXPTIME > 100"],
                        "{PROJHOME}/MASTERS/mphotref_{FILTER}"
                        "_iter{magfit_iteration:03d}.fits",
                    ),
                ],
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
            write_fits(fname, [("ORIGF001", raw[frame])] + common_cards)
        # Named by a template, but for a channel never processed.
        self._unprocessed = self._get_abs_path("CAL", "frame1_B.fits.fz")
        write_fits(self._unprocessed, [("ORIGF001", raw["frame1"])])

        self._dr = {
            self._sphotref: "frame1",
            self._get_abs_path("DR", "frame1_G.h5"): "frame1",
            self._get_abs_path("DR", "frame2_B.h5"): "frame2",
        }
        for fname, frame in self._dr.items():
            write_dr(
                fname,
                self._sphotref,
                [("ORIGF001", raw[frame]), ("OBJECT", "field1")] + common_cards,
            )

        write_fits(
            bias,
            [
                ("ORIGF001", raw["frame1"]),
                ("PROJHOME", self._project_home),
            ],
        )
        lc_catalog = self._get_abs_path("MASTERS", "lc_catalog.fits")
        Table(
            {"source_id": numpy.array([1, 2], dtype="uint64")},
            meta={"PROJHOME": self._project_home},
        ).write(lc_catalog)
        # Magnitude fitting leaves a master for each iteration, named by the
        # template filled with the single photometric reference's header, and
        # only the last is recorded.
        self._mphotrefs = [
            self._get_abs_path(
                "MASTERS", f"mphotref_field1_iter{iteration:03d}.fits"
            )
            for iteration in range(3)
        ]
        for fname in self._mphotrefs:
            write_master_photref(
                fname,
                [("MBIASFNM", bias)],
                {
                    dr_fname: frame == "frame1"
                    for dr_fname, frame in self._dr.items()
                },
            )
        self._masters = {
            "zero": bias,
            "lightcurve_catalog": lc_catalog,
            "master_photref": self._mphotrefs[-1],
        }

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
                f"AperturePhotometry/{SPHOTREF_NAME}",
                data=numpy.array([self._sphotref.encode("ascii")] * 3),
            )

    def _add_rows(self):
        """Add the observing session, its images and the masters."""

        with start_db_session() as db_session:
            # False positive: the declarative models are callable.
            # pylint: disable=not-callable
            session_id = add_observing_session(
                db_session, "night", add_cameras(db_session)[0]
            )

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
                    observing_session_id=session_id,
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
                    "{PROJHOME}/MASTERS/mphotref_field1_iter002.fits",
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
                    dr_file.attrs[SPHOTREF_NAME], "{PROJHOME}/DR/frame1_R.h5"
                )
                header = read_dr_header(dr_file)
                self.assertEqual(header["ORIGF001"], self._expected_raw[frame])
                self.assertNotIn("PROJHOME", header)

        bias_header = fits.getheader(self._masters["zero"], 1)
        self.assertEqual(bias_header["ORIGF001"], "{PROJHOME}/RAW/frame1.fits")
        self.assertNotIn("PROJHOME", bias_header)
        self.assertNotIn(
            "PROJHOME",
            fits.getheader(self._masters["lightcurve_catalog"], 1),
        )
        for fname in self._mphotrefs:
            with self.subTest(mphotref=fname), fits.open(fname) as mphotref:
                self.assertEqual(
                    mphotref[0].header["MBIASFNM"],
                    "{PROJHOME}/MASTERS/bias.fits",
                )
                population = mphotref["INMASTER"].data
                self.assertEqual(
                    list(
                        zip(
                            population["dr_fname"],
                            population["included"].tolist(),
                        )
                    ),
                    [
                        ("{PROJHOME}/DR/frame1_R.h5", True),
                        ("{PROJHOME}/DR/frame1_G.h5", True),
                        ("{PROJHOME}/DR/frame2_B.h5", False),
                    ],
                )

        for fname in self._lightcurves:
            with (
                self.subTest(lightcurve=fname),
                h5py.File(fname, "r") as light_curve,
            ):
                self.assertEqual(
                    light_curve[f"AperturePhotometry/{SPHOTREF_NAME}"][
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
