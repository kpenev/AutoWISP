"""Files for tests of revision 0018 to convert, and the paths they record.

Each test gets a project home of its own (:class:`ProjectHomeMixin`), and
writes into it the products the revision converts, laid out as the pipeline
writes them.
"""

import os
import tempfile

from astropy.io import fits
import h5py
import numpy

#: Where products record the single photometric reference they used.
SPHOTREF_NAME = "SinglePhotometricReference"  # pylint: disable=invalid-name


def get_path_cases(project_home):
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
        # Inside, whether or not the file is still there.
        "sub/missing.fits": "{PROJHOME}/sub/missing.fits",
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


def touch(fname):
    """Create an empty file, and the directories it is in."""

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb"):
        pass


def write_fits(fname, cards, *, compressed=True, checksum=False):
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


def get_hdu_bytes(fname):
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


def parse_header(header_bytes):
    """Return the header the given bytes of a FITS file hold."""

    return fits.Header.fromstring(header_bytes.decode("ascii"))


def count_cards(header_bytes):
    """Return the number of cards a header uses, its END card included."""

    count = 0
    for start in range(0, len(header_bytes), 80):
        count += 1
        if header_bytes[start : start + 80].startswith(b"END     "):
            return count
    raise ValueError("No END card in the header.")


def write_dr(fname, sphotref, header_cards):
    """Write a DR file recording a photometric reference and a header."""

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with h5py.File(fname, "w") as dr_file:
        dr_file.attrs[SPHOTREF_NAME] = sphotref
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


def write_master_photref(fname, header_cards, population):
    """
    Write a master photometric reference laid out as magnitude fitting does.

    Args:
        fname(str):    The file to write.

        header_cards([(str, str)]):    The cards of its primary header.

        population(dict):    ``{dr_fname: included}`` for its ``INMASTER``
            table.
    """

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=fits.Header(header_cards)),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="source_id",
                        format="K",
                        array=numpy.array([3, 1, 2]),
                    ),
                    fits.Column(
                        name="magnitude",
                        format="D",
                        array=numpy.array([10.125, 11.5, 12.0625]),
                    ),
                ],
                name="MPHOTREF",
            ),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="dr_fname",
                        format=f"{max(map(len, population))}A",
                        array=numpy.array(list(population), dtype=str),
                    ),
                    fits.Column(
                        name="included",
                        format="L",
                        array=numpy.array(list(population.values())),
                    ),
                ],
                name="INMASTER",
            ),
        ]
    ).writeto(fname, overwrite=True)


def read_dr_header(dr_file):
    """Return the copy of the frame header an open DR file holds."""

    return fits.Header.fromstring(
        dr_file["FITSHeader"][()].tobytes().decode("ascii")
    )


class ProjectHomeMixin:  # pylint: disable=too-few-public-methods
    """Give each test a project home holding ``sub/f.fits``."""

    # Mixed into unittest.TestCase subclasses, so setUp is an override
    # despite this class not deriving from TestCase itself.
    # pylint: disable=invalid-name
    def setUp(self):
        """Create the project home."""

        super().setUp()
        # Released via addCleanup, since it must outlive setUp.
        # pylint: disable-next=consider-using-with
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._project_home = tmp.name
        touch(self._get_abs_path("sub", "f.fits"))

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
