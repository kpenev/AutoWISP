"""Uniform interface for source extractoin using several methods."""

import numpy
from astropy.io import fits
from scipy import spatial
from numpy.lib import recfunctions

from astrowisp.utils.file_utilities import (
    prepare_file_output,
    get_unpacked_fits,
)
from autowisp import source_finder_util
from autowisp.evaluator import Evaluator


# This still makes sense as a class
# pylint: disable=too-few-public-methods
class SourceFinder:
    """Find sources in an image of the night sky and repor properties."""

    @staticmethod
    def _create_mock_source_list(fits_fname, configuration):
        """Create a source list with randomly generated properties."""

        with fits.open(fits_fname) as fits_file:
            # False positive
            # pylint: disable=no-member
            hdu_index = 0 if fits_file[0].header["NAXIS"] else 1
            xresolution = fits_file[hdu_index].header["NAXIS1"]
            yresolution = fits_file[hdu_index].header["NAXIS2"]
            # pylint: disable=no-member
            med_pixel = numpy.median(fits_file[hdu_index].data)
        nsources = 1000
        result = numpy.empty(
            nsources,
            dtype=[
                (
                    name,
                    (numpy.int32 if name in ["id", "npix"] else numpy.float64),
                )
                for name in source_finder_util.get_srcextract_columns("fistar")
            ],
        )

        result["id"] = numpy.arange(nsources)
        # False positive
        # pylint: disable=no-member
        result["x"] = numpy.random.random(nsources) * xresolution
        result["y"] = numpy.random.random(nsources) * yresolution

        result["bg"] = (1.0 + 0.1 * numpy.random.random(nsources)) * med_pixel
        result["flux"] = (
            numpy.random.random(nsources)
            * configuration["brightness_threshold"]
        )
        result["amp"] = 0.2 * result["flux"]
        result["s/n"] = result["flux"] / result["bg"]

        result["s"] = 2.3 + 0.2 * numpy.random.random(nsources)
        result["d"] = 0.3 + 0.1 * numpy.random.random(nsources)
        result["k"] = 0.3 + 0.1 * numpy.random.random(nsources)

        result["npix"] = 20 + (5 * numpy.random.random(nsources)).astype(int)
        # pylint: enable=no-member
        return result

    def _add_saturation_flags(self, fits_fname, source_list):
        """Add array of how many pixels near source are saturated."""

        with fits.open(fits_fname, "readonly") as fits_file:
            saturated_kdtree = spatial.KDTree(
                numpy.column_stack(numpy.nonzero(fits_file[2].data)) + 0.5
            )

        source_coords = numpy.column_stack((source_list["y"], source_list["x"]))
        print(
            "Match radii: " + repr(numpy.sqrt(source_list["npix"] / numpy.pi))
        )
        saturated_in_range = saturated_kdtree.query_ball_point(
            source_coords, numpy.sqrt(source_list["npix"] / numpy.pi)
        )
        print(f"Saturated in range shape: {saturated_in_range.shape}")
        print(f"Saturated in range: {saturated_in_range!r}")

        return recfunctions.append_fields(
            source_list,
            "nsatpix",
            [len(saturated) for saturated in saturated_in_range],
            usemask=False,
        )

    def __init__(
        self,
        *,
        tool="hatphot",
        brightness_threshold=10,
        filter_sources="True",
        max_sources=0,
        allow_overwrite=False,
        allow_dir_creation=False,
        always_return_sources=False,
    ):
        """Prepare to use the specified tool and define faint limit."""

        self.configuration = {
            "tool": tool,
            "brightness_threshold": brightness_threshold,
            "allow_overwrite": allow_overwrite,
            "allow_dir_creation": allow_dir_creation,
            "always_return_sources": always_return_sources,
            "filter_sources": filter_sources,
            "max_sources": max_sources,
        }

    def __call__(self, fits_fname, source_fname=None, **configuration):
        """
        Extract the sources from the given frame and save or return them.

        Args:
            fits_fname(str):    The filename of the fits file to extract
                sources from. Can be packed or unpacked.

            source_fname(str or None):    If None, the extract sources are
                returned as numpy. field array. Otherwise, this specifies a file
                to save the source extractor output.

        Returns:
            field array or None:
                The extracted source from the image, if source_fname was None.
        """

        configuration = {**self.configuration, **configuration}
        if configuration["tool"] == "mock":
            return self._create_mock_source_list(fits_fname, configuration)

        start_extraction = getattr(
            source_finder_util, "start_" + configuration["tool"]
        )
        with get_unpacked_fits(fits_fname) as unpacked_fname:
            extraction_args = (
                unpacked_fname,
                configuration["brightness_threshold"],
            )
            if source_fname:
                prepare_file_output(
                    source_fname,
                    configuration["allow_overwrite"],
                    configuration["allow_dir_creation"],
                )
                with open(source_fname, "wb") as destination:
                    start_extraction(*extraction_args, destination).wait()
                if not configuration["always_return_sources"]:
                    return None
                sources_file = source_fname
            else:
                extraction_process = start_extraction(*extraction_args)
                sources_file = extraction_process.stdout

            result = numpy.genfromtxt(
                sources_file,
                names=source_finder_util.get_srcextract_columns(
                    configuration["tool"]
                ),
                dtype=None,
                deletechars="",
            )
            if configuration["tool"] == "hatphot":
                result["x"] -= 0.5
                result["y"] -= 0.5
            if configuration["filter_sources"] != "True":
                result = result[
                    Evaluator(result)(configuration["filter_sources"])
                ]
            result.sort(order="flux")
            result = numpy.flip(result)
            if configuration["max_sources"] > 0:
                result = result[: configuration["max_sources"]]

            if not source_fname:
                extraction_process.communicate()
            return self._add_saturation_flags(unpacked_fname, result)


# pylint: enable=too-few-public-methods
