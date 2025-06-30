"""A collection of functions for finding files to process."""

from functools import partial
import os.path
from glob import iglob
from logging import getLogger

from autowisp import Evaluator
from autowisp.fits_utilities import get_primary_header

_logger = getLogger(__name__)


def get_data_filenames(
    data_collection, include_condition="True", recursive=True
):
    """
    Select FITS images or DR files that match a specified header condition.

    Args:
        data_collection([str]):    A set of patterns passed directly to glob to
            search for suitable files.

        recursive(bool):    Should inputs globs be searched recursively?

        include_condition(str):    See find_data_fnames()

    Yields:
        The files from the input collection that satisfy the given condition.
    """

    for entry in data_collection:
        for fname in iglob(entry, recursive=recursive):
            _logger.debug("fname before pass: %s", repr(fname))
            if include_condition == "True" or Evaluator(fname)(
                include_condition
            ):
                _logger.debug("fname passed: %s", repr(fname))
                yield fname


def find_data_fnames(
    image_collection,
    include_condition="True",
    *,
    recursive=False,
    search_wildcards=("*.fits", "*.fits.fz"),
):
    """
    Select FITS images matching a header condition.

    Args:
        data_collection(list):    A list of directories or individual images to
            search for suitable files. For directories, images with '.fits' or
            'fits.fz' extensions are selected.

        include_condition(str):    Expression involving the header of the
            images that evaluates to True/False if a particular image from the
            specified image collection should/should not be processed.

        recursive(bool):    Should directories be searched recursively for
            images or just their top level.

        search_wildcards(str iter):    Filename wildcards to search for.

    Yields:
        The images specified in the `image_colleciton` argument that satisfy the
        given condition.
    """

    if isinstance(image_collection, str):
        image_collection = [image_collection]
    if recursive:
        search_wildcards = ["**/" + wildcard for wildcard in search_wildcards]
    for entry in image_collection:
        if os.path.isdir(entry):
            for result in get_data_filenames(
                [
                    os.path.join(entry, wildcard)
                    for wildcard in search_wildcards
                ],
                include_condition,
            ):
                yield result
        else:
            for result in get_data_filenames([entry], include_condition):
                yield result


def find_fits_with_dr_fnames(
    image_collection, include_condition="True", *, dr_fname_format, **kwargs
):
    """Same as find_data_fnames() but eliminates those without DR files."""

    def has_dr(fits_fname):
        """Check if a FITS file has a corresponding DR file."""

        header = get_primary_header(fits_fname)
        _logger.debug(
            "looking for: %s", repr(dr_fname_format.format_map(header))
        )
        return os.path.exists(dr_fname_format.format_map(header))

    return filter(
        has_dr, find_data_fnames(image_collection, include_condition, **kwargs)
    )


find_fits_fnames = find_data_fnames
find_dr_fnames = partial(find_data_fnames, search_wildcards=("*.h5", "*.hdf5"))
find_lc_fnames = find_dr_fnames
