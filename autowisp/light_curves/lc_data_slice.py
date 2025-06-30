"""Define a class holding a slice of LC data organize by source."""

import logging
from ctypes import (
    c_bool,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_uint64,
    c_uint32,
    c_uint16,
    c_uint8,
    c_float,
    c_double,
    c_longdouble,
    Structure,
    sizeof,
    c_char_p,
)

import numpy

_logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class LCDataSliceBase(Structure):
    """A time-slice of LC data to be shared between LC dumping processes."""

    # The point is to deal with the many branches
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    @staticmethod
    def get_ctype(dtype):
        """Return the appropriate c-types type to use for the given dtype."""

        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)

        if dtype.kind == "b":
            return c_bool

        if dtype.kind == "i":
            assert dtype.itemsize <= 8
            if dtype.itemsize == 8:
                return c_int64
            if dtype.itemsize == 4:
                return c_int32
            if dtype.itemsize == 2:
                return c_int16
            if dtype.itemsize == 1:
                return c_int8

        elif dtype.kind == "u":
            assert dtype.itemsize <= 8
            if dtype.itemsize == 8:
                return c_uint64
            if dtype.itemsize == 4:
                return c_uint32
            if dtype.itemsize == 2:
                return c_uint16
            if dtype.itemsize == 1:
                return c_uint8

        elif dtype.kind == "f":
            assert dtype.itemsize <= sizeof(c_longdouble)
            if dtype.itemsize == sizeof(c_longdouble):
                return c_longdouble
            if dtype.itemsize == sizeof(c_double):
                return c_double
            if dtype.itemsize == sizeof(c_float):
                return c_float

        elif dtype.kind == "S":
            return c_char_p

        raise TypeError("Unrecognized dtype: " + repr(dtype))

    # pylint: enable=too-many-return-statements
    # pylint: enable=too-many-branches


def create_lc_data_slice_type(
    get_dtype, dataset_dimensions, max_dimension_size, max_mem
):
    """
    Return LCDataSliceBase sub-class configured to hold max LC data possible.

    Args:
        get_dtype(callable):    A function which should return the datatype
            to use for the dataset corresponding to a given pipeline key.

        dataset_dimensions:    See :attr:`LCDataReader.dataset_dimensions`

        max_dimension_size:    See :attr:`LCDataReader.max_dimension_size`

        max_mem:    The maximum amount of memory instances are allowed to
            consume in bytes.

    Returns:
        LCDataSliceBase sub-class:
            The class of a variable to use for lightcurve dumping

        int:
            The number of frames that will fit into the structure.
    """

    atomic_ctypes = {}

    dset_size = {}
    perframe_bytes = 0
    for dset_name, dset_dimensions in dataset_dimensions.items():
        if "frame" in dset_dimensions or "source" in dset_dimensions:
            if dset_name == "source_in_frame":
                atomic_ctypes[dset_name] = c_bool
            else:
                atomic_ctypes[dset_name] = LCDataSliceBase.get_ctype(
                    get_dtype(dset_name)
                )

            dset_size[dset_name] = 1
            for dimension in dset_dimensions:
                if dimension != "frame":
                    dset_size[dset_name] *= max_dimension_size[dimension]
            atomic_size = sizeof(atomic_ctypes[dset_name])
            if atomic_size == 0:
                assert dset_name.startswith("fitsheader.")
                atomic_size = 70
            perframe_bytes += atomic_size * dset_size[dset_name]

            # Too complicated to make lazy
            # pylint: disable=logging-not-lazy
            _logger.debug(
                f"Dset: {dset_name} size = {dset_size[dset_name]:d} ("
                + " x ".join(
                    f"({max_dimension_size[dimension]:d} {dimension!s})"
                    for dimension in filter(
                        lambda d: d != "frame", dset_dimensions
                    )
                )
            )
            # pylint: enable=logging-not-lazy

    num_frames = min(int(max_mem / perframe_bytes), 1000)

    # That's per the intended use
    # pylint: disable=missing-class-docstring
    class LCDataSlice(LCDataSliceBase):
        _fields_ = [
            (
                dset_name.replace(".", "_"),
                num_frames * num_entries * atomic_ctypes[dset_name],
            )
            for dset_name, num_entries in dset_size.items()
        ]

    # pylint: enable=missing-class-docstring

    return LCDataSlice, num_frames


# pylint: enable=too-few-public-methods
