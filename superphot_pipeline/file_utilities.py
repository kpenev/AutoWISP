"""Collection of utilities for working with files."""

import os
import os.path
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from subprocess import call
import logging

_logger = logging.getLogger(__name__)

from astropy.io import fits

def prepare_file_output(fname,
                        allow_overwrite,
                        allow_dir_creation,
                        delete_existing=False):
    """Ger ready to create/overwrite a file with the given name."""

    result = False
    if os.path.exists(fname):
        if not allow_overwrite:
            raise OSError(
                'Destination file %s already exists and overwritting not '
                'allowed!'
                %
                repr(fname)
            )
        if delete_existing:
            _logger.info('Overwriting %s', fname)
            os.remove(fname)
        else:
            result = True

    out_path = os.path.dirname(fname)
    if allow_dir_creation and out_path and not os.path.exists(out_path):
        _logger.info('Creating output directory: %s',
                     repr(out_path))
        os.makedirs(out_path)

    return result

@contextmanager
def get_unpacked_fits(fits_fname):
    """Ensure the result is an unpacked version of the frame."""

    with fits.open(fits_fname, 'readonly') as fits_file:
        #False positive
        #pylint: disable=no-member
        packed = fits_file[0].header['NAXIS'] == 0
        #pylint: enable=no-member

    if packed:
        with NamedTemporaryFile(buffering=0, dir='/dev/shm') as unpacked_frame:
            assert call(
                ['funpack', '-C', '-S', fits_fname],
                stdout=unpacked_frame
            ) == 0
            yield unpacked_frame.name
    else:
        yield fits_fname

def get_fits_fname_root(fits_fname):
    """Return the FITS filename withou directories or extension."""

    result = os.path.basename(fits_fname)
    while True:
        result, extension = os.path.splitext(result)
        if not extension:
            return result
