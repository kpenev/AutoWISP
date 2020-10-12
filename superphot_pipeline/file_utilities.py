"""Collection of utilities for working with files."""

import os
import os.path
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from subprocess import call

from astropy.io import fits

def prepare_file_output(fname,
                        allow_overwrite,
                        allow_dir_creation,
                        delete_existing=False):
    """Ger ready to create/overwrite a file with the given name."""

    if os.path.exists(fname):
        if not allow_overwrite:
            raise OSError(
                'Destination source extraction file %s already exists '
                'and overwritting not allowed!'
                %
                repr(fname)
            )
        if delete_existing:
            os.remove(fname)

    out_path = os.path.dirname(fname)
    if allow_dir_creation and out_path and not os.path.exists(out_path):
        os.makedirs(out_path)

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
            print('Unpacked frame name: ' + repr(unpacked_frame.name))
            assert call(
                ['funpack', '-C', '-S', fits_fname],
                stdout=unpacked_frame
            ) == 0
            call(['ls', '-lh', unpacked_frame.name])
            yield unpacked_frame.name
            assert os.path.exists(unpacked_frame.name)
    else:
        yield fits_fname
