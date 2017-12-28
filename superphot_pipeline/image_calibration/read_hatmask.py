"""
Provide an interface for working with HAT-style masks in FITS headers.

Examples:
    from astropy.io import fits

    with fits.open('/Users/kpenev/tmp/1-447491_4.fits.fz',
                   mode='readonly') as f:
        image_mask = parse_hat_mask(f[1].header)

        flag_name = 'OVERSATURATED'

        matched = numpy.bitwise_and(image_mask,
                                    mask_flags[flag_name]).astype(bool)

        #Print number of pixels for which the OVERSATURATED flag is raised
        print(flag_name + ': ' + repr(matched.sum()))

        #Output x, y, flux for the pixels flagged as OVERSATURATED
        for y, x in zip(*numpy.nonzero(matched)) :
            print('%4d %4d %15d' % (x, y, f[1].data[y, x]))
"""

from ctypes import cdll, c_long, c_byte, c_char_p
from ctypes.util import find_library
import numpy

mask_flags = dict()

def initialize_library():
    """Prepare the orbital evolution library for use."""

    library_fname = find_library('io')
    if library_fname is None:
        raise OSError("Unable to find SuperPhot's io library.")
    lib = cdll.LoadLibrary(library_fname)

    lib.parse_hat_mask.argtypes = [
        c_char_p,
        c_long,
        c_long,
        numpy.ctypeslib.ndpointer(dtype=c_byte,
                                  ndim=1,
                                  flags='C_CONTIGUOUS')
    ]
    lib.parse_hat_mask.restype = None

    for mask_flag in ['OK',
                      'CLEAR',
                      'FAULT',
                      'HOT',
                      'COSMIC',
                      'OUTER',
                      'OVERSATURATED',
                      'LEAKED',
                      'SATURATED',
                      'INTERPOLATED',
                      'BAD',
                      'ALL',
                      'NAN']:
        mask_flags[mask_flag] = c_byte.in_dll(lib, 'MASK_' + mask_flag).value

    return lib

library = initialize_library()

def parse_hat_mask(header):
    """
    Extract the HAT-style mask contained in header.

    Args:
        header:    The header of the image whose mask to parse.

    Returns:
        mask:    A dtype=uint8 numpy array with exactly the same resolution as
            the input image containing a bit-field for each pixel indicating
            any bad-pixel flags raised per the header.
    """

    mask_string = ''.join((c[1] + ' ') if c[0] == 'MASKINFO' else ''
                          for c in header.items()).encode('ascii')
    mask = numpy.zeros((header['NAXIS2'], header['NAXIS1']), dtype='int8')
    library.parse_hat_mask(mask_string,
                           header['NAXIS1'],
                           header['NAXIS2'],
                           mask.ravel())
    return mask
