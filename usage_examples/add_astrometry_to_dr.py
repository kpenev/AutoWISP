#!/usr/bin/env python3

"""Example of how to add HAT-style astrometry results to DR file."""

from os.path import dirname, join as join_paths

import scipy
from astropy.io import fits

from superphot_pipeline import DataReductionFile

if __name__ == '__main__':
    data_dir = join_paths(dirname(__file__), 'test_data')

    dr_fname = join_paths(data_dir, '10-464933_2_R1.hdf5.0')
    fits_fname = join_paths(data_dir, '10-464933_2_R1.fits.fz')

    astrom_filenames = dict(
        fistar=join_paths(data_dir, '10-464933_2_R1.fistar'),
        trans=join_paths(data_dir, '10-464933_2_R1.fistar.trans'),
        match=join_paths(data_dir, '10-464933_2_R1.fistar.match'),
        catalogue=join_paths(data_dir, 'test.ucac4')
    )

    path_substitutions = dict(srcextract_version=0,
                              catalogue_version=0,
                              skytoframe_version=0)
    with DataReductionFile(dr_fname, 'r+') as data_reduction:

        with fits.open(fits_fname, 'readonly') as frame:
            #False positive
            #pylint: disable=no-member
            data_reduction.add_frame_header(frame[1].header)
            #pylint: enable=no-member

        data_reduction.add_hat_astrometry(astrom_filenames,
                                          **path_substitutions)
        data_reduction.smooth_srcextract_psf(['S', 'D', 'K'],
                                             'O3{x, y, r, J-K}',
                                             weights_expression='1',
                                             error_avg='median',
                                             rej_level=5.0,
                                             max_rej_iter=20,
                                             **path_substitutions)
        psf_map = data_reduction.get_source_extracted_psf_map(
            **path_substitutions
        )
    print(
        repr(
            psf_map(
                scipy.array(
                    [(1000.0, 1000.0, 11.0, 12.0, 12.0)],
                    dtype=[('x', float),
                           ('y', float),
                           ('r', float),
                           ('J', float),
                           ('K', float)]
                )
            )
        )
    )
