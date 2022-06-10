#!/usr/bin/env python3

"""Example of how to add HAT-style astrometry results to DR file."""

from os.path import dirname, join as join_paths
from collections import namedtuple

import scipy
from astropy.io import fits

from superphot_pipeline import DataReductionFile
from superphot_pipeline.data_reduction.utils import get_source_extracted_psf_map

if __name__ == '__main__':
    data_dir = join_paths(dirname(__file__), 'test_data', '10-20170306')

    for fnum in [464933, 465210, 465211, 465215, 465216, 465217, 465218, 465219, 465248]:
        dr_fname = join_paths(data_dir, '10-%d_2_R1.hdf5' % fnum)
        fits_fname = join_paths(data_dir, '10-%d_2_R1.fits.fz' % fnum)

        astrom_filenames = dict(
            fistar=join_paths(data_dir, '10-%d_2_R1.fistar' % fnum),
            trans=join_paths(data_dir, '10-%d_2_R1.fistar.trans' % fnum),
            match=join_paths(data_dir, '10-%d_2_R1.fistar.match' % fnum),
            catalogue=join_paths(data_dir, 'test.ucac4')
        )

        ConfigType = namedtuple(
            'ConfigType',
            [
                'srcextract_binning',
                'astrom_catalogue_name',
                'astrom_catalogue_epoch',
                'astrom_catalogue_filter',
                'astrom_catalogue_fov',
                'astrom_catalogue_orientation',
                'skytoframe_srcextract_filter',
                'skytoframe_sky_preprojection',
                'skytoframe_max_match_distance',
                'skytoframe_frame_center',
                'skytoframe_weights_expression'

            ]
        )
        astrom_config = ConfigType(
            srcextract_binning=1,
            astrom_catalogue_name='UCAC4',
            astrom_catalogue_epoch=2457819.61889779,
            astrom_catalogue_filter='(r>=4) & (r<=14)',
            astrom_catalogue_fov=18.0,
            astrom_catalogue_orientation=0.0,
            skytoframe_srcextract_filter='True',
            skytoframe_sky_preprojection='tan',
            skytoframe_max_match_distance=1.5,
            skytoframe_frame_center=(1329.5, 1991.5),
            skytoframe_weights_expression='1.0'
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
                                              astrom_config,
                                              **path_substitutions)
            data_reduction.smooth_srcextract_psf(['S', 'D', 'K'],
                                                 'O3{x, y, r, J-K}',
                                                 weights_expression='1',
                                                 error_avg='median',
                                                 rej_level=5.0,
                                                 max_rej_iter=20,
                                                 **path_substitutions)
            psf_map = get_source_extracted_psf_map(data_reduction,
                                                   **path_substitutions)
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
