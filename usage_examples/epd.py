#!/usr/bin/env python3

"""Demonstrate the usage of EPD."""

from glob import glob
from os.path import join as join_paths, dirname
import logging

from superphot_pipeline import parallel_epd

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    lc_fname_glob = join_paths(dirname(__file__),
                               'test_data',
                               '10-20170306',
                               'lcs',
                               '*.hdf5')
    epd_result = parallel_epd(
        glob(lc_fname_glob),
        5,
        used_variables=dict(
            x=('srcproj.x', dict()),
            y=('srcproj.y', dict()),
            S=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='S')),
            D=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='D')),
            K=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='K')),
            bg=('bg.value', dict())
        ),
        fit_points_filter_expression=None,
        fit_terms_expression='O2{x}',
        fit_datasets=(
            [
                (
                    'shapefit.magfit.magnitude',
                    dict(magfit_iteration=5),
                    'shapefit.epd.magnitude'
                )
            ]
            +
            [
                (
                    'apphot.magfit.magnitude',
                    dict(magfit_iteration=5, aperture_index=ap_ind),
                    'apphot.epd.magnitude'
                )
                for ap_ind in range(39)
            ]
        ),
        error_avg='nanmedian',
        rej_level=5,
        max_rej_iter=20
    )
    print(repr(epd_result))
