#!/usr/bin/env python3

"""Demonstrate light curve dumping."""

import logging
from glob import glob
from os.path import join as join_paths, dirname
from collections import namedtuple

from superphot_pipeline.light_curves.collect_light_curves import\
    collect_light_curves

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data_dir = join_paths(dirname(__file__), 'test_data', '10-20170306')
    dr_fnames = glob(join_paths(data_dir, '10-*_2_R1.hdf5'))
    lc_dir = join_paths(data_dir, 'lcs')

    ConfigType = namedtuple('ConfigType', ['max_apertures',
                                           'max_magfit_iterations',
                                           'catalogue_fname',
                                           'srcextract_psf_params',
                                           'memblocksize',
                                           'lc_fname_pattern'])
    configuration = ConfigType(
        max_apertures=39,
        max_magfit_iterations=6,
        catalogue_fname=join_paths(data_dir,
                                   'lc_dump_catalogue.ucac4'),
        lc_fname_pattern=join_paths(lc_dir, '%d-%d-%d.hdf5'),
        srcextract_psf_params=['S', 'D', 'K'],
        memblocksize=(80.0 * 1024**2)
    )

    path_substitutions = dict(srcextract_version=0,
                              catalogue_version=0,
                              skytoframe_version=0,
                              shapefit_version=0,
                              apphot_version=0,
                              background_version=0,
                              magfit_version=0,
                              srcproj_version=0)

    collect_light_curves(dr_fnames, configuration, **path_substitutions)
