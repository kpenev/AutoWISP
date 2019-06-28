#!/usr/bin/env python3

"""Demonstrate light curve dumping."""

import logging

from os.path import join as join_paths, dirname

from collections import namedtuple

from superphot_pipeline.light_curves.collect_light_curves import\
    collect_light_curves

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data_dir = join_paths(dirname(__file__), 'test_data/10-20170306')
    dr_fname = join_paths(data_dir, '10-464933_2_R1.hdf5')

    ConfigType = namedtuple('ConfigType', ['max_apertures',
                                           'max_magfit_iterations',
                                           'catalogue_fname',
                                           'srcextract_psf_params',
                                           'memblocksize'])
    configuration = ConfigType(
        max_apertures=39,
        max_magfit_iterations=6,
        catalogue_fname=join_paths(data_dir,
                                   'cat_object_G10124500_139_2.ucac4'),
        srcextract_psf_params=['S', 'D', 'K'],
        memblocksize=1023**3
    )

    path_substitutions = dict(srcextract_version=0,
                              catalogue_version=0,
                              skytoframe_version=0,
                              shapefit_version=0,
                              apphot_version=0,
                              background_version=0,
                              magfit_version=0,
                              srcproj_version=0)

    collect_light_curves([dr_fname], configuration, **path_substitutions)
