#!/usr/bin/env python3

import logging

import os

from collections import namedtuple

from superphot_pipeline.light_curves import LCDataReader
from superphot_pipeline import DataReductionFile

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ConfigType = namedtuple('ConfigType', ['max_apertures',
                                           'max_magfit_iterations',
                                           'catalogue_fname',
                                           'srcextract_psf_params'])
    configuration = ConfigType(
        max_apertures=40,
        max_magfit_iterations=6,
        catalogue_fname=(
            '/data/HAT10DSLR_moved/MASTERS/cat_object_G10124500_139_2.ucac4'
        ),
        srcextract_psf_params=['S', 'D', 'K']
    )

    with DataReductionFile('dummy_dr.h5', 'w') as dummy_dr:
        read_data = LCDataSlice.create(configuration,
                                       dummy_dr.parse_hat_source_id)

    os.remove('dummy_dr.h5')

    data_dir = join_paths(dirname(__file__), 'test_data')
    dr_fname = join_paths(data_dir, '10-464933_2_R1.hdf5.0')
    path_substitutions = dict(srcextract_version=0,
                              catalogue_version=0,
                              skytoframe_version=0)

    read_data((dr_fname, 0), **path_substitutions)
