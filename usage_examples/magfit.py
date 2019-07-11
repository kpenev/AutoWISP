#!/usr/bin/env python3

"""Test magnitude fitting class."""

from glob import glob
from os.path import basename, dirname, join as join_paths
from collections import namedtuple
import logging

from superphot_pipeline import magnitude_fitting

if __name__ == '__main__':
    data_dir = join_paths(dirname(__file__), 'test_data')

    logging.basicConfig(level=logging.DEBUG)
    ConfigType = namedtuple('ConfigType', ['correction_parametrization',
                                           'reference_subpix',
                                           'fit_source_condition',
                                           'grouping',
                                           'error_avg',
                                           'rej_level',
                                           'max_rej_iter',
                                           'noise_offset',
                                           'max_mag_err',
                                           'num_parallel_processes',
                                           'max_photref_change'])
    configuration = ConfigType(
        correction_parametrization=(
            ' + '.join([
                'O4{xi, eta}',
                'O2{R} * O2{xi, eta}',
                'O1{J-K} * O1{xi, eta}',
                'O1{x % 1, y % 1}',
            ])
        ),
        reference_subpix=False,
        fit_source_condition='(r > 0) * (r < 16) * (J - K > 0) * (J - K < 1)',
        grouping='(enabled,)',
        error_avg='weightedmean', # also try median
        rej_level=5.0,
        max_rej_iter=20,
        noise_offset=0.01,
        max_mag_err=0.1,
        num_parallel_processes=1,
        max_photref_change=1e-4
    )

    path_substitutions = dict(shapefit_version=0,
                              srcproj_version=0,
                              apphot_version=0,
                              background_version=0,
                              magfit_version=0)

    fit_dr_filenames = sorted(glob(join_paths(data_dir, '10-*_2_R1.hdf5')))

    magnitude_fitting.iterative_refit(
        fit_dr_filenames=fit_dr_filenames,
        single_photref_dr_fname=join_paths(data_dir, '10-465248_2_R1.hdf5'),
        master_catalogue_fname=join_paths(data_dir,
                                          'cat_object_G10124500_139_2.ucac4'),
        configuration=configuration,
        master_photref_fname_pattern=join_paths(
            data_dir,
            'mphotref_iter%(magfit_iteration)03d.fits'
        ),
        magfit_stat_fname_pattern=join_paths(
            data_dir,
            'mfit_stat_iter%(magfit_iteration)03d.txt'
        ),
        **path_substitutions
    )
