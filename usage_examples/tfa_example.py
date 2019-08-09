#!/usr/bin/env python3

"""Example of how to perform TFA correction to EPD-ed lightcurves."""

from os.path import join as join_paths, dirname
from collections import namedtuple

from superphot_pipeline import TFA, load_epd_statistics

if __name__ == '__main__':
    data_dir = join_paths(dirname(__file__),
                          'test_data',
                          '10-20170306',
                          'lcs')

    ConfigType = namedtuple('ConfigType', ['saturation_magnitude',
                                           'mag_rms_dependence_order',
                                           'mag_rms_outlier_threshold',
                                           'mag_rms_max_rej_iter',
                                           'max_rms',
                                           'faint_mag_limit',
                                           'min_observations_quantile',
                                           'sqrt_num_templates',
                                           'observation_id',
                                           'lc_fname_pattern',
                                           'fit_datasets',
                                           'fit_points_filter_variables',
                                           'fit_points_filter_expression',
                                           'selected_plots'])

    configuration = ConfigType(
        saturation_magnitude=8.2,
        mag_rms_dependence_order=2,
        mag_rms_outlier_threshold=6,
        mag_rms_max_rej_iter=20,
        max_rms=0.15,
        faint_mag_limit=12.0,
        min_observations_quantile=0.7,
        sqrt_num_templates=32,
        observation_id=('fitsheader.cfg.stid',
                        'fitsheader.cfg.cmpos',
                        'fitsheader.fnum'),
        lc_fname_pattern=join_paths(data_dir, '%d-%d-%d.hdf5'),
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
        fit_points_filter_variables=dict(),
        fit_points_filter_expression=None,
        selected_plots='tfa_templates_%(plot_id)s_%(phot_index)03d.pdf'
    )

    epd_statistics = load_epd_statistics(
        join_paths(data_dir, 'epd_statistics.txt')
    )

    TFA(epd_statistics, configuration)
