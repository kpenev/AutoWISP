"""Functions for detrending light curves (EPD or TFA)."""

from os import path, makedirs
import logging

import numpy
from pytransit import QuadraticModel

from superphot_pipeline import DataReductionFile
from superphot_pipeline import LightCurveFile
from superphot_pipeline.magnitude_fitting.util import read_master_catalogue
from superphot_pipeline.light_curves.apply_correction import\
    apply_parallel_correction,\
    apply_reconstructive_correction_transit,\
    save_correction_statistics,\
    recalculate_correction_statistics

def extract_target_lc(lc_fnames, target_id):
    """Return target LC fname, & LC fname list with the target LC removed."""

    for index, fname in enumerate(lc_fnames):
        if path.basename(fname).startswith(target_id):
            return lc_fnames.pop(index), lc_fnames
    raise ValueError('None of the lightcurves seems to be for the target.')


def get_hat_source_id(lc_fname):
    """Return parsed to 3-int HAT source id for the given LC file."""

    if not hasattr(get_hat_source_id, 'parse_hat_source_id'):
        get_hat_source_id.parse_hat_source_id = (
            DataReductionFile().parse_hat_source_id
        )

    return get_hat_source_id.parse_hat_source_id(
        dict(LightCurveFile(lc_fname, 'r')['Identifiers'])[b'HAT']
    )



def add_catalogue_info(lc_fnames, catalogue_fname, magnitude_column, result):
    """Fill the catalogue information fields in result."""

    catalogue = read_master_catalogue(catalogue_fname,
                                      DataReductionFile().parse_hat_source_id)

    for lc_ind, fname in enumerate(lc_fnames):
        source_id = get_hat_source_id(fname)
        result[lc_ind]['ID'] = source_id
        cat_info = catalogue[source_id]
        result[lc_ind]['mag'] = cat_info[magnitude_column]
        result[lc_ind]['xi'] = cat_info['xi']
        result[lc_ind]['eta'] = cat_info['eta']

def correct_target_lc(target_lc_fname, configuration, correct):
    """Perform reconstructive detrending on the target LC."""

    num_limbdark_coef = len(configuration['limb_darkening'])
    assert num_limbdark_coef == 2

    transit_parameters = (
        [configuration['radius_ratio']]
        +
        list(configuration['limb_darkening'])
        +
        [
            configuration['mid_transit'],
            configuration['period'],
            configuration['scaled_semimajor'],
            configuration['inclination'] * numpy.pi / 180.0
        ]
    )
    if hasattr(configuration, 'eccentricity'):
        transit_parameters.append(configuration['eccentricity'])
    if hasattr(configuration, 'periastron'):
        transit_parameters.append(configuration['periastron'])

    fit_parameter_flags = numpy.zeros(len(transit_parameters), dtype=bool)

    param_indices = dict(depth=0,
                         limbdark=list(
                             range(1, num_limbdark_coef + 1)
                         ),
                         mid_transit=num_limbdark_coef + 1,
                         period=num_limbdark_coef + 2,
                         semimajor=num_limbdark_coef + 3,
                         inclination=num_limbdark_coef + 4,
                         eccentricity=num_limbdark_coef + 5,
                         periastron=num_limbdark_coef + 6)
    for to_fit in configuration['mutable_transit_params']:
        fit_parameter_flags[param_indices[to_fit]] = True

    return apply_reconstructive_correction_transit(
        target_lc_fname,
        correct,
        transit_model=QuadraticModel(),
        transit_parameters=numpy.array(transit_parameters),
        fit_parameter_flags=fit_parameter_flags,
        num_limbdark_coef=num_limbdark_coef
    )


def recalculate_detrending_performance(lc_fnames,
                                       catalogue_fname,
                                       magnitude_column,
                                       output_statistics_fname,
                                       **recalc_arguments):
    """
    Re-create a statistics file after de-trending directly from LCs.

    Args:
        lc_fnames:    Iterable over the filenames of the de-trended lightcurves
            to rederive the statistics for.

        catalogue_fname:     The filename of the catalogue to add information to
            the statistics.

        magnitude_column:     The column from the catalogue to use as brightness
            indicator in the statistics file.

        output_statistics_fname:    The filename to save the statistics under.

        recalc_arguments:    Passed directly to
            recalculate_correction_statistics()
    """

    statistics = recalculate_correction_statistics(lc_fnames,
                                                   **recalc_arguments)
    add_catalogue_info(lc_fnames, catalogue_fname, magnitude_column, statistics)

    if not path.exists(path.dirname(output_statistics_fname)):
        makedirs(path.dirname(output_statistics_fname))
    save_correction_statistics(statistics, output_statistics_fname)


def detrend_light_curves(lc_collection,
                         configuration,
                         correct,
                         output_statistics_fname):
    """Detrend all lightcurves and create statistics file."""

    lc_collection = list(lc_collection)
    if configuration['target_id'] is not None:
        target_lc_fname, lc_fnames = extract_target_lc(
            lc_collection,
            configuration['target_id']
        )

        _, target_result = correct_target_lc(
            target_lc_fname,
            configuration,
            correct
        )
    else:
        lc_fnames = lc_collection

    if lc_fnames:
        result = apply_parallel_correction(
            lc_fnames,
            correct,
            configuration['num_parallel_processes']
        )
        if configuration['target_id'] is not None:
            result = numpy.concatenate((result, target_result))
    else:
        result = target_result

    if configuration['target_id'] is not None:
        lc_fnames.append(target_lc_fname)

    if configuration['detrending_catalogue'] is not None:
        recalculate_detrending_performance(
            lc_fnames,
            fit_datasets=configuration['detrend_datasets'],
            catalogue_fname=configuration['detrending_catalogue'],
            magnitude_column=configuration['magnitude_column'],
            output_statistics_fname=output_statistics_fname,
            calculate_average=getattr(numpy,
                                      configuration['detrend_reference_avg']),
            calculate_scatter=getattr(numpy,
                                      configuration['detrend_error_avg']),
            outlier_threshold=configuration['detrend_rej_level'],
            max_outlier_rejections=configuration['detrend_max_rej_iter']
        )

        logging.info('Generated statistics file: %s.',
                     repr(output_statistics_fname))
