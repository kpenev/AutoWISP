"""Interface for parallel EPD correction of many LCs."""

from multiprocessing import Pool
import logging

import numpy
from scipy.optimize import minimize
import pandas

from superphot_pipeline import DataReductionFile
from superphot_pipeline.light_curves.epd_correction import EPDCorrection
from superphot_pipeline.light_curves.reconstructive_epd_transit import\
    ReconstructiveEPDTransit
from superphot_pipeline.database.interface import db_engine

def save_statistics(epd_statistics, filename):
    """Save the given statistics (result of parallel_epd) to the given file."""

    print('EPD statistics:\n' + repr(epd_statistics))
    mem_dr = DataReductionFile()
    dframe = pandas.DataFrame(
        {column: epd_statistics[column] for column in ['mag', 'xi', 'eta']},
    )

    dframe.insert(
        0,
        '2MASSID',
        [
            mem_dr.get_hat_source_id_str(int_id)
            for int_id in epd_statistics['ID']
        ]
    )

    num_photometries = epd_statistics['rms'][0].size

    for prefix in ['rms', 'num_finite']:
        for phot_index in range(num_photometries):
            dframe[prefix + '_%02d' % phot_index] = (
                epd_statistics[prefix][:, phot_index]
            )

    with open(filename, 'w') as outf:
        dframe.to_string(outf, col_space=25, index=False, justify='left')

def load_statistics(filename):
    """Read a previously stored statistics from a file."""

    mem_dr = DataReductionFile()
    dframe = pandas.read_csv(filename, delim_whitespace=True)

    num_sources, num_photometries = dframe.shape
    num_photometries = (num_photometries - 4) // 2

    result = numpy.empty(num_sources,
                         dtype=EPDCorrection.get_result_dtype(num_photometries))
    for column in ['mag', 'xi', 'eta']:
        result[column] = dframe[column]

    for prefix in ['rms', 'num_finite']:
        for phot_index in range(num_photometries):
            result[prefix][:, phot_index] = (
                dframe[prefix + '_%02d' % phot_index]
            )

    for index, source_id in enumerate(dframe['2MASSID']):
        result['ID'][index] = mem_dr.parse_hat_source_id(source_id)

    return result

def parallel_epd(lc_fnames,
                 num_parallel_processes,
                 **epd_config):
    """
    Correct LCs running EPD in parallel.

    Args:
        lc_fnames([str]):    The filenames of the light curves to correct.

        num_parallel_processes(int):    The maximum number of parallel processes
            to use.

        statistics_fname(str):    Filename to use for saving the statistics.

        epd_config:    Passed directly to EPDCorrection.__init__().

    Returns:
        numpy.array:
            The return values of EPDCorrection.__call__() in the same order as
            lc_fnames.
    """

    logger = logging.getLogger(__name__)

    #epd_config is expected to contain all requiered arguments.
    #pylint: disable=missing-kwoa
    correct = EPDCorrection(**epd_config, fit_identifier='EPD')
    #pylint: enable=missing-kwoa

    logger.info('Starting EPD for %d light curves.', len(lc_fnames))

    if num_parallel_processes == 1:
        result = numpy.concatenate([correct(lcf) for lcf in lc_fnames])

    with Pool(num_parallel_processes, db_engine.dispose()) as epd_pool:
        result = numpy.concatenate(epd_pool.map(correct, lc_fnames))

    logger.info('Finished EPD.')

    return result

def reconstructive_epd_transit(lc_fname,
                               transit_model,
                               transit_parameters,
                               fit_parameter_flags,
                               num_limbdark_coef,
                               **epd_config):
    """
    Perform a reconstructive EPD on a lightcurve assuming it contains a transit.

    The corrected lightcurve, preserving the best-fit transit is saved in the
    lightcurve just like for non-reconstructive EPD.

    Args:
        transit_model:    Object which supports the transit model intefrace of
            pytransit.

        transit_parameters(scipy float array):    The full array of parameters
            required by the transit model's evaluate() method.

        fit_parameter_flags(scipy bool array):    Flags indicating parameters
            whose values should be fit for (by having a corresponding entry of
            True). Must match exactly the shape of transit_parameters.

        num_limbdark_coef(int):    How many of the transit parameters are limb
            darkening coefficinets? Those need to be passed to the model
            separately.

        epd_config:    Configuration of how to carry out the EPD, once the
            transit model is removed from the lightcurve.

    Returns:
        (scipy array, scipy array):
            * The best fit transit parameters

            * The return value of ReconstructiveEPDTransit.__call__() for the
              best-fit transit parameters.
    """

    #This is intended to server as a callable.
    #pylint: disable=too-few-public-methods
    class MinimizeFunction:
        """Suitable callable for scipy.optimize.minimize()."""

        def __init__(self):
            """Create the EPD object."""

            self.epd = ReconstructiveEPDTransit(
                transit_model,
                fit_amplitude=False,
                fit_identifier='Reconstructive EPD',
                **epd_config
            )
            self.transit_parameters = numpy.copy(transit_parameters)

        def __call__(self, fit_params):
            """
            Return the RMS residual of the EPD after removing a transit model.

            Args:
                fit_params(scipy array):    The values of the mutable model
                    parameters for the current minimization function evaluation.

            Returns:
                float:
                    RMS of the residuals after EPD correctiong around the
                    transit model with the given parameters.
            """

            self.transit_parameters[fit_parameter_flags] = fit_params
            return self.epd(lc_fname,
                            self.transit_parameters[0],
                            self.transit_parameters[1 : num_limbdark_coef + 1],
                            *self.transit_parameters[num_limbdark_coef + 1 : ],
                            save=False)['rms']
    #pylint: enable=too-few-public-methods

    rms_function = MinimizeFunction()
    best_fit_transit = numpy.copy(transit_parameters)

    if fit_parameter_flags.any():
        minimize_result = minimize(rms_function,
                                   transit_parameters[fit_parameter_flags])
        assert minimize_result.success
        best_fit_transit[fit_parameter_flags] = minimize_result.x

    return (
        best_fit_transit,
        rms_function.epd(lc_fname,
                         best_fit_transit[0],
                         best_fit_transit[1: num_limbdark_coef + 1],
                         *best_fit_transit[num_limbdark_coef + 1 : ])
    )
