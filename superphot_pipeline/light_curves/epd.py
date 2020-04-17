"""Interface for applying EPD corrections to light curves."""

from multiprocessing import Pool
import logging

import numpy
from scipy.optimize import minimize
import pandas

from superphot_pipeline import DataReductionFile
from superphot_pipeline.light_curves.epd_correction import EPDCorrection
from superphot_pipeline.light_curves.reconstructive_correction_transit import\
    ReconstructiveCorrectionTransit

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

    with Pool(num_parallel_processes) as epd_pool:
        result = numpy.concatenate(epd_pool.map(correct, lc_fnames))

    logger.info('Finished EPD.')

    return result
