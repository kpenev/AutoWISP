"""Interface for parallel EPD correction of many LCs."""

from multiprocessing import Pool
import logging

from .epd_correction import EPDCorrection

def parallel_epd(lc_fnames, num_parallel_processes, **epd_config):
    """
    Correct LCs running EPD in parallel.

    Args:
        lc_fnames([str]):    The filenames of the light curves to correct.

        num_parallel_processes(int):    The maximum number of parallel processes
            to use.

        epd_config:    Passed directly to EPDCorrection.__init__().

    Returns:
        [numpy.array]:
            The return values of EPDCorrection.__call__() in the same order as
            lc_fnames.
    """

    logger = logging.getLogger(__name__)

    #epd_config is expected to contain all requiered arguments.
    #pylint: disable=missing-kwoa
    correct = EPDCorrection(**epd_config, fit_identifier='EPD')
    #pylint: enable=missing-kwoa

    logger.info('Starting EPD for %d frames.', len(lc_fnames))

    if num_parallel_processes == 1:
        return [correct(lcf) for lcf in lc_fnames]

    with Pool(num_parallel_processes) as epd_pool:
        return epd_pool.map(correct, lc_fnames)
