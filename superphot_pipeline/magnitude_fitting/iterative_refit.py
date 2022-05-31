"""Interface for performing iterative magnitude fitting."""

from tempfile import TemporaryDirectory
import functools
from multiprocessing import Pool, Lock
import logging

import numpy
from superphot_pipeline import DataReductionFile

from superphot_pipeline.magnitude_fitting import\
    LinearMagnitudeFit,\
    MasterPhotrefCollector
from superphot_pipeline.magnitude_fitting.util import\
    get_single_photref,\
    get_master_photref,\
    read_master_catalogue

_logger = logging.getLogger(__name__)

#Could not come up with a sensible way to simplify
#pylint: disable=too-many-locals
def iterative_refit(fit_dr_filenames,
                    *,
                    single_photref_dr_fname,
                    master_catalogue_fname,
                    configuration,
                    master_photref_fname_format,
                    magfit_stat_fname_format,
                    master_scatter_fit_terms,
                    max_iterations=5,
                    **path_substitutions):
    """
    Iteratively performa magnitude fitting/generating master until convergence.

    Args:
        fit_dr_filenames(str iterable):    A list of the data reduction files to
            fit.

        single_photref_dr_fname(str):    The name of the data reduction file of
            the single photometric reference to use to start the magnitude
            fitting iterations.

        master_catalogue_fname(str):    The name of the catalogue file to use as
            extra information in magnitude fitting terms and for excluding
            sources from the fit.

        configuration:    Passed directly as the config argument to
            LinearMagnitudeFit.__init__() but it must also contain the following
            attributes:

                * num_parallel_processes(int): the the maximum number of
                  magnitude fitting parallel processes to use.

                * max_photref_change(float): the maximum square average change
                  of photometric reference magnitudes to consider the iterations
                  converged.

        master_photref_fname_format(str):    A format string involving
            a {magfit_iteration} substitution along with any variables from the
            header of the single photometric reference or passed through the
            path_substitutions arguments, that expands to the name of the file
            to save the master photometric reference for a particular iteration.

        magfit_stat_fname_format(str):    Similar to
            ``master_photref_fname_format``, but defines the name to use for
            saving the statistics of a magnitude fitting iteration.

        num_parallel_processes(int):     How many processes to use for
            simultaneus fitting.

        master_scatter_fit_terms(str):    Terms to include in the fit for the
            scatter when deciding which stars to include in the master.

        max_iterations(int):    The maximum number of iterations of deriving a
            master and re-fitting to allow.

        path_substitutions:     Any variables to substitute in
            ``master_photref_fname_format`` or to pass to data reduction files
            to identify components to use in the fit.

    Returns:
        None
    """

    def update_photref(magfit_stat_collector,
                       old_reference,
                       source_id_parser,
                       num_photometries,
                       single_photref_header):
        """
        Return the next iteration photometric reference or None if converged.

        Args:
            magfit_stat_collector(MasterPhotrefCollector):    The object used by
                the magnitude fitting processes to generate the magnitude
                fitting statistics.

            old_reference(dict):    The photometric reference used for the last
                magnitude fitting iteration.

            source_id_parser(callable):    Should return the integers
                identifying a source, given its string ID.

            num_photometries(int):    How many different photometric
                measurements are being fit.
        """

        master_reference_fname = master_photref_fname_format.format(
            **single_photref_header,
            **path_substitutions
        )
        magfit_stat_collector.generate_master(
            master_reference_fname=master_reference_fname,
            catalogue=catalogue,
            fit_terms_expression=master_scatter_fit_terms,
            parse_source_id=source_id_parser
        )
        new_reference = get_master_photref(master_reference_fname)

        common_sources = set(new_reference) & set(old_reference)

        average_square_change = numpy.zeros(num_photometries,
                                            dtype=numpy.float64)
        num_finite = numpy.zeros(num_photometries, dtype=numpy.float64)
        for source in common_sources:
            square_diff = (old_reference[source]['mag'][0]
                           -
                           new_reference[source]['mag'][0])**2
            #False positive
            #pylint: disable=assignment-from-no-return
            finite_entries = numpy.isfinite(square_diff)
            #pylint: enable=assignment-from-no-return
            print('Num photometries: ' + repr(num_photometries))
            print('square_diff (shape=%s): ' % repr(square_diff.shape)
                  +
                  repr(square_diff))
            print('finite_entries (shape=%s): ' % repr(finite_entries.shape)
                  +
                  repr(finite_entries))
            print('average_square_change (shape=%s): '
                  %
                  repr(average_square_change.shape)
                  +
                  repr(average_square_change))

            average_square_change[finite_entries] += square_diff[finite_entries]
            num_finite += finite_entries

        average_square_change /= num_finite
        _logger.debug(
            'Fit iteration resulted in average square change in magnitudes of: '
            '%s',
            repr(average_square_change)
        )

        if average_square_change.max() <= configuration.max_photref_change:
            return None

        return new_reference

    with DataReductionFile(single_photref_dr_fname, 'r') as photref_dr:
        photref = get_single_photref(photref_dr, **path_substitutions)
        single_photref_header = photref_dr.get_frame_header()

    catalogue = read_master_catalogue(master_catalogue_fname,
                                      photref_dr.parse_hat_source_id)
    path_substitutions['magfit_iteration'] = 0

    num_photometries = next(iter(photref.values()))['mag'].size

    with TemporaryDirectory() as grcollect_tmp_dir:
        while (
                photref
                and
                path_substitutions['magfit_iteration'] <= max_iterations
        ):
            assert next(iter(photref.values()))['mag'].size == num_photometries

            magfit_stat_collector = MasterPhotrefCollector(
                magfit_stat_fname_format.format(
                    **single_photref_header,
                    **path_substitutions
                ),
                num_photometries,
                grcollect_tmp_dir,
                output_lock=(
                    Lock() if configuration.num_parallel_processes > 1
                    else None
                )
            )
            magfit = LinearMagnitudeFit(config=configuration,
                                        reference=photref,
                                        master_catalogue=catalogue,
                                        magfit_collector=magfit_stat_collector)
            if configuration.num_parallel_processes > 1:
                pool_magfit = functools.partial(magfit, **path_substitutions)
                with Pool(configuration.num_parallel_processes) as magfit_pool:
                    magfit_pool.map(pool_magfit, fit_dr_filenames)
            else:
                for dr_fname in fit_dr_filenames:
                    magfit(dr_fname, **path_substitutions)

            photref = update_photref(magfit_stat_collector,
                                     photref,
                                     photref_dr.parse_hat_source_id,
                                     num_photometries,
                                     single_photref_header)
            path_substitutions['magfit_iteration'] += 1
#pylint: enable=too-many-locals
