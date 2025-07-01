"""Interface for performing iterative magnitude fitting."""

from tempfile import TemporaryDirectory
from multiprocessing import Pool
import logging
from functools import partial
from os import getpid

import numpy
from astropy.io import fits

from autowisp.multiprocessing_util import setup_process_map
from autowisp import DataReductionFile
from autowisp.fits_utilities import update_stack_header
from autowisp.magnitude_fitting import (
    LinearMagnitudeFit,
    MasterPhotrefCollector,
)
from autowisp.magnitude_fitting.util import (
    get_single_photref,
    get_master_photref,
    format_master_catalog,
)


def _get_common_header(fit_dr_filenames):
    """Return header containing all keywords common to all input frames."""

    result = fits.Header()
    first = True
    for dr_fname in fit_dr_filenames:
        with DataReductionFile(dr_fname, "r") as data_reduction:
            update_stack_header(
                result, data_reduction.get_frame_header(), dr_fname, first
            )
            first = False
    return result


# Could not come up with a sensible way to simplify
# pylint: disable=too-many-arguments
def single_iteration(
    fit_dr_filenames,
    *,
    photref,
    configuration,
    path_substitutions,
    mark_start,
    mark_end,
    magfit_stat_collector=None,
):
    """Do a single magfit iteration using parallel processes."""

    magfit = LinearMagnitudeFit(
        config=configuration,
        reference=photref,
        source_name_format=configuration.source_name_format,
    )

    pool_magfit = partial(
        magfit,
        mark_start=(
            partial(
                mark_start, status=2 * path_substitutions["magfit_iteration"]
            )
            if (
                path_substitutions["magfit_iteration"] == 0
                or magfit_stat_collector is None
            )
            else partial(
                mark_end,
                status=2 * path_substitutions["magfit_iteration"],
                final=False,
            )
        ),
        mark_end=partial(
            mark_end,
            status=2 * path_substitutions["magfit_iteration"] + 1,
            final=configuration.master_photref_fname is not None,
        ),
        **path_substitutions,
    )

    if configuration.num_parallel_processes > 1:
        configuration.parent_pid = getpid()
        with Pool(
            configuration.num_parallel_processes,
            initializer=setup_process_map,
            initargs=(vars(configuration),),
        ) as magfit_pool:
            if magfit_stat_collector is None:
                magfit_pool.map(pool_magfit, fit_dr_filenames)
            else:
                magfit_stat_collector.add_input(
                    magfit_pool.imap_unordered(pool_magfit, fit_dr_filenames)
                )
    elif magfit_stat_collector is None:
        for dr_fname in fit_dr_filenames:
            pool_magfit(dr_fname)
    else:
        magfit_stat_collector.add_input(map(pool_magfit, fit_dr_filenames))


# pylint: enable=too-many-arguments


# Could not come up with a sensible way to simplify
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def iterative_refit(
    fit_dr_filenames,
    *,
    single_photref_dr_fname,
    catalog_sources,
    configuration,
    mark_start,
    mark_end,
    path_substitutions,
):
    """
    Iteratively performa magnitude fitting/generating master until convergence.

    Args:
        fit_dr_filenames(str iterable):    A list of the data reduction files to
            fit.

        single_photref_dr_fname(str):    The name of the data reduction file of
            the single photometric reference to use to start the magnitude
            fitting iterations.

        catalog(pandas.DataFrame):    The the catalog to use as extra
            information in magnitude fitting terms and for excluding sources
            from the fit.

        configuration:    Passed directly as the config argument to
            LinearMagnitudeFit.__init__() but it must also contain the following
            attributes:

                * num_parallel_processes(int): the the maximum number of
                  magnitude fitting parallel processes to use.

                * max_photref_change(float): the maximum square average change
                  of photometric reference magnitudes to consider the iterations
                  converged.

                * master_photref_fname_format(str): A format string involving a
                  {magfit_iteration} substitution along with any variables from
                  the header of the single photometric reference or passed
                  through the path_substitutions arguments, that expands to the
                  name of the file to save the master photometric reference for
                  a particular iteration.

                * magfit_stat_fname_format(str): Similar
                  to ``master_photref_fname_format``, but defines the name to
                  use for saving the statistics of a magnitude fitting
                  iteration.

                * num_parallel_processes(int): How many processes to use
                  for simultaneus fitting.

                * master_scatter_fit_terms(str): Terms to include in the fit
                  for the scatter when deciding which stars to include in the
                  master.

        mark_start(callable):    A function called at the start of each DR file
            fitting.

        mark_end(callable):    A function called after each DR file has finished
            fitting.

        max_iterations(int):    The maximum number of iterations of deriving a
            master and re-fitting to allow.

        path_substitutions(dict):     Any variables to substitute in
            ``master_photref_fname_format`` or to pass to data reduction files
            to identify components to use in the fit.

    Returns:
        The filename of the last master photometric reference created.
    """

    def update_photref(
        *,
        magfit_stat_collector,
        old_reference,
        source_id_parser,
        num_photometries,
        fname_substitutions,
        common_header,
    ):
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

        logger = logging.getLogger(__name__)
        master_reference_fname = (
            configuration.master_photref_fname_format.format_map(
                fname_substitutions
            )
        )
        try:
            magfit_stat_collector.generate_master(
                master_reference_fname=master_reference_fname,
                catalog=catalog,
                fit_terms_expression=configuration.mphotref_scatter_fit_terms,
                parse_source_id=source_id_parser,
                extra_header=common_header,
            )
        except RuntimeError:
            return None, None
        new_reference = get_master_photref(master_reference_fname)

        common_sources = set(new_reference) & set(old_reference)

        average_square_change = numpy.zeros(
            num_photometries, dtype=numpy.float64
        )
        num_finite = numpy.zeros(num_photometries, dtype=numpy.float64)
        for source in common_sources:
            square_diff = (
                old_reference[source]["mag"][0]
                - new_reference[source]["mag"][0]
            ) ** 2
            # False positive
            # pylint: disable=assignment-from-no-return
            finite_entries = numpy.isfinite(square_diff)
            # pylint: enable=assignment-from-no-return
            logger.debug("Num photometries: %s", repr(num_photometries))
            logger.debug(
                "square_diff (shape=%s): %s",
                repr(square_diff.shape),
                repr(square_diff),
            )
            logger.debug(
                "finite_entries (shape=%s): %s",
                repr(finite_entries.shape),
                repr(finite_entries),
            )
            logger.debug(
                "average_square_change (shape=%s): %s",
                repr(average_square_change.shape),
                repr(average_square_change),
            )

            average_square_change[finite_entries] += square_diff[finite_entries]
            num_finite += finite_entries

        average_square_change /= num_finite
        logger.debug(
            "Fit iteration resulted in average square change in magnitudes of: "
            "%s",
            repr(average_square_change),
        )

        if average_square_change.max() <= configuration.max_photref_change:
            return None, master_reference_fname

        return new_reference, master_reference_fname

    path_substitutions["magfit_iteration"] = (
        configuration.continue_from_iteration - 1
    )

    with DataReductionFile(single_photref_dr_fname, "r") as photref_dr:
        common_header = photref_dr.get_frame_header()
        fname_substitutions = dict(common_header)
        fname_substitutions.update(path_substitutions)
        if configuration.continue_from_iteration > 0:
            master_reference_fname = (
                configuration.master_photref_fname_format.format_map(
                    fname_substitutions
                )
            )
            photref = get_master_photref(master_reference_fname)
        else:
            photref = get_single_photref(photref_dr, **path_substitutions)

    catalog = format_master_catalog(
        catalog_sources, photref_dr.parse_hat_source_id
    )

    num_photometries = next(iter(photref.values()))["mag"].size

    photref_fname = None
    common_header["IMAGETYP"] = "mphotref"
    with TemporaryDirectory() as mphotref_collect_tmp_dir:
        while (
            photref
            and path_substitutions["magfit_iteration"]
            < configuration.max_magfit_iterations
        ):
            path_substitutions["magfit_iteration"] += 1
            fname_substitutions["magfit_iteration"] += 1

            assert next(iter(photref.values()))["mag"].size == num_photometries

            stat_fname = configuration.magfit_stat_fname_format.format_map(
                fname_substitutions
            )

            magfit_stat_collector = MasterPhotrefCollector(
                stat_fname,
                num_photometries,
                len(fit_dr_filenames),
                mphotref_collect_tmp_dir,
                source_name_format=configuration.source_name_format,
            )

            single_iteration(
                fit_dr_filenames,
                photref=photref,
                configuration=configuration,
                path_substitutions=path_substitutions,
                mark_start=mark_start,
                mark_end=mark_end,
                magfit_stat_collector=magfit_stat_collector,
            )

            photref, photref_fname = update_photref(
                magfit_stat_collector=magfit_stat_collector,
                old_reference=photref,
                source_id_parser=photref_dr.parse_hat_source_id,
                num_photometries=num_photometries,
                fname_substitutions=fname_substitutions,
                common_header=common_header,
            )
    for fit_dr_fname in fit_dr_filenames:
        mark_end(
            fit_dr_fname,
            status=2 * path_substitutions["magfit_iteration"] - 1,
            final=True,
        )
    return photref_fname, stat_fname


# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals
