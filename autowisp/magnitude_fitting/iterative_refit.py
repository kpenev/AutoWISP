"""Interface for performing iterative magnitude fitting."""

import logging
import os
from functools import partial
from tempfile import TemporaryDirectory

import numpy
from astropy.io import fits

from autowisp.error_context import run_pool
from autowisp.exceptions import FileKind, FitMagnitudesError, RelatedFile
from autowisp.data_reduction.data_reduction_file import DataReductionFile
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


def _magfit_related_files(dr_fname, single_photref=None, master_photref=None):
    """``related_files`` classifier for a magnitude-fit work item.

    The item is the DR file being fit; the batch is fit against the single
    photometric reference (and, once it exists, the master photometric
    reference). Module-level so a ``partial`` binding the references is
    picklable to the workers.
    """

    related = [RelatedFile(FileKind.DR_FILE, dr_fname, role="input")]
    if single_photref:
        related.append(
            RelatedFile(FileKind.DR_FILE, single_photref, role="single_photref")
        )
    if master_photref:
        related.append(
            RelatedFile(
                FileKind.MASTER_PHOTREF, master_photref, role="master_photref"
            )
        )
    return related


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
        mark_start=partial(
            mark_start, status=2 * path_substitutions["magfit_iteration"]
        ),
        mark_end=partial(
            mark_end,
            status=2 * path_substitutions["magfit_iteration"] + 1,
            final=configuration.master_photref_fname is not None,
        ),
        **path_substitutions,
    )

    if configuration.num_parallel_processes > 1:
        run_pool(
            pool_magfit,
            fit_dr_filenames,
            config=vars(configuration),
            num_processes=configuration.num_parallel_processes,
            stream_consumer=(
                None
                if magfit_stat_collector is None
                else magfit_stat_collector.add_input
            ),
            related_files=partial(
                _magfit_related_files,
                single_photref=getattr(
                    configuration, "single_photref_dr_fname", None
                ),
                master_photref=configuration.master_photref_fname,
            ),
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
        str or None:
            The filename of the master photometric reference the last pass
            was fit against, or None if every pass was against the single
            photometric reference.

        str:
            The filename of the statistics of the last pass.
    """

    def update_photref(
        *,
        magfit_stat_collector,
        old_reference,
        num_photometries,
        fname_substitutions,
        sphotref_header,
    ):
        """
        Return the reference for the next pass, or None if this pass was last.

        The master built from the pass just completed is saved only if another
        pass is fit against it: once it agrees with the reference that pass
        used, or the iterations run out, that reference is final and the new
        master, used for nothing, is discarded. A pass against the single
        photometric reference is never the last, so the final master is
        always one that images were fit against.

        Args:
            magfit_stat_collector(MasterPhotrefCollector):    The object used by
                the magnitude fitting processes to generate the magnitude
                fitting statistics.

            old_reference(dict):    The photometric reference used for the last
                magnitude fitting iteration.

            num_photometries(int):    How many different photometric
                measurements are being fit.

        Returns:
            dict or None:
                The reference to fit the next pass against, or None if there
                is no next pass.

            str or None:
                The file the returned reference was saved to, or None if there
                is no next pass.
        """

        logger = logging.getLogger(__name__)
        master_reference_fname = (
            configuration.master_photref_fname_format.format_map(
                fname_substitutions
            )
        )
        with TemporaryDirectory(
            dir=os.path.dirname(os.path.abspath(master_reference_fname))
        ) as candidate_dir:
            candidate_fname = os.path.join(
                candidate_dir, os.path.basename(master_reference_fname)
            )
            try:
                magfit_stat_collector.generate_master(
                    master_reference_fname=candidate_fname,
                    catalog=catalog,
                    fit_terms_expression=(
                        configuration.mphotref_scatter_fit_terms
                    ),
                    extra_header=sphotref_header,
                )
            # Catch only the master-photref generation failure, so an
            # unrelated error inside generate_master surfaces instead of
            # being swallowed.
            except FitMagnitudesError:
                return None, None
            new_reference = get_master_photref(candidate_fname)

            if (
                fname_substitutions["magfit_iteration"]
                >= configuration.max_magfit_iterations
            ):
                return None, None
            if fname_substitutions["magfit_iteration"] > 0 and _converged(
                old_reference, new_reference, num_photometries, logger
            ):
                return None, None
            os.replace(candidate_fname, master_reference_fname)
        return new_reference, master_reference_fname

    def _converged(old_reference, new_reference, num_photometries, logger):
        """True iff the references agree within ``max_photref_change``."""

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

        return average_square_change.max() <= configuration.max_photref_change

    path_substitutions["magfit_iteration"] = (
        configuration.continue_from_iteration - 1
    )

    with DataReductionFile(single_photref_dr_fname, "r") as photref_dr:
        sphotref_header = photref_dr.get_frame_header()
        fname_substitutions = dict(sphotref_header)
        fname_substitutions.update(path_substitutions)
        if configuration.continue_from_iteration > 0:
            master_reference_fname = (
                configuration.master_photref_fname_format.format_map(
                    fname_substitutions
                )
            )
            photref = get_master_photref(master_reference_fname)
            photref_fname = master_reference_fname
        else:
            photref = get_single_photref(photref_dr, **path_substitutions)
            photref_fname = None

    catalog = format_master_catalog(
        catalog_sources, photref_dr.parse_hat_source_id
    )

    num_photometries = next(iter(photref.values()))["mag"].size

    sphotref_header["IMAGETYP"] = "mphotref"
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
            source_name_format=configuration.source_name_format,
            tempstore_dir=configuration.tempstore_dir,
            outlier_threshold=configuration.stat_rej_level,
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

        photref, next_photref_fname = update_photref(
            magfit_stat_collector=magfit_stat_collector,
            old_reference=photref,
            num_photometries=num_photometries,
            fname_substitutions=fname_substitutions,
            sphotref_header=sphotref_header,
        )
        if photref is not None:
            photref_fname = next_photref_fname
        mark_start = partial(mark_end, final=False)
    for fit_dr_fname in fit_dr_filenames:
        mark_end(
            fit_dr_fname,
            status=2 * path_substitutions["magfit_iteration"] - 1,
            final=True,
        )
    return photref_fname, stat_fname


# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals
