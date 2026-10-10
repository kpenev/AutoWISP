"""Interface for performing iterative magnitude fitting."""

import logging
import os
from functools import partial
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy
from astropy.io import fits

from autowisp.error_context import run_pool
from autowisp.exceptions import (
    ConfigurationError,
    FileKind,
    FitMagnitudesError,
    RelatedFile,
)
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
    get_path_substitutions,
    read_exclusions,
)
from autowisp.project_paths import fill_path_template, resolve_path


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


def _get_population_record(exclusion_rule, dr_filenames, excluded_dr_filenames):
    """
    Return the header keywords and table recording what masters are built from.

    Args:
        exclusion_rule(str):    The rule that decided the exclusions, empty if
            they were not decided by a rule.

        dr_filenames([str]):    Every DR file fit while the masters are built,
            excluded ones included, as given to magnitude fitting.

        excluded_dr_filenames([str]):    The DR files among ``dr_filenames``
            fit but left out of the masters.

    Returns:
        dict:
            The ``QCRULE``, ``QCNIMG`` and ``QCNEXCL`` header keywords, as
            ``(value, comment)`` tuples.

        fits.BinTableHDU:
            The ``INMASTER`` table listing every DR file fit, with an
            ``included`` column flagging those the masters are built from.
    """

    header = {
        "QCRULE": (exclusion_rule, "Rule excluding images from the master"),
        "QCNIMG": (len(dr_filenames), "Images fit, excluded included"),
        "QCNEXCL": (len(excluded_dr_filenames), "Images excluded from master"),
    }
    excluded = set(excluded_dr_filenames)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="dr_fname",
                format=f"{max(len(fname) for fname in dr_filenames)}A",
                array=numpy.array(dr_filenames, dtype=str),
            ),
            fits.Column(
                name="included",
                format="L",
                array=numpy.array(
                    [fname not in excluded for fname in dr_filenames],
                    dtype=bool,
                ),
            ),
        ],
        name="INMASTER",
    )
    return header, table


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


# A callable: __call__() is the whole interface.
# pylint: disable=too-few-public-methods
class MagnitudeFitting:
    """
    Magnitude fitting of one batch of DR files.

    With an existing master photometric reference configured, every image is
    fit once against it. Otherwise the images are fit and the master
    re-derived from them, pass after pass, until it converges; images on the
    exclusion list are fit too, but kept out of the masters.

    Attributes:
        sphotref_header(fits.Header):    The header of the single photometric
            reference.
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, configuration, *, start_status, mark_start, mark_end):
        """
        Get ready to fit a batch of DR files.

        Args:
            configuration(dict):    The configuration of the fit_magnitudes
                step.

            start_status(int or None):    The status an interrupted run left
                the batch at, or None to start from scratch.

            mark_start(callable):    Called at the start of fitting each DR
                file.

            mark_end(callable):    Called after each DR file has been fit.
        """

        if start_status is None:
            start_status = -1
        else:
            assert start_status % 2 == 1, (
                f"Magnitude fitting recorded an even status {start_status}; "
                "only odd ones mark a completed iteration it could resume "
                "from!"
            )
        assert (
            configuration["master_photref_fname"] is None or start_status == -1
        ), (
            f"Magnitude fitting was asked to resume from {start_status} "
            "against an existing master photometric reference, which takes a "
            "single pass and so can only start from scratch!"
        )
        self._configuration = SimpleNamespace(
            **configuration,
            continue_from_iteration=(start_status + 1) // 2,
            source_name_format="{0:d}",
        )
        self._mark_start = mark_start
        self._mark_end = mark_end
        with DataReductionFile(
            configuration["single_photref_dr_fname"], "r"
        ) as sphotref_dr:
            self.sphotref_header = sphotref_dr.get_frame_header()
            self._parse_source_id = sphotref_dr.parse_hat_source_id
        self._path_substitutions = get_path_substitutions(
            configuration, self.sphotref_header
        )

    def __call__(self, dr_fnames, catalog_sources):
        """
        Fit the given DR files.

        Args:
            dr_fnames([str]):    The DR files to fit.

            catalog_sources(pandas.DataFrame):    The catalog to use as extra
                information in magnitude fitting terms and for excluding
                sources from the fit.

        Returns:
            [dict] or None:
                The new masters to record, as for
                ``ImageProcessingManager.add_masters()``, or None when fitting
                against an existing master, which creates none.
        """

        excluded = read_exclusions(self._configuration.qc_exclude_file)
        # Recorded in each DR file, even where no master is built.
        self._configuration.qc_excluded = excluded

        if self._configuration.master_photref_fname is not None:
            self._logger.info(
                "Fitting %d images against existing master photometric "
                "reference %s",
                len(dr_fnames),
                self._configuration.master_photref_fname,
            )
            self._fit_pass(
                dr_fnames,
                get_master_photref(self._configuration.master_photref_fname),
            )
            return None

        excluded_dr_fnames = [
            dr_fname
            for dr_fname in dr_fnames
            if os.path.realpath(dr_fname) in excluded
        ]
        fit_dr_fnames = [
            dr_fname
            for dr_fname in dr_fnames
            if os.path.realpath(dr_fname) not in excluded
        ]
        if not fit_dr_fnames:
            raise ConfigurationError(
                f"All {len(dr_fnames)} images are excluded from the master "
                "photometric reference, leaving nothing to build it from!"
            )

        population_header, population_table = _get_population_record(
            # Defined only when the pipeline configures the step.
            getattr(self._configuration, "magfit_exclusion_rule", None) or "",
            dr_fnames,
            excluded_dr_fnames,
        )
        master_inputs = SimpleNamespace(
            catalog=format_master_catalog(
                catalog_sources, self._parse_source_id
            ),
            header=self.sphotref_header.copy(),
            extra_hdus=[population_table],
        )
        master_inputs.header["IMAGETYP"] = "mphotref"
        master_inputs.header.update(population_header)

        self._logger.info(
            "Starting iterative magfit of %d images, %d of them excluded from "
            "the master, for single photref %s",
            len(dr_fnames),
            len(excluded_dr_fnames),
            self._configuration.single_photref_dr_fname,
        )
        master_fname, stat_fname = self._refit(
            fit_dr_fnames, excluded_dr_fnames, master_inputs
        )
        new_masters = [
            {
                "filename": stat_fname,
                "preference_order": None,
                "type": "magfit_stat",
            }
        ]
        # None when no pass was fit against a master, e.g. with
        # --max-magfit-iterations 0.
        if master_fname is not None:
            new_masters.append(
                {
                    "filename": master_fname,
                    "preference_order": None,
                    "type": "master_photref",
                }
            )
        return new_masters

    def _expand(self, fname_format):
        """Return the given file name format expanded for the current pass."""

        # dict() first: a header may repeat a keyword.
        return fill_path_template(
            fname_format,
            {**dict(self.sphotref_header), **self._path_substitutions},
        )

    def _refuse_clash(self):
        """
        Raise if a file this pass would write exists already.

        Only the pass itself writes its files, and cleaning up an interrupted
        run deletes the partial ones, so a file there can only belong to
        another master: one built from a single photometric reference the
        file name formats do not tell apart from this one, or one built
        earlier from this reference. Either way it is not overwritten.
        """

        for option in (
            "master_photref_fname_format",
            "magfit_stat_fname_format",
        ):
            fname = resolve_path(
                self._expand(getattr(self._configuration, option))
            )
            if os.path.exists(fname):
                raise ConfigurationError(
                    f"Magnitude fitting against single photometric reference "
                    f"{self._configuration.single_photref_dr_fname!r} would "
                    f"overwrite {fname!r}. If it belongs to another single "
                    "photometric reference, make --"
                    + option.replace("_", "-")
                    + " tell the two apart, e.g. by including {FNUM}. If it "
                    "is an earlier master of this reference, remove that "
                    "master's files to rebuild it: disabling it is not enough."
                )

    def _fit_pass(self, dr_fnames, photref, magfit_stat_collector=None):
        """Fit the given DR files once, against the given reference."""

        single_iteration(
            dr_fnames,
            photref=photref,
            configuration=self._configuration,
            path_substitutions=self._path_substitutions,
            mark_start=self._mark_start,
            mark_end=self._mark_end,
            magfit_stat_collector=magfit_stat_collector,
        )

    def _mark_pass(self, dr_fnames):
        """Record the current pass for DR files that are not fit in it."""

        iteration = self._path_substitutions["magfit_iteration"]
        for dr_fname in dr_fnames:
            self._mark_start(dr_fname, status=2 * iteration)
            self._mark_end(dr_fname, status=2 * iteration + 1, final=False)

    def _start_reference(self):
        """Return the first pass's reference, and the master it is, if any."""

        if self._configuration.continue_from_iteration > 0:
            master_fname = self._expand(
                self._configuration.master_photref_fname_format
            )
            return get_master_photref(master_fname), master_fname
        with DataReductionFile(
            self._configuration.single_photref_dr_fname, "r"
        ) as sphotref_dr:
            return (
                get_single_photref(sphotref_dr, **self._path_substitutions),
                None,
            )

    def _refit(self, fit_dr_fnames, excluded_dr_fnames, master_inputs):
        """
        Fit and re-derive the master photometric reference until it converges.

        Args:
            fit_dr_fnames([str]):    The DR files to fit and build the masters
                from.

            excluded_dr_fnames([str]):    DR files to fit but leave out of the
                masters. They are fit only once, against the reference of the
                last pass, which gives them exactly the fit they would get in
                that pass. Each pass still marks their progress along with the
                rest, so an interrupted run leaves the whole batch at one
                status.

            master_inputs:    What the masters are built with: ``catalog``,
                ``header`` and ``extra_hdus`` attributes, see
                MasterPhotrefCollector.generate_master().

        Returns:
            str or None:
                The filename of the master photometric reference the last pass
                was fit against, or None if every pass was against the single
                photometric reference.

            str:
                The filename of the statistics of the last pass.
        """

        self._path_substitutions["magfit_iteration"] = (
            self._configuration.continue_from_iteration - 1
        )
        photref, photref_fname = self._start_reference()
        num_photometries = next(iter(photref.values()))["mag"].size

        while (
            photref
            and self._path_substitutions["magfit_iteration"]
            < self._configuration.max_magfit_iterations
        ):
            self._path_substitutions["magfit_iteration"] += 1
            self._refuse_clash()
            assert next(iter(photref.values()))["mag"].size == num_photometries

            stat_fname = self._expand(
                self._configuration.magfit_stat_fname_format
            )
            magfit_stat_collector = MasterPhotrefCollector(
                stat_fname,
                num_photometries,
                len(fit_dr_fnames),
                source_name_format=self._configuration.source_name_format,
                tempstore_dir=self._configuration.tempstore_dir,
                outlier_threshold=self._configuration.stat_rej_level,
            )
            self._fit_pass(fit_dr_fnames, photref, magfit_stat_collector)
            self._mark_pass(excluded_dr_fnames)
            self._mark_start = partial(self._mark_end, final=False)

            next_photref, next_photref_fname = self._next_reference(
                magfit_stat_collector, photref, master_inputs
            )
            if next_photref is None:
                if excluded_dr_fnames:
                    self._fit_pass(excluded_dr_fnames, photref)
                break
            photref, photref_fname = next_photref, next_photref_fname

        for dr_fname in list(fit_dr_fnames) + list(excluded_dr_fnames):
            self._mark_end(
                dr_fname,
                status=2 * self._path_substitutions["magfit_iteration"] - 1,
                final=True,
            )
        return photref_fname, stat_fname

    def _next_reference(
        self, magfit_stat_collector, old_reference, master_inputs
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

            master_inputs:    See _refit().

        Returns:
            dict or None:
                The reference to fit the next pass against, or None if there
                is no next pass.

            str or None:
                The file the returned reference was saved to, or None if there
                is no next pass.
        """

        master_fname = self._expand(
            self._configuration.master_photref_fname_format
        )
        master_path = resolve_path(master_fname)
        with TemporaryDirectory(
            dir=os.path.dirname(os.path.abspath(master_path))
        ) as candidate_dir:
            candidate_fname = os.path.join(
                candidate_dir, os.path.basename(master_fname)
            )
            try:
                magfit_stat_collector.generate_master(
                    master_reference_fname=candidate_fname,
                    catalog=master_inputs.catalog,
                    fit_terms_expression=(
                        self._configuration.mphotref_scatter_fit_terms
                    ),
                    extra_header=master_inputs.header,
                    extra_hdus=master_inputs.extra_hdus,
                )
            # Catch only the master-photref generation failure, so an
            # unrelated error inside generate_master surfaces instead of
            # being swallowed.
            except FitMagnitudesError:
                return None, None
            new_reference = get_master_photref(candidate_fname)

            iteration = self._path_substitutions["magfit_iteration"]
            if iteration >= self._configuration.max_magfit_iterations:
                return None, None
            if iteration > 0 and self._converged(old_reference, new_reference):
                return None, None
            os.replace(candidate_fname, master_path)
        return new_reference, master_fname

    def _converged(self, old_reference, new_reference):
        """True iff the references agree within ``max_photref_change``."""

        num_photometries = next(iter(old_reference.values()))["mag"].size
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
            self._logger.debug("Num photometries: %s", repr(num_photometries))
            self._logger.debug(
                "square_diff (shape=%s): %s",
                repr(square_diff.shape),
                repr(square_diff),
            )
            self._logger.debug(
                "finite_entries (shape=%s): %s",
                repr(finite_entries.shape),
                repr(finite_entries),
            )
            self._logger.debug(
                "average_square_change (shape=%s): %s",
                repr(average_square_change.shape),
                repr(average_square_change),
            )

            average_square_change[finite_entries] += square_diff[finite_entries]
            num_finite += finite_entries

        average_square_change /= num_finite
        self._logger.debug(
            "Fit iteration resulted in average square change in magnitudes of: "
            "%s",
            repr(average_square_change),
        )

        return (
            average_square_change.max()
            <= self._configuration.max_photref_change
        )


# pylint: enable=too-few-public-methods
