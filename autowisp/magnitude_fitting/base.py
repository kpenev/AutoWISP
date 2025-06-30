"""Magnitude fitting interface."""

import sys
import logging
from multiprocessing import current_process
from traceback import format_exception
from abc import ABC, abstractmethod

from numpy.lib import recfunctions
import numpy

from autowisp import DataReductionFile
from autowisp.evaluator import Evaluator
from autowisp.magnitude_fitting.util import get_magfit_sources


# Could not think of a sensible way to reduce number of attributes
# pylint: disable=too-many-instance-attributes
# Still makes sense es a class.
# pylint: disable=too-few-public-methods
class MagnitudeFit(ABC):
    """
    A base class for all classes doing magnitude fitting.

    Takes care of adding fitted magnitudes to data reduction files and updating
    the database.

    Attributes:
        config:    See `config` argument to __init__().

        logger:    A python logging logger for emitting messages on the progress
            and status of magnitude fitting.

        _dr_fname:    The name of the data reduction file currently undergoing
            magnitude fitting.

        _magfit_collector:    See `magfit_collector` argument to __init__().

        _reference:    See `reference` argument to __init__().

        _catalogue:    See `master_catalogue` argument to __init__().

        _source_name_format:    See `source_name_format` argument to __init__().
    """

    # TODO: revive once database design is complete
    def _add_fit_to_db(self, coefficients, **fit_diagnostics):
        """
        Record the given best fit coefficient and diagnostics in the database.

        Args:
            coefficients (iterable of values):    The best fit coefficients for
                the magnitude fitting of the current data reduction file.

            fit_diagnostics:    Any information about the fit that should be
                recorded in the database. The names of the arguments are assumed
                to correspond to column names in the magnitude fitting table.

        Returns:
            None

        For now disabled.

        Code from HATpipe:

        def _update_db(self,
                       values,
                       apind,
                       fit_res,
                       start_src_count,
                       final_src_count):

            if self.database is None : return
            args=((self._header['STID'],
                   self._header['FNUM'],
                   self._header['CMPOS'],
                   self._header['PROJID'],
                   self._header['SPRID'],
                   apind,
                   self.config.version,
                   start_src_count,
                   final_src_count,
                   (None if fit_res is None else float(fit_res)))
                  +
                  tuple((None if v is None else float(v)) for v in values))
            statement=('REPLACE INTO `'+self.config.dest_table+
                       '` (`station_id`, `fnum`, `cmpos`, `project_id`, '
                       '`sphotref_id`, `aperture`, `magfit_version`, '
                       '`input_src`, `non_rej_src`, `rms_residuals`, `'+
                       '`, `'.join(self._db_columns)+'`) VALUES (%s'+
                       ', %s'*(len(args)-1)+')')
            self._log_to_file(
                'Inserting into DB:\n'
                +
                '\t' + statement + '\n'
                +
                '\targs: ' + repr(args) + '\n'
                +
                '\targ types: ' + repr([type(v) for v in args]) + '\n'
            )
            self.database(statement, args)
        """

    @abstractmethod
    def _fit(self, fit_data):
        """
        Perform a fit for the magfit correction.

        Args:
            fit_data (numpy structured array):    The current photometry being
                fitted. It should contain the source information from the frame
                being fit, the photometry from the reference (and optionally
                reference position if used) and catalogue information for each
                source.

        Returns:
            [[dict]]:
                For each photometry (i.e. different aperture of PSF fitting)
                a list of dictionaries is returned containing a fit result
                dictionary for each group of sources with the following entries:

                    :coefficients: The best fit coefficients after the rejection
                                   iterations of the fit has converged.

                    :residual: The residual of the latest iteration of the fit,

                    :initial_src_count: The number of stars the fit started with
                                        before any rejections,

                    :final_src_count: The number of stars left in the fit after
                                      the final rejection iteration,
                    :group_id: The ID of the group of sources this fit result
                               corresponds to.
        """

    @abstractmethod
    def _apply_fit(self, phot, fit_results):
        """
        Return corrected magnitudes using best fit magfit coefficients.

        Args:
            phot:    The current photometry being fit, including catalogue
                information.

            fit_results:    The best fit parameters derived using _fit().

        Returns:
            numpy.array (number sources x number photometry methods):
                The magnitude fit corrected magnitudes.
        """

    def _solved(
        self,
        *,
        data_reduction,
        deleted_phot_indices,
        num_phot,
        num_sources,
        dr_path_substitutions,
    ):
        """Return fitted mags if fit is already in DR, None otherwise."""

        try:
            fitted = numpy.empty((num_sources, num_phot), dtype=float)
            phot_ind = 0
            if data_reduction.has_shape_fit(
                accept_zeropsf=False, **dr_path_substitutions
            ):
                fitted[:, 0] = data_reduction.get_dataset(
                    "shapefit.magfit.magnitude",
                    expected_shape=(num_sources,),
                    **dr_path_substitutions,
                )
                phot_ind += 1
            while phot_ind < num_phot:
                fitted[:, phot_ind] = data_reduction.get_dataset(
                    "apphot.magfit.magnitude",
                    expected_shape=(num_sources,),
                    **dr_path_substitutions,
                )
                phot_ind += 1
                numpy.delete(fitted, deleted_phot_indices, axis=0)
            return fitted
        except (KeyError, IOError):
            return None

    def _set_group(self, evaluator, result):
        """Set the fit_group column in result per grouping configuration."""

        self.logger.debug("Grouping expression: %s", repr(self.config.grouping))
        conditions = evaluator(self.config.grouping)
        self.logger.debug("Grouping conditions: %s", repr(conditions))
        groups = numpy.unique(conditions)
        self.logger.debug("Groups: %s", repr(groups))
        for group_id, group_condition in enumerate(groups):
            in_group = conditions == group_condition
            self.logger.debug(
                "Group %d contains %d entries", group_id, in_group.sum()
            )
            result["fit_group"][in_group] = group_id

    def _get_fit_indices(self, phot, evaluator, no_catalogue):
        """
        Return a list of the indices within phot of sources to use for mag fit.

        Exclude sources based on their catalague information.

        Args:
            phot:   See return of DataReductionFile.get_photometry().

            evaluator(Evaluator):    An object capable of evaluating expressions
                involving fields from phot.

            no_catalogue:    The list of source indices for which no catalogue
                information is available (all are rejected).

        Returns:
            [int]:
                A list of the indices within phot of the sources which pass all
                catalogue requirements for inclusion in the magnitude fit.

            int:
                The number of dropped sources.

            tuple:
                The ID of one of the sources dropped, None if no sources were
                dropped.
        """

        include_flag = evaluator(self.config.fit_source_condition)
        include_flag[no_catalogue] = False

        result = include_flag.nonzero()[0]

        num_skipped = len(phot["source_id"]) - result.size
        if num_skipped:
            first_skipped = numpy.logical_not(include_flag).nonzero()[0][0]
            skipped_example = phot["source_id"][first_skipped]
        else:
            skipped_example = None

        return result, num_skipped, skipped_example

    def _match_to_reference(self, phot, no_catalogue, evaluator):
        """
        Add photometric reference information, and filter per self.config.

        Args:
            phot:    See return of add_catalogue_info()

            no_catalogue (list):    The source indices from phot for which
                no catalogue information was available, so default values were
                used. All identified sources are omitted from the result.

            evaluator (Evaluator):     An object capable of evaluating
                expressions involving photometry and catalogue columns.

        Returns:
            a photometry structure like phot but for each aperture 'ref mag' is
            added - the magnitude the source has in the reference - and sources
            that should not be used in the fit because they are not in the
            reference or do not satisfy the criteria based on catalogue
            quantities are removed.
        """

        def initialize_result():
            """Return an empty result structure."""

            dtype = [
                (field[0], field[1][0]) for field in phot.dtype.fields.items()
            ]
            if self.config.reference_subpix:
                dtype.extend(
                    [("x_ref", numpy.float64), ("y_ref", numpy.float64)]
                )
            dtype.extend(
                [
                    ("ref_mag", numpy.float64, phot["mag"][0].shape),
                    ("ref_mag_err", numpy.float64, phot["mag_err"][0].shape),
                ]
            )
            print("Result dtype: " + repr(dtype))
            return numpy.empty(phot.shape, dtype=dtype)

        result = initialize_result()
        not_in_ref = [0, None]
        fit_indices, num_skipped, skipped_example = self._get_fit_indices(
            phot, evaluator, no_catalogue
        )
        result_ind = 0
        for phot_ind in fit_indices:
            source_id = phot["source_id"][phot_ind]
            if source_id.shape != ():
                source_id = tuple(source_id)
            ref_info = self._reference.get(source_id)
            if ref_info is None:
                if not_in_ref[0] == 0:
                    not_in_ref[1] = result["source_id"][result_ind]
                not_in_ref[0] += 1
                continue

            for colname in phot.dtype.names:
                result[colname][result_ind] = phot[colname][phot_ind]

            if self.config.reference_subpix:
                result["x_ref"][result_ind] = ref_info["x"]
                result["y_ref"][result_ind] = ref_info["y"]
            result["ref_mag"][result_ind] = ref_info["mag"]
            result["ref_mag_err"][result_ind] = ref_info["mag_err"]
            result_ind += 1

        if result_ind == 0:
            print(repr(not_in_ref[1]))
            self.logger.error(
                (
                    "All %d sources discarded from %s: %d skipped "
                    "(example %s), %d not in the %d sources of the reference "
                    "(example %s.), fit source condition: %s"
                ),
                len(phot["source_id"]),
                self._dr_fname,
                num_skipped,
                (
                    self._source_name_format.format(skipped_example)
                    if skipped_example is not None
                    else "-"
                ),
                not_in_ref[0],
                len(self._reference.keys()),
                (
                    self._source_name_format.format(not_in_ref[1])
                    if not_in_ref[1] is not None
                    else "-"
                ),
                self.config.fit_source_condition,
            )
        return result[:result_ind], fit_indices

    # TODO: revive once database design is complete
    def _update_calib_status(self):
        """
        Record in the database that the current header has been magfitted.

        For now disabled.

        Code from HATpipe:

        self.database(
            'UPDATE `' + raw_db_table(self._header['IMAGETYP'])
            + '` SET `calib_status`=%s WHERE `station_id`=%s AND `fnum`=%s '
            'AND `cmpos`=%s',
            (
                self.config.calib_status,
                self._header['STID'],
                self._header['FNUM'],
                self._header['CMPOS']
            )
        )
        """

    # TODO: Is this necessary?
    def _downgrade_calib_status(self):
        """
        Deal with bad photometry for a frame.

        Decrements the calibration status of the given file to astrometry
        and deletes the raw photometry file.

        For now disabled.

        Code from HATpipe:

        sys.stderr.write('bad photometry encountered:'+str(self._header)+
                         '\n')
        if(self.database is None) : return
        self._log_to_file(
            'Downgrading status of header: ' + str(self._header) + '\n'
        )
        sys.stderr.write('downgrading calibration status of'+
                         str(self._header)+'\n')
        self.database('UPDATE `'+raw_db_table(self._header['IMAGETYP'])+'` SET '
                '`calib_status`=%s WHERE `station_id`=%s AND `fnum`=%s AND '
                '`cmpos`=%s', (object_status['good_astrometry'],
                               self._header['STID'], self._header['FNUM'],
                               self._header['CMPOS']))
        sys.stderr.write('removing:'+self._fit_file+'\n')
        os.remove(self._fit_file)
        """

    @staticmethod
    def _combine_fit_statistics(fit_results):
        """
        Combine the statistics summarizing how the fit went from all groups.

        Properly combines values from the individual group fits into single
        numbers for each photometry method. The quantities processed are:
        residual, initial_src_count, final_src_count.

        Args:
            fit_results:    The best fit results for this photometry.

        Returns:
            dict:
                The derived fit statistics. Keys are residual,
                initial_src_count, and final_src_count, with one entry for
                each input photometry method.
        """

        num_photometries = len(fit_results)
        result = {
            "residual": numpy.empty(num_photometries, numpy.float64),
            "initial_src_count": numpy.zeros(num_photometries, numpy.int_),
            "final_src_count": numpy.zeros(num_photometries, numpy.int_),
        }

        for phot_ind, phot_result in enumerate(fit_results):
            for group_result in phot_result:
                for key in ["initial_src_count", "final_src_count"]:
                    result[key][phot_ind] += group_result[key]
            result["residual"][phot_ind] = numpy.nanmedian(
                [
                    group_result["residual"] or numpy.nan
                    for group_result in phot_result
                ]
            )

        return result

    def __init__(
        self,
        *,
        reference,
        #                 master_catalogue,
        config,
        magfit_collector=None,
        source_name_format,
    ):
        """
        Initializes a magnditude fitting object.

        Args:
            reference(dict):    the reference against which fitting is done.
                Should be indexed by source and contain entries implementing
                the dict interface with keys 'mag', 'mag_err' and optionally
                'x' and 'y' if the sub-pixel position of the source in the
                reference is to be used in magnitude fitting.

            #master_catalogue(pandas.DataFrame):    should be indexed by source
            #    id and contain relevant catalog information.


            config:    An object with attributes configuring how to
                perform magnitude fitting. It should provide at least the
                following attributes:

                    * fit_source_condition: An expression involving catalogue,
                      reference and/or photometry variables which evaluates to
                      zero if a source should be excluded and any non-zero value
                      if it should be included in the magnitude fit.

                    * reference_subpix: Should the magnitude fitting correction
                      depend on the sub-pixel position of the source in the
                      reference frame.

                    * grouping: An expressions using catalogue, and/or
                      photometry variables which evaluates to a tuple of boolean
                      values. Each distinct tuple defines a separate fitting
                      group (i.e. a group of sources which participate in
                      magnitude fitting together, excluding sources belonging to
                      other groups).

            magfit_collector(MasterPhotrefCollector):    Object collecting
                fitted magnitedes for generating statistics of the scatter after
                magnitude fitting.
        """

        self.config = config
        self._dr_fname = None
        self._magfit_collector = magfit_collector
        self._reference = reference
        #        self._catalogue = master_catalogue
        self._source_name_format = source_name_format
        self.logger = None

    def __call__(self, dr_fname, mark_start, mark_end, **dr_path_substitutions):
        """
        Performs the fit for the latest magfit iteration for a single frame.

        Args:
            dr_fname:    The name of the data reductio file to fit.

            dr_path_substitutions:    See path_substitutions argument
                to DataReduction.get_source_data().

        Returns:
            array:
                The non-rejected photometry for the frame.

            array:
                The magfit corrected non-rejected photometry for the frame.
        """

        self.logger = logging.getLogger(__name__)

        try:
            self.logger.debug(
                "Process %d fitting: %s.", current_process().pid, dr_fname
            )
            with DataReductionFile(dr_fname, mode="r+") as data_reduction:
                self._dr_fname = dr_fname
                phot = get_magfit_sources(
                    data_reduction,
                    magfit_iterations=[-1],
                    **dr_path_substitutions,
                )
                self.logger.debug("Starting photometry: %s", repr(phot))

                self.logger.debug(
                    "Starting photometry columns: %s", repr(phot.dtype.names)
                )

                if not phot.size:
                    self.logger.warning("Downgrading calib status.")
                    self._downgrade_calib_status()
                    return None, None
                # TODO: revive filtering by catalogue
                no_catalogue, deleted_phot_indices = [], []
                if getattr(self.config, "grouping", None) is not None:
                    phot = recfunctions.append_fields(
                        phot, ["fit_group"], [[]], usemask=False
                    )

                self.logger.debug(
                    "Photometry columns: %s", repr(phot.dtype.names)
                )
                evaluator = Evaluator(phot)
                if self.config.grouping is not None:
                    self._set_group(evaluator, phot)

                self.logger.debug("Checking for existing solution.")
                fitted = self._solved(
                    data_reduction=data_reduction,
                    deleted_phot_indices=deleted_phot_indices,
                    num_phot=phot["mag"].shape[2],
                    num_sources=phot["mag"].shape[0],
                    dr_path_substitutions=dr_path_substitutions,
                )

                self.logger.debug("Matching to reference.")
                fit_base, fit_indices = self._match_to_reference(
                    phot, no_catalogue, evaluator
                )

                if fitted:
                    return phot[fit_indices], fitted[fit_indices]

                if fit_base.size > 0:
                    self.logger.debug("Performing linear fit.")
                    fit_results = self._fit(fit_base)

                if fit_results:
                    self.logger.debug("Post-processing fit.")
                    fitted = self._apply_fit(phot, fit_results)
                    assert fitted.shape == (
                        phot["mag"].shape[0],
                        phot["mag"].shape[2],
                    )
                    self.logger.debug("Adding to DR file.")
                    mark_start(dr_fname)
                    data_reduction.add_magnitude_fitting(
                        fitted_magnitudes=fitted,
                        fit_statistics=self._combine_fit_statistics(
                            fit_results
                        ),
                        magfit_configuration=self.config,
                        missing_indices=deleted_phot_indices,
                        **dr_path_substitutions,
                    )
                    mark_end(dr_fname)
                    self.logger.debug("Updating calibration status.")
                    return phot[fit_indices], fitted[fit_indices]
                return None, None
        except Exception as ex:
            # Does not make sense to avoid building message.
            # pylint: disable=logging-not-lazy
            self.logger.critical(
                str(ex)
                + "\n"
                + "".join(format_exception(*sys.exc_info()))
                + "\nBad DR:"
                + dr_fname
            )
            # pylint: enable=logging-not-lazy
            raise


# pylint: enable=too-many-instance-attributes
# pylint: enable=too-few-public-methods
