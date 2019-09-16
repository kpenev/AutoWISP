"""Magnitude fitting interface."""

import sys
import logging
from multiprocessing import current_process
from traceback import format_exception
from abc import ABC, abstractmethod

import scipy
from numpy.lib import recfunctions

from superphot_pipeline import DataReductionFile
from superphot_pipeline.evaluator import Evaluator

#Could not think of a sensible way to reduce number of attributes
#pylint: disable=too-many-instance-attributes
#Still makes sense es a class.
#pylint: disable=too-few-public-methods
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

    #TODO: revive once database design is complete
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
            dict:

                * initial_src_count(int): How many sources did the fit start
                    with before any iterations of outlier rejection.

                * final_src_count(int): After all iterations of outlier
                    rejections, how many sources was the final fit based on.

                * residual(float): The residual of the final fit iteration.

                * parameters(numpy.array): The best fit parameters defining
                    the magnitude correction.
        """

    @abstractmethod
    def _apply_fit(self, phot, fit_results):
        """
        Return corrected magnitudes using best fit magfit coefficients.

        Args:
            phot:    The current photometry being fit, including catalogue
                information, i.e. the object returned by add_catalogue_info().

            fit_results:    The best fit parameters derived using _fit().

        Returns:
            numpy.array (number sources x number photometry methods):
                The magnitude fit corrected magnitudes.
        """

    #TODO: revive once database design is complete
    def _solved(self, num_photometries):
        """
        Return the best fit coefficients if already in the database.

        For row disabled.

        Code from HATpipe:

        if self.database is None : return False
        coefficients=[]
        for phot_ind in range(num_photometries) :
            statement=(' FROM `'+self.config.dest_table+
                       '` WHERE `station_id`=%s AND `fnum`=%s AND '
                       '`cmpos`=%s AND `aperture`=%s AND '
                       '`sphotref_id`=%s AND `magfit_version`=%s')
            args=(self._header['STID'], self._header['FNUM'],
                  self._header['CMPOS'], phot_ind, self._header['SPRID'],
                  self.config.version)
            if self.__rederive_fit :
                statement='DELETE '+statement
                self.database(statement, args)
            else :
                statement=('SELECT `input_src`, `non_rej_src`, '
                           '`rms_residuals`, `'
                           +
                           '`, `'.join(self._db_columns)
                           +
                           '`'
                           +
                           statement)
                record=self.database(statement, args)
                if record is None : return False
                else :
                    coefficients.append(
                        dict(coefficients=(None if record[3] is None
                                           else record[3:]),
                             residual=record[2],
                             initial_src_count=record[0],
                             final_src_count=record[1])
                    )
        if self.__rederive_fit : return False
        else : return coefficients
        """

    def _add_catalogue_info(self, phot):
        """
        Add source information from the input master catalogue and grouping.

        If no information is found for the given source, if the catalogue has
        a default defined that is used instead, otherwise the source is deleted
        from phot.

        Args:
            phot:    See return of DataReductionFile.get_photometry()

        Returns:
            scipy record array:
                Same structure as phot, but augmented with the catalogue columns
                for the included sources. May drop sources from phot if not
                catalogue default is provided.

            set:
                The indices for which no catalogue information was found, and
                the catalogue default was used.

            sorted list:
                The indices within the original phot which were deleted bacause
                of missing catalogue information, and no default.
        """

        new_column_names = next(iter(self._catalogue.values())).dtype.names
        special_dtypes = dict(phqual='S3',
                              magsrcflag='S9')

        new_column_data = [
            scipy.empty(phot.size,
                        dtype=special_dtypes.get(colname, scipy.float64))
            for colname in new_column_names
        ]
        default_cat = (self._catalogue['default']
                       if 'default' in self._catalogue else None)
        default_cat_indices = []
        remove_source_indices = []
        for source_ind in range(phot.size):
            src = tuple(phot['ID'][source_ind])
            if src in self._catalogue:
                cat_source = self._catalogue[src]
            elif default_cat:
                cat_source = default_cat
                default_cat_indices.append(source_ind
                                           -
                                           len(remove_source_indices))
                self.logger.warning(
                    '%s has no catalogue informaiton, using default.',
                    self._source_name_format % src
                )
            else:
                remove_source_indices.append(source_ind)
                continue
            for col_index, col_name in enumerate(new_column_names):
                new_column_data[col_index][source_ind] = cat_source[col_name]

        if hasattr(self.config, 'grouping'):
            new_column_names += ('fit_group',)
            new_column_data.append([])

        result = scipy.delete(
            recfunctions.append_fields(phot,
                                       new_column_names,
                                       data=new_column_data,
                                       usemask=False),
            remove_source_indices
        )

        return result, default_cat_indices, remove_source_indices

    def _set_group(self, evaluator, result):
        """Set the fit_group column in result per grouping configuration."""

        conditions = scipy.array(
            evaluator(self.config.grouping)
        ).transpose()
        groups = scipy.unique(conditions, axis=0)
        for group_id, group_condition in enumerate(groups):
            result['fit_group'][
                #false positive
                #pylint: disable=no-member
                scipy.logical_and.reduce(conditions == group_condition, 1)
                #pylint: enable=no-member
            ] = group_id

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

        num_skipped = len(phot['ID']) - result.size
        if num_skipped:
            first_skipped = scipy.logical_not(include_flag).nonzero()[0][0]
            skipped_example = phot['ID'][first_skipped]
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

            dtype = [(field[0], field[1][0])
                     for field in phot.dtype.fields.items()]
            if self.config.reference_subpix:
                dtype.extend([('x_ref', scipy.float64),
                              ('y_ref', scipy.float64)])
            dtype.extend([
                ('ref_mag', scipy.float64, phot['mag'][0].shape),
                ('ref_mag_err', scipy.float64, phot['mag_err'][0].shape)
            ])
            print('Result dtype: ' + repr(dtype))
            return scipy.empty(phot.shape, dtype=dtype)

        result = initialize_result()
        not_in_ref = [0, (0, 0)]
        fit_indices, num_skipped, skipped_example = self._get_fit_indices(
            phot,
            evaluator,
            no_catalogue
        )
        result_ind = 0
        for phot_ind in fit_indices:
            ref_info = self._reference.get(tuple(phot['ID'][phot_ind]))
            if ref_info is None:
                if not_in_ref[0] == 0:
                    not_in_ref[1] = result['ID'][result_ind]
                not_in_ref[0] += 1
                continue

            for colname in phot.dtype.names:
                result[colname][result_ind] = phot[colname][phot_ind]

            if self.config.reference_subpix:
                result['x_ref'][result_ind] = ref_info['x']
                result['y_ref'][result_ind] = ref_info['y']
            result['ref_mag'][result_ind] = ref_info['mag']
            result['ref_mag_err'][result_ind] = ref_info['mag_err']
            result_ind += 1

        if result_ind == 0:
            print(repr(skipped_example))
            self.logger.error(
                (
                    'All %d sources discarded from %s: %d skipped '
                    '(example %s), %d not in the %d sources of the reference '
                    '(example %d-%d.), fit source condition: %s'
                ),
                len(phot['ID']),
                self._dr_fname,
                num_skipped,
                self._source_name_format % tuple(skipped_example[1:]),
                not_in_ref[0],
                len(self._reference.keys()),
                *not_in_ref[1],
                self.config.fit_source_condition
            )
        return result[:result_ind]

    #TODO: revive once database design is complete
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

    #TODO: Is this necessary?
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

    def __init__(self,
                 *,
                 reference,
                 master_catalogue,
                 config,
                 magfit_collector=None,
                 source_name_format='HAT-%03d-%07d'):
        """
        Initializes a magnditude fitting object.

        Args:
            reference(dict):    the reference against which fitting is done.
                Should be indexed by source and contain entries implementing
                the dict interface with keys 'mag', 'mag_err' and optionally
                'x' and 'y' if the sub-pixel position of the source in the
                reference is to be used in magnitude fitting.

            master_catalogue(dict):    should be indexed by sources (field,
                source number) containing dictonaries with relevant 2mass
                information.


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
        self._catalogue = master_catalogue
        self._source_name_format = source_name_format
        self.logger = logging.getLogger(__name__)

    def __call__(self, dr_fname, **dr_path_substitutions):
        """
        Performs the fit for the latest magfit iteration for a single frame.

        Args:
            dr_fname:    The name of the data reductio file to fit.

            dr_path_substitutions:    See path_substitutions argument
                to DataReduction.get_source_data().

        Returns:
            None
        """

        def combine_fit_statistics(fit_results):
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
            result = dict(
                residual=scipy.empty(num_photometries, scipy.float64),
                initial_src_count=scipy.zeros(num_photometries, scipy.int_),
                final_src_count=scipy.zeros(num_photometries, scipy.int_),
            )

            for phot_ind, phot_result in enumerate(fit_results):
                for group_result in phot_result:
                    for key in ['initial_src_count', 'final_src_count']:
                        result[key][phot_ind] += group_result[key]
                result['residual'][phot_ind] = scipy.nanmedian([
                    group_result['residual'] or scipy.nan
                    for group_result in phot_result
                ])

            return result

        try:
            self.logger.debug('Process %d fitting: %s.',
                              current_process().pid,
                              dr_fname)
            data_reduction = DataReductionFile(dr_fname, mode='r+')
            self._dr_fname = dr_fname
            phot = data_reduction.get_source_data(magfit_iterations=[-1],
                                                  string_source_ids=False,
                                                  **dr_path_substitutions)

            if not phot.size:
                self.logger.warning('Downgrading calib status.')
                self._downgrade_calib_status()
                return
            self.logger.debug('Adding catalogue information.')
            phot, no_catalogue, deleted_phot_indices = (
                self._add_catalogue_info(phot)
            )
            evaluator = Evaluator(phot)
            if hasattr(self.config, 'grouping'):
                self._set_group(evaluator, phot)

            self.logger.debug('Checking for existing solution.')
            fit_results = self._solved(phot['mag'].shape[2]) or False
            if fit_results:
                assert 'fit_groups' not in phot
                fit_results = [fit_results]
            else:
                self.logger.debug('Matching to reference.')
                fit_base = self._match_to_reference(phot,
                                                    no_catalogue,
                                                    evaluator)
                if fit_base.size > 0:
                    self.logger.debug('Performing linear fit.')
                    fit_results = self._fit(fit_base)
            if fit_results:
                self.logger.debug('Post-processing fit.')
                fitted = self._apply_fit(phot, fit_results)
                assert fitted.shape == (phot['mag'].shape[0],
                                        phot['mag'].shape[2])
                fit_statistics = combine_fit_statistics(fit_results)

                self.logger.debug('Adding to DR file.')
                data_reduction.add_magnitude_fitting(
                    fitted_magnitudes=fitted,
                    fit_statistics=fit_statistics,
                    magfit_configuration=self.config,
                    missing_indices=deleted_phot_indices,
                    **dr_path_substitutions
                )
                data_reduction.close()
                self.logger.debug('Updating calibration status.')
                if self._magfit_collector is not None:
                    self.logger.debug('Outputting %d sources.',
                                      len(phot['ID']))
                    self._magfit_collector.add_input(phot, fitted)
        except Exception as ex:
            #Does not make sense to avoid building message.
            #pylint: disable=logging-not-lazy
            self.logger.critical(str(ex)
                                 +
                                 "\n"
                                 +
                                 "".join(format_exception(*sys.exc_info()))
                                 +
                                 "\nBad DR:"
                                 +
                                 dr_fname)
            #pylint: enable=logging-not-lazy
            raise
#pylint: enable=too-many-instance-attributes
#pylint: enable=too-few-public-methods
