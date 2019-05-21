"""Define class for collecting statistics and making a master photref."""

import os
from subprocess import Popen, PIPE

import scipy
from astropy.io import fits

from superphot_pipeline.fit_expression import Interface as FitTermsInterface
from superphot_pipeline.fit_expression import iterative_fit

class MasterPhotrefCollector:
    """
    Class collecting magfit output to generate a master photometric reference.

    Attributes:
        _statistics_fname:    The file name to save the collected statistics
            under.

        _grcollect:    The running ``grcollect`` process responsible for
            generating the statistics.

        _output_lock:    A lock to ensure only only process a multiprocessing
            pool is adding output to ``grcollect`` at any one time.

        _num_photometries:    How many photometry measurements being fit.

        _source_name_format:    A %-substitution string for turning a source ID
            tuple to as string in the statistics file.

        _stat_quantities:    The quantities that ``grcollect`` is told to
            calculate for each input magnitude and error estimate.
    """

    def _calculate_statistics(self):
        """Creating the statics file from all input collected so far."""

        assert self._grcollect.poll() is None
        grcollect_output, grcollect_error = self._grcollect.communicate()
        if self._grcollect.returncode:
            raise ChildProcessError(
                'grcollect command failed! '
                +
                'Return code: %d.' % self._grcollect.returncode
                +
                '\nOutput:\n' + grcollect_output.decode()
                +
                '\nError:\n' + grcollect_error.decode()
            )

    def _get_med_count(self):
        """Return median number of observations per source in the stat. file."""

        if 'rcount' in self._stat_quantities:
            count_col = self._stat_quantities.index('rcount')
        else:
            count_col = self._stat_quantities.index('count')
        with open(self._statistics_fname, 'r') as stat_file:
            med = scipy.median([float(l.split()[count_col]) for l in stat_file])
        return med

    def _read_statistics(self, catalogue, catalogue_filter, parse_source_id):
        """
        Read the magnitude & scatter for each source for each photometry.

        Args:
            catalogue(dict):    See ``master_catalogue`` argument to
                MagnitudeFit.__init__().

            catalogue_filter(str):    See same name argument to
                generate_master().

            parse_source_id(callable):    See same name argument to
                generate_master().

        Returns:
            scipy structured array:
                The fields are as follows:

                    ID: The ID of the source.

                    <filter>: The catalogue estimated magnitude of the source.
                        The name of the column is given by ``catalogue_filter``.

                    xi, eta: The projected angular coordinates of the source
                        from the catalogue.

                    full_count: The full number of measurements available for
                        the sources

                    rejected_count: The number of measurements of the source
                        after outlier rejection during statistics collection.

                    mediandev: An estimate of the scatter for the source,
                        as the deviation around the median, calculated during
                        statistics collection.

                    medianmeddev: An estimate of the scatter for the source as
                        the median deviation around the median, calculated
                        during statistics collection.
        """

        def get_stat_data():
            """Read the statistics file."""

            column_names = ['ID']
            for phot_ind in range(self._num_photometries):
                column_names.extend(
                    [
                        quantity + '_mag_%d' % phot_ind
                        for quantity in self._stat_quantities
                    ]
                    +
                    [
                        quantity + '_mag_err_%d' % phot_ind
                        for quantity in self._stat_quantities
                    ]
                )
            return scipy.genfromtxt(self._statistics_fname,
                                    dtype=None,
                                    names=column_names)

        def create_result(num_sources, catalogue_columns):
            """Create an empty result to fill with data."""

            special_dtypes = dict(phqual='S3', magsrcflag='S9')
            dtype = (
                [
                    ('ID', scipy.intc, (len(next(iter(catalogue.keys()))),)),
                    ('full_count', scipy.intc, (self._num_photometries,)),
                    ('rejected_count', scipy.intc, (self._num_photometries,)),
                    ('median', scipy.float64, (self._num_photometries,)),
                    ('mediandev', scipy.float64, (self._num_photometries,)),
                    ('medianmeddev', scipy.float64, (self._num_photometries,))
                ]
                +
                [
                    (colname, special_dtypes.get(colname, scipy.float64))
                    for colname in catalogue_columns
                ]
            )
            return scipy.empty(
                num_sources,
                dtype=dtype
            )

        def add_stat_data(stat_data, result):
            """Add the information from get_stat_data() to result."""

            for source_index, source_id in enumerate(stat_data['ID']):
                result['ID'][source_index] = parse_source_id(source_id)

            print('Stat data columns: ' + repr(stat_data.dtype.names))
            for phot_index in range(self._num_photometries):
                result['full_count'][:, phot_index] = stat_data[
                    'count_mag_%d' % phot_index
                ]
                result['rejected_count'][:, phot_index] = stat_data[
                    'rcount_mag_%d' % phot_index
                ]
                for statistic in ['median', 'mediandev', 'medianmeddev']:
                    result[statistic][:, phot_index] = stat_data[
                        'r' + statistic + '_mag_%d' % phot_index
                    ]

        def add_catalogue_info(catalogue_columns, result):
            """Add the catalogue data for each source to the result."""

            for source_index, source_id in enumerate(result['ID']):
                catalogue_source = catalogue[tuple(source_id)]
                for colname in catalogue_columns:
                    result[colname][source_index] = catalogue_source[colname]


        catalogue_columns = next(iter(catalogue.values())).dtype.names
        stat_data = get_stat_data()

        result = create_result(stat_data.size, catalogue_columns)
        add_stat_data(stat_data, result)
        add_catalogue_info(catalogue_columns, result)

        return result

    #Can't simplify further
    #pylint: disable=too-many-locals
    @staticmethod
    def _fit_scatter(statistics,
                     fit_terms_expression,
                     *,
                     min_counts,
                     outlier_average,
                     outlier_threshold,
                     max_rej_iter,
                     scatter_quantity='medianmeddev'):
        """
        Fit for the dependence of scatter on source properties.

        Args:
            statistics:    The return value of _read_statistics().

            fit_terms_expression(str):    A fitting terms expression to use to
                generate the terms to include in the fit of the scatter.

            min_counts(int):    The smallest number of observations to require
                for a source to participate in the fit.

            outlier_average:    See ``fit_outlier_average`` argument to
                generate_master().

            outlier_threshold:    See ``fit_outlier_threshold`` argument to
                generate_master().

            max_rej_iter:    See ``fit_max_rej_iter`` argument to
                generate_master().

            scatter_quantity(str):    The name of the field in ``statistics``
                which contains the estimated scatter to fit.

        Returns:
            scipy array:
                The residuals of the scatter from ``statistics`` from the
                best-fit values found.
        """

        predictors = FitTermsInterface(fit_terms_expression)(statistics)
        num_photometries = statistics['full_count'][0].size
        residuals = scipy.empty((statistics.size, num_photometries))
        for phot_ind in range(num_photometries):
            enough_counts = (
                statistics['rejected_count'][:, phot_ind] >= min_counts
            )
            phot_predictors = predictors[:, enough_counts]
            target_values = scipy.log10(
                statistics[scatter_quantity][enough_counts, phot_ind]
            )
            coefficients = iterative_fit(
                phot_predictors,
                target_values,
                error_avg=outlier_average,
                rej_level=outlier_threshold,
                max_rej_iter=max_rej_iter,
                fit_identifier=('Generating master photometric reference, '
                                'photometry #'
                                +
                                repr(phot_ind))
            )[0]
            residuals[:, phot_ind] = (statistics[scatter_quantity][:, phot_ind]
                                      -
                                      scipy.dot(coefficients, predictors))
        return residuals
    #pylint: enable=too-many-locals

    @staticmethod
    def _create_reference(statistics,
                          residual_scatter,
                          *,
                          min_counts,
                          outlier_average,
                          outlier_threshold,
                          reference_fname):
        """
        Create the master photometric reference.

        Args:
            statistics:    The return value of _read_statistics().

            residual_scatter:    The return value of _fit_scatter().

            min_counts(int):    The smallest number of observations to require
                for a source to be included in the refreence.

            outlier_average:    See ``fit_outlier_average`` argument to
                generate_master().

            outlier_threshold:    See ``fit_outlier_threshold`` argument to
                generate_master().

            reference_fname(str):    The name to use for the generated master
                photometric reference file.

        Returns:
            None
        """

        def get_phot_reference_data(phot_ind):
            """
            Return the reference magnitude fit data as scipy structured array.
            """

            max_scatter = getattr(scipy, outlier_average)(
                residual_scatter[:, phot_ind]
            ) * outlier_threshold
            include_source = scipy.logical_and(
                statistics['rejected_count'][:, phot_ind] >= min_counts,
                residual_scatter[:, phot_ind] <= max_scatter
            )

            num_phot_sources = include_source.sum()
            reference_data = scipy.empty(
                num_phot_sources,
                dtype=[('IDprefix', 'i1'),
                       ('IDfield', scipy.intc),
                       ('IDsource', scipy.intc),
                       ('full_count', scipy.float64),
                       ('rejected_count', scipy.float64),
                       ('magnitude', scipy.float64),
                       ('mediandev', scipy.float64),
                       ('medianmeddev', scipy.float64),
                       ('scatter_excess', scipy.float64)]
            )
            for reference_column, stat_column, stat_index in [
                    ('IDprefix', 'ID', 0),
                    ('IDfield', 'ID', 1),
                    ('IDsource', 'ID', 2),
                    ('full_count', 'full_count', phot_ind),
                    ('rejected_count', 'rejected_count', phot_ind),
                    ('magnitude', 'median', phot_ind,),
                    ('mediandev', 'mediandev', phot_ind),
                    ('medianmeddev', 'medianmeddev', phot_ind)
            ]:
                reference_data[reference_column] = statistics[
                    stat_column
                ][
                    include_source,
                    stat_index
                ]
            reference_data['scatter_excess'] = residual_scatter[include_source,
                                                                phot_ind]
            return reference_data

        num_photometries = statistics['full_count'][0].size
        master_hdus = [fits.BinTableHDU(get_phot_reference_data(phot_ind))
                       for phot_ind in range(num_photometries)]
        fits.HDUList([fits.PrimaryHDU()] + master_hdus).writeto(reference_fname)

    #Could not refactor to simply.
    #pylint: disable=too-many-locals
    def __init__(self,
                 statistics_fname,
                 num_photometries,
                 temp_directory,
                 *,
                 output_lock=None,
                 outlier_threshold=5.0,
                 max_rejection_iterations=10,
                 rejection_center='median',
                 rejection_units='meddev',
                 max_memory='2g',
                 source_name_format='HAT-%03d-%07d'):
        """
        Prepare for collecting magfit results for master photref creation.

        Args:
            statistics_fname(str):    The filename where to save the statistics
                relevant for creating a master photometric reference.

            num_photometries(int):    The number of photometric measurements
                available for each star (e.g. number of apertures + 1 if psf
                fitting + ...).

            outlier_threshold(float):    A threshold value for outlier
                rejection. The units of this are determined by the
                ``rejection_units`` argument.

            max_rejection_iterations(int):    The maximum number of iterations
                between rejecting outliers and re-deriving the statistics to
                allow.

            temp_directory(str):    A location in the file system to use for
                storing temporary files during statistics colletion.

            rejection_center(str):    Outliers are define around some central
                value, either ``'mean'``, or ``'median'``.

            rejection_units(str):    The units of the outlier rejection
                threshold. One of ``'stddev'``, ``'meddev'``, or ``'absolute'``.

            max_memory(str):    The maximum amount of RAM the statistics process
                is allowed to use (if exceeded intermediate results are dumped
                to files in ``temp_dir``).

        Returns:
            None
        """

        grcollect_cmd = ['grcollect', '-', '-V', '--stat']
        stat_columns = range(2, 2 * num_photometries + 2)
        self._num_photometries = num_photometries
        self._stat_quantities = ['count',
                                 'count',
                                 'median',
                                 'mediandev',
                                 'medianmeddev']
        if outlier_threshold:
            for i in range(1, len(self._stat_quantities)):
                self._stat_quantities[i] = 'r' + self._stat_quantities[i]
            grcollect_cmd.append(','.join(self._stat_quantities))
            for col in stat_columns:
                grcollect_cmd.extend([
                    '--rejection',
                    (
                        'column=%d,iterations=%d,%s,%s=%f'
                        %
                        (
                            col,
                            max_rejection_iterations,
                            rejection_center,
                            rejection_units,
                            outlier_threshold
                        )
                    )
                ])
        else:
            grcollect_cmd.append(','.join(self._stat_quantities))

        self._statistics_fname = statistics_fname

        grcollect_cmd.extend([
            '--col-base', '1',
            '--col-stat', ','.join([str(c) for c in stat_columns]),
            '--max-memory', max_memory,
            '--tmpdir', temp_directory,
            '--output', statistics_fname
        ])
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        print("Starting grcollect command: '" + "' '".join(grcollect_cmd) + "'")
        self._grcollect = Popen(grcollect_cmd,
                                stdin=PIPE,
                                stdout=PIPE,
                                stderr=PIPE)
        self._output_lock = output_lock
        self._num_photometries = num_photometries
        self._source_name_format = source_name_format
    #pylint: enable=too-many-locals

    def add_input(self, phot, fitted):
        """Ingest a fitted frame's photometry into the statistics."""

        assert self._grcollect.poll() is None

        formal_errors = phot['mag_err'][:, -1]
        phot_flags = phot['phot_flag'][:, -1]

        src_count = formal_errors.shape[0]
        assert self._num_photometries == formal_errors.shape[1]
        assert formal_errors.shape == phot_flags.shape
        assert fitted.shape == formal_errors.shape

        line_format = (self._source_name_format
                       +
                       (' %9.5f') * (2 * self._num_photometries))
        if self._output_lock is not None:
            self._output_lock.acquire()
        for source_ind in range(src_count):
            print_args = (
                tuple(phot['ID'][source_ind][1:])
                +
                tuple(fitted[source_ind])
                +
                tuple(formal_errors[source_ind])
            )

            all_finite = True
            for value in print_args:
                if scipy.isnan(value):
                    all_finite = False
                    break

            if all_finite:
                self._grcollect.stdin.write(
                    (line_format % print_args + '\n').encode('ascii')
                )
        if self._output_lock is not None:
            self._output_lock.release()

    #TODO: Add support for scatter cut based on quantile of fit residuals.
    def generate_master(self,
                        *,
                        master_reference_fname,
                        catalogue,
                        catalogue_filter,
                        fit_terms_expression,
                        parse_source_id,
                        min_nobs_median_fraction=0.5,
                        fit_outlier_average='median',
                        fit_outlier_threshold=3.0,
                        fit_max_rej_iter=20):
        """
        Finish the work of the object and generate a master.

        Args:
            master_reference_fname(str):   The file name to use for the newly
                created master photometric reference.

            catalogque:     See ``master_catalogue`` argument to
                MagnitudeFit.__init__().

            catalogque_filter(str):   The column in the catalogue that contains
                the magntiude closest to what is measured using the input
                images.

            fit_terms_expression(str):    An expression expanding to the terms
                to include in the scatter fit. May use any catalogue column.

            parse_source_id(callable):    Should convert a string source ID into
                the source ID format expected by the catalogue.

            min_nobs_median_fraction(float):    The minimum fraction of the
                median number of observations a source must have to be inclruded
                in the master.

            fit_outlier_average(str):    The averaging method to use for scaling
                averaging residuals from scatter fit. The result is used as the
                unit for ``fit_outlier_threshold``.

            fit_outlier_threshold(float):    A factor to multiply the
                ``fit_outlier_average`` averaged residual from scatter fit in
                order to get the threshold to consider a scatter point an
                outlier, and hence discard from cantributing to the reference.

            fit_max_rej_iter(int):    The maximum number of iterations to allow
                for fitting/rejecting outliers. If this number is reached, the
                last result is accepted.

        Returns:
            None
        """

        self._calculate_statistics()
        statistics = self._read_statistics(
            catalogue,
            catalogue_filter,
            parse_source_id
        )
        min_counts = min_nobs_median_fraction * self._get_med_count()
        residual_scatter = self._fit_scatter(
            statistics,
            fit_terms_expression,
            min_counts=min_counts,
            outlier_average=fit_outlier_average,
            outlier_threshold=fit_outlier_threshold,
            max_rej_iter=fit_max_rej_iter
        )
        self._create_reference(statistics=statistics,
                               residual_scatter=residual_scatter,
                               min_counts=min_counts,
                               outlier_average=fit_outlier_average,
                               outlier_threshold=fit_outlier_threshold,
                               reference_fname=master_reference_fname)
