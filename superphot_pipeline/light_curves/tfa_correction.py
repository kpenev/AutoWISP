"""Define aclass for applying TFA corrections to lightcurves."""

from matplotlib import pyplot

import scipy
import scipy.linalg
#pylint false positive
#pylint: disable=no-name-in-module
from scipy.spatial import cKDTree
#pylint: enable=no-name-in-module

from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.fit_expression import iterative_fit, iterative_fit_qr
from .light_curve_file import LightCurveFile
from .correction import Correction

class TFACorrection(Correction):
    """
    Class for performing TFA corrections to a set of lightcurves.

    Attributes:
        _configuration:    An object with attributes configuring how TFA is to
            be done (see `configuration` argument to __init__()).

        _template_qr([]):    The QR decomposition of the template lightcurves.
            Each entry is the value returned by scipy.linalg.qr for one of the
            fit datasets.
    """

    def _get_io_configurations(self):
        """
        Return properly formatted configuration to pass to :func:`_save_result`.
        """

        def default_format(config):
            """Encode strings as ascii, leave everything else unchanged."""

            try:
                return config.encode('ascii')
            except AttributeError:
                return b'None' if config is None else config

        result = []
        for fit_target in self.fit_datasets:
            pipeline_key_prefix = self._get_config_key_prefix(fit_target)
            result.append(
                [
                    (
                        (
                            pipeline_key_prefix
                            +
                            ('num_templates' if cfg_key == 'sqrt_num_templates'
                             else cfg_key)
                        ),
                        default_format(
                            getattr(self._configuration,
                                    cfg_key,
                                    None
                                    )
                        )
                    )
                    for cfg_key in ['saturation_magnitude',
                                    'mag_rms_dependence_order',
                                    'mag_rms_outlier_threshold',
                                    'mag_rms_max_rej_iter',
                                    'max_rms',
                                    'faint_mag_limit',
                                    'min_observations_quantile',
                                    'sqrt_num_templates',
                                    'fit_points_filter_variables',
                                    'fit_points_filter_expression']
                ]
                +
                self._get_io_iterative_fit_config(pipeline_key_prefix)
            )

        return result

    def get_xi_eta_grid(self, epd_statistics):
        """Return a grid of (xi, eta) values to select closest templates to."""

        min_xi = epd_statistics['xi'].min()
        max_xi = epd_statistics['xi'].max()
        min_eta = epd_statistics['eta'].min()
        max_eta = epd_statistics['eta'].max()
        grid_res = self._configuration.sqrt_num_templates
        return scipy.mgrid[
            min_xi : max_xi : grid_res * 1j,
            min_eta : max_eta : grid_res * 1j
        ].transpose().reshape(grid_res * grid_res, 2)


    def _plot_template_selection(self,
                                 epd_statistics,
                                 template_indices,
                                 allowed_stars,
                                 plot_fname_pattern):
        """
        Create plots showing the stars selected as template (1 plot/phot).

        Args:
            epd_statistics:    See __init__().

            template_indices:    The return value of _select_template_stars().

            plot_fname_pattern(str):    A %(phot_index)??d and %(plot_id)??s
                substitutions expanding to a unique filename to save the
                collection of plots for each given photometry index. The plot_id
                substitution will be one of: `'xi_eta'`, `'mag_rms'`, and
                `'mag_nobs'` each containing a plot of the associated
                quantities as x and y coordinates respectively.

        Returns:
            None
        """

        def split_epd_statistics(phot_template_indices, phot_index):
            """Split the EPD statistics into selected/allowed/rejected."""

            selected = epd_statistics[phot_template_indices]

            rejected = scipy.ones(epd_statistics.shape, dtype=bool)
            rejected[allowed_stars[:, phot_index]] = False
            rejected = epd_statistics[rejected]
            allowed = epd_statistics[allowed_stars[:, phot_index]]

            return selected, allowed, rejected

        def create_xi_eta_plot(selected,
                               allowed,
                               rejected,
                               phot_index,
                               **fname_substitutions):
            """Create a plot of projected catalogue positions."""

            fname_substitutions['plot_id'] = 'xi_eta'
            plot_fname = plot_fname_pattern % dict(
                phot_index=phot_index,
                **fname_substitutions
            )

            axis = pyplot.gca()

            grid = self.get_xi_eta_grid(allowed)

            for (grid_xi, grid_eta), source in zip(grid, selected):
                radius = (
                    (grid_xi - source['xi'])**2
                    +
                    (grid_eta - source['eta'])**2
                )**0.5
                axis.add_artist(
                    pyplot.Circle(
                        (grid_xi, grid_eta),
                        radius,
                        facecolor='grey',
                        edgecolor='none'
                    )
                )

            axis.plot(grid[:, 0], grid[:, 1], 'ok')
            axis.plot(rejected['xi'], rejected['eta'], 'rx', markersize=3)

            axis.plot(allowed['xi'], allowed['eta'], 'b.', markersize=3)

            axis.plot(selected['xi'],
                      selected['eta'],
                      'g+',
                      markersize=5,
                      markeredgewidth=4)
            pyplot.savefig(plot_fname)
            pyplot.cla()

        def create_mag_rms_plot(selected,
                                allowed,
                                rejected,
                                phot_index,
                                **fname_substitutions):
            """Create a plot of RMS vs magnitude showing template selection."""

            fname_substitutions['plot_id'] = 'mag_rms'
            plot_fname = plot_fname_pattern % dict(
                phot_index=phot_index,
                **fname_substitutions
            )
            pyplot.semilogy(rejected['mag'],
                            rejected['rms'][:, phot_index],
                            'rx')
            pyplot.semilogy(allowed['mag'],
                            allowed['rms'][:, phot_index],
                            'b.')
            pyplot.semilogy(selected['mag'],
                            selected['rms'][:, phot_index],
                            'g+')
            pyplot.ylim(1e-3, 0.1)
            pyplot.savefig(plot_fname)
            pyplot.cla()

        def create_mag_nobs_plot(selected,
                                 allowed,
                                 rejected,
                                 phot_index,
                                 **fname_substitutions):
            """Create a plot of number of observations vs magnitude."""

            fname_substitutions['plot_id'] = 'mag_nobs'
            plot_fname = plot_fname_pattern % dict(
                phot_index=phot_index,
                **fname_substitutions
            )
            pyplot.plot(rejected['mag'],
                        rejected['num_finite'][:, phot_index],
                        'rx')
            pyplot.plot(allowed['mag'],
                        allowed['num_finite'][:, phot_index],
                        'b.')
            pyplot.plot(selected['mag'],
                        selected['num_finite'][:, phot_index],
                        'g+')
            pyplot.savefig(plot_fname)
            pyplot.cla()

        for phot_index, phot_template_indices in enumerate(template_indices):
            plot_args = (
                split_epd_statistics(phot_template_indices, phot_index)
                +
                (phot_index,)
            )
            create_xi_eta_plot(*plot_args)
            create_mag_rms_plot(*plot_args)
            create_mag_nobs_plot(*plot_args)


    def _select_template_stars(self, epd_statistics):
        """
        Select the stars tha will be used as TFA templates.

        The algorithm is as follows:

            1. Select all stars that are either saturated (see
               `saturation_magnitude` configuration) or are close to the typical
               RMS vs magnitude dependence (see `mag_rms_outlier_threshold`
               configuration).

            2. Remove all stars with RMS exceeding a specified values (see
               `max_rms` configuration) or fanter than a specified magnitude
               (see `faint_mag_limit` configuration) or do not containing
               sufficient number of data points in their LC (see
               `min_observation_factor` configuration).

            3. Create a uniform grid of points spanning the xi and eta region
               covered by the stars selected by steps 1 and 2 above and select
               the closest star to each point (see `sqrt_num_templates`
               configuration).

        The HATSouth TFA uses the following values:
            saturation_magnitude: 8
            mag_rms_outlier_threshold: 6
            max_rms: 0.15
            faint_mag_limit: 11.5
            min_observations: median of the number of observations for all LCs

        Args:
            epd_statistics:    See __init__().

        Returns:
            [scipy.array(shape=(num_template_stars,), dtype=scipy.int)]:
                A list of sorted 1-D arrays of indices within epd_statistics
                identifying stars selected to serve as templates for each
                photometry method.
        """

        def select_typical_rms_stars(not_saturated):
            """Select unsaturated stars with "typical" rms for their mag."""

            predictors = scipy.empty(
                (
                    self._configuration.mag_rms_dependence_order + 1,
                    not_saturated.sum()
                ),
                dtype=scipy.float64
            )
            predictors[0, :] = 1
            for mag_order in range(1, predictors.shape[0]):
                predictors[mag_order, :] = (
                    predictors[mag_order - 1, :]
                    *
                    epd_statistics['mag'][not_saturated]
                )

            num_photometries = epd_statistics['rms'][0].size

            result = scipy.zeros(epd_statistics['rms'].shape, dtype=bool)

            for phot_index in range(num_photometries):
                finite = scipy.isfinite(epd_statistics['rms'][:, phot_index])
                in_fit = scipy.logical_and(not_saturated, finite)
                phot_predictors = predictors[:, finite[not_saturated]]
                fit_lg_rms = scipy.log10(
                    epd_statistics['rms'][in_fit, phot_index]
                )
                coefficients, max_excess_lg_rms = iterative_fit(
                    phot_predictors,
                    fit_lg_rms,
                    error_avg='nanmedian',
                    rej_level=self._configuration.mag_rms_outlier_threshold,
                    max_rej_iter=self._configuration.mag_rms_max_rej_iter,
                    fit_identifier='EPD RMS vs mag'
                )[:2]
                max_excess_lg_rms = (
                    self._configuration.mag_rms_outlier_threshold
                    *
                    scipy.sqrt(max_excess_lg_rms)
                )
                excess_lg_rms = fit_lg_rms - scipy.dot(coefficients,
                                                       phot_predictors)
                result[in_fit, phot_index] = (excess_lg_rms < max_excess_lg_rms)

            return result


        def select_template_stars(allowed_stars):
            """Select TFA template stars from the set of allowed ones."""

            print('RMS fit left %d allowed stars.' % allowed_stars.sum())
            #False positive
            #pylint: disable=not-callable
            tree = cKDTree(
                data=scipy.stack(
                    (
                        epd_statistics['xi'][allowed_stars],
                        epd_statistics['eta'][allowed_stars]
                    )
                ).transpose()
            )
            #pylint: enable=not-callable
            template_indices = scipy.unique(
                tree.query(
                    self.get_xi_eta_grid(epd_statistics[allowed_stars])
                )[1]
            )
            print('Template indices: ' + repr(template_indices))
            return scipy.nonzero(allowed_stars)[0][template_indices]

        saturated = (epd_statistics['mag']
                     <
                     self._configuration.saturation_magnitude)
        min_observations = scipy.quantile(
            epd_statistics['num_finite'],
            self._configuration.min_observations_quantile
        )

        acceptable_rms = select_typical_rms_stars(
            scipy.logical_not(saturated)
        )

        bright_and_long_lc = scipy.logical_and(
            (
                epd_statistics['mag']
                <
                self._configuration.faint_mag_limit
            )[:, None],
            epd_statistics['num_finite'] >= min_observations
        )
        if self._configuration.allow_saturated_templates:
            acceptable_rms = scipy.logical_or(
                saturated[:, None],
                acceptable_rms
            )
        else:
            acceptable_rms = scipy.logical_and(
                scipy.logical_not(saturated)[:, None],
                acceptable_rms
            )
        acceptable_rms = scipy.logical_and(
            epd_statistics['rms'] < self._configuration.max_rms,
            acceptable_rms
        )

        print('Requiring at least %d observations' % min_observations)

        print(
            'There are %s non-faint stars with sufficient observations'
            %
            bright_and_long_lc.sum(0)
        )

        print(
            'There are %s stars with low RMS'
            %
            acceptable_rms.sum(0)
        )

        allowed_stars = scipy.logical_and(bright_and_long_lc, acceptable_rms)

        print('There are %s overlap stars' % allowed_stars.sum(0))

        num_photometries = epd_statistics['rms'][0].size

        result = []

        for photometry_index in range(num_photometries):
            result.append(
                select_template_stars(
                    allowed_stars[:, photometry_index]
                )
            )

        if getattr(self._configuration, 'selected_plots', None) is not None:
            self._plot_template_selection(epd_statistics,
                                          result,
                                          allowed_stars,
                                          self._configuration.selected_plots)

        return result

    def _get_observation_ids(self, light_curve, substitutions):
        """Return the observation IDs from the given light curve."""

        return light_curve.read_data_array({
            str(i): (dset_key, substitutions)
            for i, dset_key in enumerate(self._configuration.observation_id)
        })

    def get_fit_points(self, light_curve):
        """Return a bool array indicating which points from LC to use in fit."""

        return Evaluator(
            light_curve.read_data_array(
                self._configuration.fit_points_filter_variables
            )
        )(
            self._configuration.fit_points_filter_expression
        )

    def read_template_data(self, light_curve, phot_dset_key, substitutions):
        """Read the data for a single photometry method in a template LC."""


        phot_data = light_curve.get_dataset(phot_dset_key, **substitutions)
        phot_observation_ids = self._get_observation_ids(light_curve,
                                                         substitutions)
        assert phot_data.shape == phot_observation_ids.shape

        selected_points = scipy.isfinite(phot_data)

        if self._configuration.fit_points_filter_expression is not None:
            selected_points = scipy.logical_and(
                selected_points,
                self.get_fit_points(light_curve)
            )

        assert selected_points.shape == phot_data.shape
        phot_data = phot_data[selected_points]
        phot_observation_ids = phot_observation_ids[selected_points]

        return phot_data, phot_observation_ids

    #Organized into pieces as much as I could figure out how to.
    #pylint: disable=too-many-locals
    def _prepare_template_data(self, epd_statistics, max_padding_factor=1.1):
        """
        Organize the template star data into predictors and observation IDs.

        Args:
            epd_statistics:    See __init__().

            max_padding_factor(float):    Because light curves are read
                sequentially, at early times the ultimate number of observations
                is unknown. Any time observations exceed the size of the
                currently allocated array for the result, the result is resized
                to to accomodate this factor times the newly required length of
                observations, hopefully avoiding too many resize operations.

        Returns:
            [scipy.array]:
                The source IDs of the stars selected to serve as templates for
                each photometry method.

            [scipy.array]:
                The brightness measurements from all the templates for all the
                photometry methods at all observation points where at least one
                template has a measurement. Entries at observation points not
                represented in a template are zero. The shape of each array is:
                (
                    number template stars
                    number observations
                ), and there are number of photemetry methods arrays in the
                list.

            [scipy.array]:
                The sorted observation IDs for which at least one template has a
                measurement for each photometry method.
        """

        def initialize_result(phot_data,
                              phot_observation_ids,
                              num_templates):
            """Create the result arrays for the first time."""

            template_measurements = scipy.zeros(
                shape=(round(phot_data.size * max_padding_factor),
                       num_templates),
                dtype=phot_data.dtype
            )

            sorting_indices = scipy.argsort(phot_observation_ids)

            template_measurements[:phot_data.size, 0] = phot_data[
                sorting_indices
            ]

            template_observation_ids = phot_observation_ids[sorting_indices]

            return template_measurements, template_observation_ids

        def add_to_result(*,
                          phot_data,
                          phot_observation_ids,
                          source_index,
                          template_measurements,
                          template_observation_ids):
            """Add a single template/photometry combination to result arrays."""

            phot_destination_ind = scipy.searchsorted(
                template_observation_ids,
                phot_observation_ids
            )
            matched_observations = (
                template_observation_ids[
                    scipy.minimum(
                        phot_destination_ind,
                        template_observation_ids.size - 1
                    )
                ]
                ==
                phot_observation_ids
            )
            template_measurements[
                phot_destination_ind[matched_observations],
                source_index,
            ] = phot_data[matched_observations]

            unmatched_observations = scipy.logical_not(matched_observations)

            if not unmatched_observations.any():
                return template_measurements, template_observation_ids

            observations_to_append = phot_observation_ids[
                unmatched_observations
            ]
            phot_to_append = phot_data[unmatched_observations]

            combined_observation_ids = scipy.union1d(
                template_observation_ids,
                observations_to_append
            )

            if template_observation_ids.size < combined_observation_ids.size:
                print('Resizing template data from shape %s to shape %s'
                      %
                      (
                          template_measurements.shape,
                          (
                              (combined_observation_ids.size,)
                              +
                              template_measurements.shape[1:]
                          )
                      ))
                template_measurements = scipy.resize(
                    template_measurements,
                    (
                        (combined_observation_ids.size,)
                        +
                        template_measurements.shape[1:]
                    )
                )

            print('Template observation IDs size: '
                  +
                  repr(template_observation_ids.size))
            print('Min and max old indices: %s, %s'
                  %
                  (
                      repr(
                          scipy.searchsorted(combined_observation_ids,
                                             template_observation_ids).min()
                      ),
                      repr(
                          scipy.searchsorted(combined_observation_ids,
                                             template_observation_ids).max()
                      )
                  ))
            print(
                'Destinations: '
                +
                repr(
                    scipy.searchsorted(combined_observation_ids,
                                       template_observation_ids)
                )
            )
            print('Values at destination: '
                  +
                  repr(
                      template_measurements[
                          scipy.searchsorted(combined_observation_ids,
                                             template_observation_ids),
                          :
                      ]
                  ))
            print(
                'Values to set: '
                +
                repr(template_measurements[:template_observation_ids.size, :])
            )
            template_measurements[
                scipy.searchsorted(combined_observation_ids,
                                   template_observation_ids),
                :
            ] = template_measurements[:template_observation_ids.size, :]
            new_indices = scipy.searchsorted(combined_observation_ids,
                                             observations_to_append)
            template_measurements[new_indices, :] = 0.0
            template_measurements[new_indices, source_index] = phot_to_append

            return template_measurements, combined_observation_ids

        template_stars = (
            epd_statistics['ID'][phot_source_indices]
            for phot_source_indices in self._select_template_stars(epd_statistics)
        )

        template_measurements = []
        template_observation_ids = []

        for (phot_template_stars, fit_dataset) in zip(
                template_stars,
                self._configuration.fit_datasets
        ):
            phot_template_measurements = None
            num_templates = phot_template_stars.shape[0]
            for source_index, source_id in enumerate(phot_template_stars):
                with LightCurveFile(
                        self._configuration.lc_fname_pattern % tuple(source_id),
                        'r'
                ) as light_curve:
                    phot_data, phot_observation_ids = self.read_template_data(
                        light_curve,
                        *fit_dataset[:2]
                    )
                    if phot_template_measurements is None:
                        (
                            phot_template_measurements,
                            phot_template_observation_ids
                        ) = initialize_result(phot_data,
                                              phot_observation_ids,
                                              num_templates)
                    else:
                        (
                            phot_template_measurements,
                            phot_template_observation_ids
                        ) = add_to_result(
                            phot_data=phot_data,
                            phot_observation_ids=phot_observation_ids,
                            source_index=source_index,
                            template_measurements=phot_template_measurements,
                            template_observation_ids=(
                                phot_template_observation_ids
                            )
                        )
            phot_template_measurements.resize(
                (phot_template_observation_ids.size,)
                +
                phot_template_measurements.shape[1:]
            )

            template_measurements.append(phot_template_measurements)
            template_observation_ids.append(phot_template_observation_ids)

        return (
            template_stars,
            template_measurements,
            template_observation_ids
        )
    #pylint: enable=too-many-locals

    def _verify_template_data(self):
        """Assert that template data is correctly organized."""

        for (
                phot_source_ids,
                phot_template_data,
                phot_template_observation_ids,
                fit_dataset
        ) in zip(
            self._template_source_ids,
            self.template_measurements,
            self._template_observation_ids,
            self._configuration.fit_datasets
        ):
            for source_id, template_data, template_observation_ids in zip(
                    phot_source_ids,
                    phot_template_data.T,
                    phot_template_observation_ids
            ):
                template_selection = (template_data != 0.0)
                with LightCurveFile(
                        self._configuration.lc_fname_pattern % tuple(source_id),
                        'r'
                ) as light_curve:
                    lc_data, lc_observation_ids = self.read_template_data(
                        light_curve,
                        *fit_dataset[:2]
                    )
                    lc_selection = (lc_data != 0.0)
                    assert (
                        template_data[template_selection]
                        ==
                        lc_data[lc_selection]
                    )
                    assert (
                        template_observation_ids[template_selection]
                        ==
                        lc_observation_ids[lc_selection]
                    )

    def __init__(self,
                 epd_statistics,
                 configuration,
                 verify_template_data=False,
                 **iterative_fit_config):
        """
        Get ready to apply TFA corrections.

        Args:
            epd_statistics(scipy structured array):    An array containing
                information about the input sources and summary statistics for
                their EPD fit. The array must contain the following fields:

                ID ((scipy.int, #)):
                    Array if integers uniquely identifying the source (see
                    DataReductionFile.get_source_data for more info.

                mag (scipy.float64):
                    The magnitude of the source per the catalogue in the band
                    most approximating the observations.

                xi(scipy.float64):
                    The pre-projected `xi` coordinate of the source from the
                    catalogue.

                eta(scipy.float64):
                    The pre-projected `eta` coordinate of the source from the
                    catalogue.

                rms ((scipy.float64, #)):
                    Array of the RMS residuals of the EPD fit for each source
                    for each photometry method.

                num_finite((scipy.uint, #)):
                    Array of the number of finite observations with EPD
                    corrections.

            configuration:    An object with attributes specifying how TFA
                should be done. At least the following attributes must be
                defined (extra ones are ignored):

                saturation_magnitude
                    The magnitude at which sources start to saturate. See
                    _select_template_stars()

                allow_saturated_templates
                    Whether saturated stars should be represented in the
                    templates.

                mag_rms_dependence_order
                    The maximum order of magnitude to include in the fit for
                    typical rms vs magnitude.

                mag_rms_outlier_threshold
                    Stars are not allowed to be in the template if their RMS is
                    more than this many sigma away from the mag-rms fit. This is
                    also the threshold used for rejecting outliers when doing
                    the iterative fit for the rms as a function of magnutude.

                mag_rms_max_rej_iter
                    The maximum number of rejection fit iterations to do when
                    deriving the rms(mag) dependence.

                max_rms
                    Stars are allowed to be in the template only if their RMS is
                    no larger than this.

                faint_mag_limit
                    Stars fainter than this cannot be template stars.

                min_observations_quantile
                    The minimum number of observations required of template
                    stars is this quantile among the input collection of stars.

                sqrt_num_templates
                    The number of template stars is the square of this number.

                observation_id
                    The datasets to use for matching observations across light
                    curves. For example, the following works for HAT:
                    ```
                    (
                        'fitseader.cfg.stid',
                        'fitsheader.cfg.cmpos',
                        'fitsheader.fnum'
                    )
                    ```.

                lc_fname_pattern
                    A %-substitution pattern that expands to the filename of a
                    lightcurve given a source ID.

                fit_datasets
                    See same name argument to EPDCorrection.__init__().

                fit_points_filter_varibales
                    See used_variables argument to EPDCorrection.__init__(). In
                    this case only required to evaluate
                    fit_points_filter_expression.

                fit_points_filter_expression
                    See same name argument to EPDCorrection.__init__().

                [selected_plots] (str):    Optional template for naming plots
                    showing the template selection in action. If not specified,
                    no such plots are generated. Should include %(plot_id)s and
                    %(phot_index)d substitutions.

            verify_template_data(bool):    If True a series of assert statements
                are issued to check that the photometry data is correctly
                matched to observation IDs. Only useful for debugging.

            iterative_fit_config:    Any other arguments to pass directly to
                iterative_fit_qr().

        Returns:
            None
        """

        super().__init__(configuration.fit_datasets, **iterative_fit_config)

        self._configuration = configuration

        self._io_configurations = self._get_io_configurations()

        assert (epd_statistics['rms'][0].size
                ==
                len(configuration.fit_datasets))

        assert (epd_statistics['num_finite'][0].size
                ==
                len(configuration.fit_datasets))

        (
            self._template_source_ids,
            self.template_measurements,
            self._template_observation_ids
        ) = self._prepare_template_data(epd_statistics)

        if verify_template_data:
            self._verify_template_data()

        #False positive
        #pylint: disable=unexpected-keyword-arg
        self._template_qrp = [
            scipy.linalg.qr(template_measurements,
                            mode='economic',
                            pivoting=True)
            for template_measurements in self.template_measurements
        ]
        #pylint: enable=unexpected-keyword-arg

    def __call__(self,
                 lc_fname,
                 get_fit_data=LightCurveFile.get_dataset,
                 extra_predictors=None,
                 save=True):

        """
        Apply TFA to the given LC, optionally protecting an expected signal.
        """

        def correct_one_dataset(light_curve,
                                fit_target,
                                fit_index,
                                result):
            """
            Calculate and apply TFA correction to a single dataset.

            Args:
                light_curve(LightCurveFile):    The opened for writing light
                    curve to apply EPD corrections to.

                fit_target((str, dict)):    The dataset key and substitutions
                    identifying a uniquedataset in the lightcurve to fit.

                fit_index(int):    The index of the dataset being fit within the
                    list of datasets that will be fit for this lightcurve.

                result:    The result variable for the parent update for this
                    fit.

            Returns:
                None
            """

            raw_values = get_fit_data(light_curve,
                                      fit_target[0],
                                      **fit_target[1])
            lc_observation_ids = self._get_observation_ids(light_curve,
                                                           fit_target[1])
            if isinstance(raw_values, tuple):
                raw_values, fit_data = raw_values
            else:
                fit_data = raw_values

            matched_indices = scipy.searchsorted(
                self._template_observation_ids[fit_index],
                lc_observation_ids
            )
            fit_points = (
                lc_observation_ids
                ==
                self._template_observation_ids[fit_index][matched_indices]
            )
            if self._configuration.fit_points_filter_expression is not None:
                fit_points = scipy.logical_and(
                    fit_points,
                    self.get_fit_points(light_curve)
                )
            matched_indices = matched_indices[fit_points]

            matched_fit_data = scipy.full(
                self._template_observation_ids[fit_index].shape,
                scipy.nan
            )
            matched_fit_data[matched_indices] = fit_data[fit_points]

            #Error average specified through iterative_fit_config
            #pylint: disable=missing-kwoa
            fit_results = iterative_fit_qr(
                self.template_measurements[fit_index].T,
                self._template_qrp[fit_index],
                matched_fit_data,
                **self.iterative_fit_config
            )
            #pylint: enable=missing-kwoa
            fit_results = self._process_fit(
                fit_results=fit_results,
                raw_values=raw_values[fit_points],
                predictors=self.template_measurements[
                    fit_index
                ][
                    matched_indices,
                    :
                ].T,
                fit_index=fit_index,
                result=result,
                num_extra_predictors=0
            )
            if save:
                self._save_result(
                    fit_index=fit_index,
                    configuration=self._io_configurations[fit_index],
                    **fit_results,
                    fit_points=fit_points,
                    light_curve=light_curve
                )

        if extra_predictors is not None:
            raise NotImplementedError(
                'Adding extra templates is not implemented yet.'
            )

        with LightCurveFile(lc_fname, 'r+') as light_curve:
            result = scipy.empty(
                1,
                dtype=self.get_result_dtype(len(self.fit_datasets),
                                            extra_predictors)
            )

            for fit_index, fit_target in enumerate(self.fit_datasets):
                correct_one_dataset(
                    light_curve,
                    fit_target,
                    fit_index,
                    result
                )

        return result
