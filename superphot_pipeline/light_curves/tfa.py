"""Define aclass for applying TFA corrections to lightcurves."""

from matplotlib import pyplot

import scipy
#pylint false positive
#pylint: disable=no-name-in-module
from scipy.spatial import cKDTree
#pylint: enable=no-name-in-module

from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.fit_expression import iterative_fit
from .light_curve_file import LightCurveFile

class TFA:
    """
    Class for performing TFA corrections to a set of lightcurves.

    Attributes:
        _configuration:    An object with attributes configuring how TFA is to
            be done (see `configuration` argument to __init__()).
    """

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

        def create_xi_eta_plot(phot_template_indices,
                               phot_index,
                               **fname_substitutions):
            """Create a plot of projected catalogue positions."""

            fname_substitutions['plot_id'] = 'xi_eta'
            plot_fname = plot_fname_pattern % dict(
                phot_index=phot_index,
                **fname_substitutions
            )
            selected = epd_statistics[phot_template_indices]

            axis = pyplot.gca()

            grid = self.get_xi_eta_grid(epd_statistics)

            for (grid_xi, grid_eta), source in zip(grid, selected):
                radius = (
                    (grid_xi - source['xi'])**2
                    +
                    (grid_eta - source['eta'])**2
                )**0.5
                if radius < 0.3:
                    axis.add_artist(
                        pyplot.Circle(
                            (grid_xi, grid_eta),
                            radius,
                            facecolor='grey',
                            edgecolor='none'
                        )
                    )

            rejected = scipy.ones(epd_statistics.shape, dtype=bool)
            rejected[allowed_stars[:, phot_index]] = False
            rejected = epd_statistics[rejected]
            axis.plot(rejected['xi'], rejected['eta'], 'rx', markersize=1)

            allowed = epd_statistics[allowed_stars[:, phot_index]]
            axis.plot(allowed['xi'], allowed['eta'], 'b.', markersize=1)

            axis.plot(selected['xi'], selected['eta'], 'g+', markersize=1)
            pyplot.savefig(plot_fname)
            pyplot.cla()

        for phot_index, phot_template_indices in enumerate(template_indices):
            create_xi_eta_plot(phot_template_indices, phot_index=phot_index)


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
            scipy.array(shape=(num_photometries, num_template_stars),
                        dtype=scipy.int):
                A 2-D array of indices within epd_statistics identifying stars
                selected to serve as templates for each photometry method.
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
                predictors[mag_order, :] = (predictors[mag_order - 1, :]
                                            *
                                            epd_statistics['mag'][not_saturated])

            num_photometries = epd_statistics['rms'][0].size

            result = scipy.empty(epd_statistics['rms'].shape, dtype=bool)

            for phot_index in range(num_photometries):
                finite = scipy.isfinite(epd_statistics['rms'][:, phot_index])
                phot_predictors = predictors[:, finite[not_saturated]]
                fit_rms = (
                    epd_statistics['rms'][
                        scipy.logical_and(not_saturated, finite),
                        phot_index
                    ]
                )
                coefficients = iterative_fit(
                    phot_predictors,
                    fit_rms,
                    error_avg='nanmedian',
                    rej_level=self._configuration.mag_rms_outlier_threshold,
                    max_rej_iter=self._configuration.mag_rms_max_rej_iter,
                    fit_identifier='EPD RMS vs mag'
                )[0]
                excess_rms = fit_rms - scipy.dot(coefficients, phot_predictors)
                max_rms = (self._configuration.mag_rms_outlier_threshold
                           *
                           scipy.std(excess_rms))
                result[:, phot_index] = scipy.logical_and(
                    not_saturated,
                    epd_statistics['rms'][:, phot_index] < max_rms
                )

            return result


        def select_template_stars(allowed_stars):
            """Select TFA template stars from the set of allowed ones."""

            tree = cKDTree(
                data=scipy.stack(
                    (
                        epd_statistics['xi'][allowed_stars],
                        epd_statistics['eta'][allowed_stars]
                    )
                ).transpose()
            )
            template_indices = tree.query(
                self.get_xi_eta_grid(epd_statistics)
            )[1]
            print('Template indices: ' + repr(template_indices))
            return scipy.nonzero(allowed_stars)[0][template_indices]

        saturated = (epd_statistics['mag']
                     <
                     self._configuration.saturation_magnitude)
        min_observations = scipy.quantile(
            epd_statistics['num_finite'],
            self._configuration.min_observations_quantile
        )

        allowed_stars = scipy.logical_and(
            scipy.logical_and(
                (
                    epd_statistics['mag']
                    <
                    self._configuration.faint_mag_limit
                )[:, None],
                epd_statistics['num_finite'] > min_observations
            ),
            scipy.logical_and(
                epd_statistics['rms'] < self._configuration.max_rms,
                scipy.logical_or(
                    saturated[:, None],
                    select_typical_rms_stars(scipy.logical_not(saturated))
                )
            )
        )

        num_photometries = epd_statistics['rms'][0].size
        sqrt_num_templates = self._configuration.sqrt_num_templates**2

        result = scipy.empty(shape=(num_photometries, sqrt_num_templates),
                             dtype=scipy.int_)
        for photometry_index in range(num_photometries):
            result[photometry_index] = select_template_stars(
                allowed_stars[:, photometry_index]
            )

        if hasattr(self._configuration, 'selected_plots'):
            self._plot_template_selection(epd_statistics,
                                          result,
                                          allowed_stars,
                                          self._configuration.selected_plots)

        return result

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
            numpy.array:
                The brightness measurements from all the templates for all the
                photometry methods at all observatin points where at least one
                template has a measurement. Entries at observation points not
                represented in a template are zero. The shape of the array is:
                (
                    number photemetry methods,
                    number template stars,
                    number observations
                ).

            numpy.array:
                The sorted observation IDs for which at least one template has a
                measurement.
        """

        def read_template_data(light_curve, phot_dset_key, substitutions):
            """Read the data for a single photometry method in a template LC."""


            phot_data = light_curve.get_dataset(phot_dset_key, **substitutions)
            phot_observation_ids = light_curve.read_data_array({
                str(i): (dset_key, substitutions)
                for i, dset_key in enumerate(self._configuration.observation_id)
            })
            assert phot_data.shape == phot_observation_ids.shape

            if self._configuration.fit_points_filter_variables is not None:
                selected_points = Evaluator(
                    light_curve.read_data_array(
                        self._configuration.fit_points_filter_variables
                    )
                )(
                    self._configuration.fit_points_filter_variables
                )
                assert selected_points.shape == phot_data.shape
                phot_data = phot_data[selected_points]
                phot_observation_ids = phot_observation_ids[selected_points]

            return phot_data, phot_observation_ids

        def initialize_result(phot_data,
                              phot_observation_ids,
                              num_photometries,
                              num_templates):
            """Create the result arrays for the first time."""

            template_measurements = scipy.zeros(
                shape=(phot_data.size * max_padding_factor,
                       num_templates,
                       num_photometries),
                dtype=phot_data.dtype
            )

            sorting_indices = scipy.argsort(phot_observation_ids)

            template_measurements[:phot_data.size, 0, 0] = phot_data[
                sorting_indices
            ]

            template_observation_ids = phot_observation_ids[sorting_indices]

            return template_measurements, template_observation_ids

        def add_to_result(*,
                          phot_data,
                          phot_observation_ids,
                          phot_index,
                          source_index,
                          template_measurements,
                          template_observation_ids):
            """Add a single template/photometry combination to result arrays."""

            phot_destination_ind = scipy.searchsorted(
                template_observation_ids[:num_observations],
                phot_observation_ids
            )
            matched_observations = (
                template_observation_ids[phot_destination_ind]
                ==
                phot_observation_ids
            )
            template_measurements[
                phot_destination_ind[matched_observations],
                source_index,
                phot_index,
            ] = phot_data[matched_observations]

            unmatched_observations = scipy.logical_not(matched_observations)

            if not unmatched_observations.any():
                return

            observations_to_append = phot_observation_ids[
                unmatched_observations
            ]
            phot_to_append = phot_data[unmatched_observations]

            combined_observation_ids = scipy.union1d(
                template_observation_ids,
                observations_to_append
            )

            if template_observation_ids.size < combined_observation_ids.size:
                template_measurements.resize(
                    (combined_observation_ids.size,)
                    +
                    template_measurements.shape[1:]
                )

            template_measurements[
                scipy.searchsorted(combined_observation_ids,
                                   template_observation_ids),
                :,
                :
            ] = template_measurements
            template_measurements[
                scipy.searchsorted(combined_observation_ids,
                                   observations_to_append),
                :,
                :
            ] = phot_to_append

        template_star_indices = self._select_template_stars(epd_statistics)


        num_photometries, num_templates = template_star_indices.shape

        template_measurements = None
        for (
                phot_index,
                (
                    phot_source_indices,
                    (
                        phot_dset_key,
                        substitutions
                    )
                )
        ) in enumerate(
            zip(template_star_indices,
                self._configuration.fit_datasets)
        ):
            for source_index, source_id in enumerate(
                    epd_statistics['ID'][phot_source_indices]
            ):
                with LightCurveFile(
                        self._configuration.lc_fname_pattern % source_id,
                        'r'
                ) as light_curve:
                    phot_data, phot_observation_ids = read_template_data(
                        light_curve,
                        phot_dset_key,
                        substitutions
                    )
                    if template_measurements is None:
                        (
                            template_measurements,
                            template_observation_ids
                        ) = initialize_result(phot_data,
                                              phot_observation_ids,
                                              num_photometries,
                                              num_templates)
                        num_observations = phot_data.size
                    else:
                        add_to_result(
                            phot_data=phot_data,
                            phot_observation_ids=phot_observation_ids,
                            phot_index=phot_index,
                            source_index=source_index,
                            template_measurements=template_measurements,
                            template_observation_ids=template_observation_ids
                        )
        template_measurements.resize(
            (phot_observation_ids.size,)
            +
            template_measurements.shape[1:]
        )
        return template_measurements.transpose(), phot_observation_ids
    #pylint: enable=too-many-locals

    def __init__(self, epd_statistics, configuration):
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

                mag_rms_dependence_order
                    The maximum order of magnitude to include in the fit for
                    typical rms vs magnitude.

                mag_rms_outlier_threshold
                    Stars are allowed to be in the template if their RMS in more
                    than this many sigma away from the mag-rms fit. This is also
                    the threshold used for rejecting outliers when doing the
                    iterative fit for the rms as a function of magnutude.

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
                    no such plots are generated.

        Returns:
            None
        """

        self._configuration = configuration

        assert (epd_statistics['rms'][0].size
                ==
                len(configuration.fit_datasets))

        assert (epd_statistics['num_finite'][0].size
                ==
                len(configuration.fit_datasets))

        self._template_data = self._prepare_template_data(epd_statistics)
