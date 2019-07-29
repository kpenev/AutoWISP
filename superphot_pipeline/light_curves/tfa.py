"""Define aclass for applying TFA corrections to lightcurves."""

import scipy
from scipy.spatial import cKDTree

from superphot_pipeline.fit_expression import iterative_fit
from .lc_data_io import _config_dset_key_rex
from .light_curve_file import LightCurveFile

class TFA:
    """
    Class for performing TFA corrections to a set of lightcurves.

    Attributes:
        _configuration:    An object with attributes configuring how TFA is to
            be done (see `configuration` argument to __init__()).
    """
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
                        dtype=(scipy.int, #)):
                A 2-D array of source IDs identifying stars selected to serve as
                templates for each photometry method.
        """

        def select_typcial_rms_stars(not_saturated):
            """Select unsaturated stars with "typical" rms for their mag."""

            predictors = scipy.empty(
                (
                    not_saturated.sum(),
                    self._configuration.mag_rms_dependence_order
                ),
                dtype=scipy.float64
            )
            predictors[:, 0] = 1
            for mag_order in range(1, predictors.shape[0]):
                predictors[:, mag_order] = (predictors[:, mag_order - 1]
                                            *
                                            epd_statistics['mag'][not_saturated])

            not_saturated_rms = epd_statistics['epd_rms'][not_saturated]
            coefficients = iterative_fit(
                predictors,
                not_saturated_rms,
                error_avg='nanmedian',
                rej_level=self._configuration.mag_rms_outlier_threshold,
                max_rej_iter=self._configuration.mag_rms_max_rej_iter,
                fit_identifier='EPD RMS vs mag'
            )[0]
            excess_rms = not_saturated_rms - scipy.dot(coefficients, predictors)
            max_rms = (self._configuration.mag_rms_outlier_threshold
                       *
                       scipy.std(excess_rms))
            return scipy.logical_and(not_saturated[:, None],
                                     epd_statistics['epd_rms'] < max_rms)


        def select_template_stars(allowed_stars):
            """Select TFA template stars from the set of allowed ones."""

            min_xi = epd_statistics['xi'].min()
            max_xi = epd_statistics['xi'].max()
            min_eta = epd_statistics['eta'].min()
            max_eta = epd_statistics['eta'].max()
            grid_res = self._configuration.sqrt_num_templates
            query_points = scipy.mgrid[
                min_xi : max_xi : grid_res * 1j,
                min_eta : max_eta : grid_res * 1j
            ].transpose().reshape(grid_res * grid_res, 2)
            tree = cKDTree(
                data=scipy.stack(
                    (
                        epd_statistics['xi'][allowed_stars],
                        epd_statistics['eta'][allowed_stars]
                    )
                ).transpose()
            )
            template_indices = tree.query(query_points)
            return epd_statistics['ID'][allowed_stars][template_indices]

        saturated = (epd_statistics['mag']
                     <
                     self._configuration.saturation_magnitude)
        min_observations = scipy.quantile(
            epd_statistics['num_finite_epd'],
            self._configuration.min_observations_quantile
        )

        allowed_stars = scipy.logical_and(
            scipy.logical_and(
                (
                    epd_statistics['mag']
                    <
                    self._configuration.faint_mag_limit
                )[:, None],
                epd_statistics['num_finite_epd'] > min_observations
            ),
            scipy.logical_and(
                epd_statistics['epd_rms'] < self._configuration.max_rms,
                scipy.logical_or(
                    saturated[:, None],
                    select_typcial_rms_stars(scipy.logical_not(saturated))
                )
            )
        )

        id_dtype = (epd_statistics['ID'].dtype, epd_statistics['ID'].shape)
        num_photometries = epd_statistics['epd_rms'][0].size
        sqrt_num_templates = self._configuration.sqrt_num_templates**2

        result = scipy.empty(shape=(num_photometries, sqrt_num_templates),
                             dtype=id_dtype)
        for photometry_index in range(num_photometries):
            result[photometry_index] = select_template_stars(
                allowed_stars[:, photometry_index]
            )

        return result

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

        template_stars = self._select_template_stars(epd_statistics)

        num_photometries, num_templates = template_stars.shape()[:2]

        template_measurements = None
        for phot_index, phot_source_ids in enumerate(template_stars):
            for source_id in phot_source_ids:
                with LightCurveFile(
                        self._configuration.lc_fname_pattern % source_source_id,
                        'r'
                ) as light_curve:
                    if (
                            self._configuration.fit_points_filter_variables
                            is not None
                    ):
                        fit_points = Evaluator(
                            light_curve.read_data_array(
                                self._configuration.fit_points_filter_variables
                            )
                        )(
                            self._configuration.fit_points_filter_variables
                        )
        template_measurements = scipy.empty(
            shape=(num_photometries, num_templates, 0),
            dtype=scipy.float64
        )

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

                epd_rms ((scipy.float64, #)):
                    Array of the RMS residuals of the EPD fit for each source
                    for each photometry method.

                num_finite_epd((scipy.uint, #)):
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
                    (fitseader.cfg.stid, fitsheader.cfg.cmpos, fitsheader.fnum)
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

        Returns:
            None
        """

        self._configuration = configuration

        assert (epd_statistics['epd_rms'][0].size
                ==
                len(configuration.fit_datasets))

        assert (epd_statistics['num_finite_epd'][0].size
                ==
                len(configuration.fit_datasets)
