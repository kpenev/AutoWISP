"""Define class for performing EPD correction on lightcurves."""

from itertools import repeat

import scipy

from superphot_pipeline import LightCurveFile
from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.fit_expression import\
    Interface as FitTermsInterface,\
    iterative_fit

#Attempts to re-organize reduced readability
#pylint: disable=too-many-instance-attributes
#Using a class is justified.
#pylint: disable=too-few-public-methods
class EPDCorrection:
    """
    Class for deriving and applying EPD corrections to lightcurves.

    Attributes:
        used_variables:    See __init__.

        fit_points_filter_expression:    See __init__.

        fit_terms(FitTermsInterface):    Instance set-up to generate the
            independent variables matrix needed for the fits.

        fit_datasets:    See __init__.

        fit_weights:    See __init__.

        iterative_fit_config(dict):    Configuration to use for iterative
            fitting. See iterative_fit() for details.

        _fit_config([]):    A list storing the configuration used for the
            fitting. Formatter properly for passing as the only entry directly
            to LightCurveFile.add_configurations().
    """

    def _get_fit_configurations(self, fit_terms_expression):
        """Return the current fitting configurations (see self._fit_config)."""

        def format_substitutions(substitutions):
            """Return a string of `var=value` containing the given subs dict."""

            return '; '.join('%s = %s' % item for item in substitutions.items())

        fit_variables_str = '; '.join([
            '%s = %s (%s)'
            %
            (var_name, dataset_key, format_substitutions(substitutions))
            for var_name, (dataset_key,
                           substitutions) in self.used_variables.items()
        ])
        result = []
        for fit_target in (self.fit_datasets if self.fit_weights is None
                           else zip(self.fit_datasets, fit_weights)):

            if self.fit_weights is None:
                pipeline_key_prefix = fit_target[2]
                fit_weights = b''
            else:
                pipeline_key_prefix = fit_target[0][2]
                fit_weights = fit_target[1].encode('ascii')

            if self.fit_points_filter_expression is None:
                point_filter = b''
            else:
                point_filter = self.fit_points_filter_expression.encode('ascii')

            pipeline_key_prefix = (pipeline_key_prefix.rsplit('.', 1)[0]
                                   +
                                   '.cfg.')
            result.append(
                [
                    (
                        pipeline_key_prefix + 'variables',
                        fit_variables_str.encode('ascii')
                    ),
                    (
                        pipeline_key_prefix + 'fit_filter',
                        point_filter
                    ),
                    (
                        pipeline_key_prefix + 'fit_terms',
                        fit_terms_expression.encode('ascii')
                    ),
                    (
                        pipeline_key_prefix + 'fit_weights',
                        fit_weights
                    ),
                    (
                        pipeline_key_prefix + 'error_avg',
                        self.iterative_fit_config['error_avg'].encode('ascii')
                    )
                ]
                +
                [
                    (
                        pipeline_key_prefix + cfg_key,
                        self.iterative_fit_config[cfg_key]
                    )
                    for cfg_key in ['rej_level', 'max_rej_iter']
                ]
            )

        return result

    def _save_result(self,
                     *,
                     fit_index,
                     corrected_values,
                     fit_residual,
                     non_rejected_points,
                     fit_points,
                     light_curve):
        """Stores the fit results and configuration to the light curve."""

        original_key, substitutions, destination_key = (
            self.fit_datasets[fit_index]
        )

        light_curve.add_corrected_dataset(
            original_key=original_key,
            corrected_key=destination_key,
            corrected_values=corrected_values,
            corrected_selection=fit_points,
            **substitutions
        )
        config_key_prefix = destination_key.rsplit('.', 1)[0]


        configurations = self._fit_config[fit_index][:]
        configurations.extend([
            (config_key_prefix + '.fit_residual', fit_residual),
            (config_key_prefix + '.num_fit_points', non_rejected_points),
        ])
        light_curve.add_configurations(
            component=config_key_prefix,
            configurations=(tuple(configurations),),
            config_indices=scipy.zeros(shape=(corrected_values.size,),
                                       dtype=scipy.uint),
            config_index_selection=fit_points,
            **substitutions
        )

    def __init__(self,
                 *,
                 used_variables,
                 fit_points_filter_expression,
                 fit_terms_expression,
                 fit_datasets,
                 fit_weights=None,
                 **iterative_fit_config):
        """
        Configure the fitting.

        Args:
            used_variables(dict):    Keys are variables used in
                `fit_points_filter_expression` and `fit_terms_expression` and
                the corresponding values are 2-tuples of pipeline keys
                corresponding to each variable and an associated dictionary of
                path substitutions. Each entry defines a unique independent
                variable to use in the fit or based on which to select points to
                fit.

            fit_points_filter_expression(str):    An expression using
                `used_variables` which evalutes to either True or False
                indicating if a given point in the lightcurve should be fit and
                corrected.

            fit_terms_expression(str):    A fitting terms expression involving
                only variables from `used_variables` which expands to the
                various terms to use in a linear least squares EPD correction.

            fit_datasets([]):    A list of 3-tuples of pipeline keys
                corresponding to each variable identifying a dataset to fit and
                correct, an associated dictionary of path substitutions, and a
                pipeline key for the output dataset. Configurations of how the
                fitting was done and the resulting residual and non-rejected
                points are added to configuration datasets generated by removing
                the tail of the destination and adding `'.cfg.' + <parameter
                name>` for configurations and just `'.' + <parameter name>` for
                fitting statistics. For example, if the output dataset key is
                `'shapefit.epd.magnitude'`, the configuration datasets will look
                like `'shapefit.epd.cfg.fit_terms'`, and
                `'shapefit.epd.residual'`.

            fit_weights(str or [str]):    Weights to use when fitting each
                fit_dataset. Follows the same format as
                `fit_points_filter_expression`. Can be either a single
                expression, which is applied to all datasets or a list of
                expressions, one for each entry in `fit_datasets`.

            iterative_fit_config:    Any other arguments to pass directly to
                iterative_fit().

        Returns:
            None
        """

        self.used_variables = used_variables
        self.fit_points_filter_expression = fit_points_filter_expression
        self.fit_terms_expression = fit_terms_expression
        self.fit_datasets = fit_datasets
        self.fit_weights = fit_weights
        self.iterative_fit_config = iterative_fit_config

        self._fit_config = self._get_fit_configurations(fit_terms_expression)

    @staticmethod
    def get_result_dtype(num_photometries):
        """Return the data type for the result of __call__."""

        return [('ID', (scipy.int_, 3)),
                ('mag', scipy.float64),
                ('xi', scipy.float64),
                ('eta', scipy.float64),
                ('rms', (scipy.float64, num_photometries)),
                ('num_finite', (scipy.uint, num_photometries))]

    #No straightforward way to simplify.
    #pylint: disable=too-many-locals
    def __call__(self, lc_fname):
        """
        Fit and correct the given lightcurve.

        Args:
            lc_fname(str):    The filename of the light curve to fit.

        Returns:
            numpy.array(dtype=[('rms', numpy.float64),
                               ('num_finite', numpy.uint)]):
                The RMS of the corrected values and the number of finite points
                for each corrected dataset in the order in which the datasets
                were supplied to __init__().
        """

        with LightCurveFile(lc_fname, 'r+') as light_curve:
            evaluate = Evaluator(
                light_curve.read_data_array(self.used_variables)
            )

            predictors = FitTermsInterface(self.fit_terms_expression)(evaluate)

            fit_points = (
                evaluate(self.fit_points_filter_expression)
                if self.fit_points_filter_expression is not None else
                scipy.ones(predictors.shape[1], dtype=bool)
            )
            predictors = predictors[:, fit_points]

            if self.fit_weights is None:
                fit_weight_iter = repeat(None)
            elif isinstance(self.fit_weights, str):
                fit_weight_iter = repeat(evaluate(self.fit_weights)[fit_points])
            else:
                assert len(self.fit_datasets) == len(self.fit_weights)
                fit_weight_iter = (
                    evaluate(weight_expression)[fit_points]
                    for weight_expression in self.fit_weights
                )

            result = scipy.empty(
                1,
                dtype=self.get_result_dtype(len(self.fit_datasets))
            )

            for fit_index, (to_fit, fit_weights) in enumerate(
                    zip(self.fit_datasets,
                        fit_weight_iter)
            ):

                original_dset_key, substitutions = to_fit[:2]
                fit_data = light_curve.get_dataset(original_dset_key,
                                                   **substitutions)[fit_points]

                #Those should come from self.iteritave_fit_config.
                #pylint: disable=missing-kwoa
                fit_results = iterative_fit(
                    predictors=predictors,
                    target_values=fit_data,
                    weights=fit_weights,
                    **self.iterative_fit_config
                )
                #pylint: enable=missing-kwoa

                if fit_results[0] is None:
                    fit_results = dict(
                        corrected_values=scipy.full(fit_data.shape,
                                                    scipy.nan,
                                                    dtype=fit_data.dtype),
                        fit_residual=scipy.nan,
                        non_rejected_points=0
                    )
                else:
                    fit_results = dict(
                        corrected_values=(
                            fit_data
                            -
                            scipy.dot(fit_results[0], predictors)
                        ),
                        fit_residual=fit_results[1]**0.5,
                        non_rejected_points=fit_results[2]
                    )

                result['rms'][0][fit_index] = scipy.sqrt(
                    scipy.nanmean(
                        scipy.power(
                            fit_results['corrected_values'],
                            2
                        )
                    )
                )
                result['num_finite'][0][fit_index] = scipy.isfinite(
                    fit_results['corrected_values']
                ).sum()

                self._save_result(
                    fit_index=fit_index,
                    **fit_results,
                    fit_points=fit_points,
                    light_curve=light_curve,
                )
        return result
    #pylint: disable=too-many-locals
#pylint: enable=too-many-instance-attributes
#pylint: enable=too-few-public-methods
