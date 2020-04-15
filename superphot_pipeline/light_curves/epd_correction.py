"""Define class for performing EPD correction on lightcurves."""

from itertools import repeat

import scipy
from numpy.lib import recfunctions

from superphot_pipeline import LightCurveFile
from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.fit_expression import\
    Interface as FitTermsInterface,\
    iterative_fit
from superphot_pipeline.light_curves.correction import Correction

#Attempts to re-organize reduced readability
#pylint: disable=too-many-instance-attributes
#Using a class is justified.
#pylint: disable=too-few-public-methods
class EPDCorrection(Correction):
    """
    Class for deriving and applying EPD corrections to lightcurves.

    Attributes:
        used_variables:    See __init__.

        fit_points_filter_expression:    See __init__.

        fit_terms(FitTermsInterface):    Instance set-up to generate the
            independent variables matrix needed for the fits.

        fit_weights:    See __init__.

        _io_fit_config([]):    A list storing the configuration used for the
            fitting. Formatted properly for passing as the only entry
            directly to LightCurveFile.add_configurations().

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
        for fit_target, fit_weights in zip(
                self.fit_datasets,
                (
                    repeat(self.fit_weights or '') if (
                        self.fit_weights is None
                        or
                        isinstance(self.fit_weights, str)
                    ) else self.fit_weights
                )
        ):

            pipeline_key_prefix = fit_target[2]

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
                        fit_weights.encode('ascii')
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

            fit_datasets:    See Correction.__init__().

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

        super().__init__(fit_datasets, **iterative_fit_config)

        self.used_variables = used_variables
        self.fit_points_filter_expression = fit_points_filter_expression
        self.fit_terms_expression = fit_terms_expression
        self.fit_weights = fit_weights
        self._io_fit_config = self._get_fit_configurations(fit_terms_expression)

    @staticmethod
    def get_result_dtype(num_photometries, extra_predictors=None):
        """Return the data type for the result of __call__."""

        return (
            [
                ('ID', (scipy.int_, 3)),
                ('mag', scipy.float64),
                ('xi', scipy.float64),
                ('eta', scipy.float64),
                ('rms', (scipy.float64, (num_photometries,))),
                ('num_finite', (scipy.uint, (num_photometries,)))
            ]
            +
            [
                (predictor_name, scipy.float64)
                for predictor_name in (
                    [] if extra_predictors is None
                    else (extra_predictors.keys()
                          if isinstance(extra_predictors, dict) else
                          extra_predictors.dtype.names)
                )
            ]
        )

    #Re-factored as much as I could (KP)
    #pylint: disable=too-many-locals
    def __call__(self,
                 lc_fname,
                 get_fit_data=LightCurveFile.get_dataset,
                 extra_predictors=None,
                 save=True):
        """
        Fit and correct the given lightcurve.

        Args:
            lc_fname(str):    The filename of the light curve to fit.

            get_fit_data(callable):    A function that takes a LightCurveFile
                instance, dataset key and substitutions and returns either a
                single array which is the dataset to calculate and apply EPD
                correction to, or a 2-tuple of arrays, the first of which is
                what the calculated correction is applied to, and the second one
                is used to calculate the EPD correction. The intention is to
                allow for protecting a signal from being modified by the fit, in
                which case the second dataset should have the protected signal
                removed from it, and the first dataset should be the original
                datasets stored in the lightcurve.

            extra_predictors(None, dict, or scipy structured array):
                Additional predictor datasets to add to the ones configured
                through __init__, for this lightcurve only. The intent is to
                allow for reconstructive EPD, by passing an expected signal or a
                set of signals which are fit simultaneously to the EPD
                corrections. The derived corrections for these components are
                not applied when calculating the corrected dataset, but the best
                fit amplitudes are added to the result.

            save(bool):    Should the result be saved to the lightcurve. Can be
                used to disable saving if the current EPD evaluation is not the
                final one during reconstructive EPD.

        Returns:
            numpy.array(dtype=[('rms', numpy.float64),
                               ('num_finite', numpy.uint)]):
                The RMS of the corrected values and the number of finite points
                for each corrected dataset in the order in which the datasets
                were supplied to __init__().

            numpy.array(dtype=[(extra predictor 1, numpy.float64),
                               ...]:
                The best-fit amplitudes for the `extra_predictors`.
        """

        def prepare_fit(extra_predictor_order):
            """Return predictors, weights, and array flagging points to fit."""

            evaluate = Evaluator(
                light_curve.read_data_array(self.used_variables)
            )

            predictors = FitTermsInterface(self.fit_terms_expression)(evaluate)

            if extra_predictors:
                predictors = recfunctions.append_fields(
                    predictors,
                    extra_predictor_order,
                    [
                        extra_predictors[predictor]
                        for predictor in extra_predictor_order
                    ],
                    usemask=False
                )

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
            return predictors, fit_weight_iter, fit_points

        def correct_one_dataset(light_curve,
                                *,
                                predictors,
                                fit_target,
                                weights,
                                fit_index,
                                result,
                                num_extra_predictors):
            """
            Calculate and apply EPD correction to a single dataset.

            Args:
                light_curve(LightCurveFile):    The opened for writing light
                    curve to apply EPD corrections to.

                fit_target((str, dict)):    The dataset key and substitutions
                    identifying a uniquedataset in the lightcurve to fit.

                predictors(structured array):    The predictors to use for EPD
                    corrections, including the `extra_predictors`.

                weights(array):    The weight to use for each point in the
                    dataset being fit.

                fit_index(int):    The index of the dataset being fit within the
                    list of datasets that will be fit for this lightcurve.

                result:    The result variable for the parent update for this
                    fit.

                num_extra_predictors(int):    How many extra predictors are
                    there.

            Returns:
                (float, float, float64 array):
                    * The square root of the average square residuals after the
                      fit.

                    * The `'num_finite'` entry in the parent's result.

                    * A list of the best-fit amplitudes of the
                      `extra_predictors` signals.
            """

            raw_values = get_fit_data(light_curve,
                                      fit_target[0],
                                      **fit_target[1])
            if isinstance(raw_values, tuple):
                raw_values, fit_data = raw_values
            else:
                fit_data = raw_values

            raw_values = raw_values[fit_points]
            fit_data = fit_data[fit_points]

            #Those should come from self.iteritave_fit_config.
            #pylint: disable=missing-kwoa
            fit_results = iterative_fit(
                predictors=predictors,
                target_values=fit_data,
                weights=weights,
                **self.iterative_fit_config
            )
            #pylint: enable=missing-kwoa

            fit_results = self._process_fit(
                fit_results=fit_results,
                raw_values=raw_values,
                predictors=predictors,
                fit_index=fit_index,
                result=result,
                num_extra_predictors=num_extra_predictors
            )
            if save:
                self._save_result(
                    fit_index=fit_index,
                    configuration=self._io_fit_config[fit_index],
                    **fit_results,
                    fit_points=fit_points,
                    light_curve=light_curve,
                )

        if extra_predictors is None:
            num_extra_predictors = 0
        elif isinstance(extra_predictors, dict):
            num_extra_predictors = len(extra_predictors)
        else:
            num_extra_predictors = len(extra_predictors.dtype.names)

        with LightCurveFile(lc_fname, 'r+') as light_curve:

            result = scipy.empty(
                1,
                dtype=self.get_result_dtype(len(self.fit_datasets),
                                            extra_predictors)
            )

            predictors, fit_weight_iter, fit_points = prepare_fit(
                result.dtype.names[-num_extra_predictors:] if extra_predictors
                else []
            )

            for fit_index, to_fit in enumerate(
                    zip(self.fit_datasets,
                        fit_weight_iter)
            ):
                correct_one_dataset(
                    light_curve=light_curve,
                    predictors=predictors,
                    fit_target=to_fit[0],
                    weights=to_fit[1],
                    fit_index=fit_index,
                    result=result,
                    num_extra_predictors=num_extra_predictors
                )

        return result
    #pylint: enable=too-many-locals
#pylint: enable=too-many-instance-attributes
#pylint: enable=too-few-public-methods
