"""Define base class for all LC de-trend algorithms."""

import scipy

#Intended to serve as base class only
#pylint: disable=too-few-public-methods
class Correction:
    """
    Functionality and interface shared by all LC de-trending corrections.

    Attributes:
        fit_datasets:    See __init__().

        iterative_fit_config(dict):    Configuration to use for iterative
            fitting. See iterative_fit() for details.
    """

    def _save_result(self,
                     *,
                     fit_index,
                     corrected_values,
                     fit_residual,
                     non_rejected_points,
                     fit_points,
                     configuration,
                     light_curve):
        """
        Stores the de-treneded results and configuration to the light curve.

        Args:
            fit_index(int):    The index of the dataset for which a correction
                was applied in within the list of datasets specified at init.

            corrected_values(array):    The corrected data to save.

            fit_residual(float):    The residual from the fit, calculated as
                specified at init.

            non_rejected_points(int):    The number of points used in the last
                iteration of the itaritive fit.

            fit_points(bool array):    Flags indicating for each entry in the
                input (uncorrected) dataset, whether it is represented in
                `corrected_values`.

            configuration([]):    The configuration used for the fit, properly
                formatted to be converted to an entry in the configurations
                argument to LightCurveFile.add_configurations().

            light_curve(LightCurveFile):    A light curve file opened for
                writing.

        Returns:
            None
        """

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


        configuration = tuple(
            configuration
            +
            [
                (config_key_prefix + '.fit_residual', fit_residual),
                (config_key_prefix + '.num_fit_points', non_rejected_points)
            ]
        )
        light_curve.add_configurations(
            component=config_key_prefix,
            configurations=(configuration,),
            config_indices=scipy.zeros(shape=(corrected_values.size,),
                                       dtype=scipy.uint),
            config_index_selection=fit_points,
            **substitutions
        )

    @staticmethod
    def _process_fit(*,
                     fit_results,
                     raw_values,
                     predictors,
                     fit_index,
                     result,
                     num_extra_predictors):
        """
        Incorporate the results of a single fit in final result of __call__().

        Args:
            fit_results:    The return value of the iterative_fit() used to
                calculate the correction

            raw_values(1-D array):    The data to apply the correction to.
                Should already exclude any points not selected for correction.

            predictors(2-D array):    The predictors to use for the correction
                (e.g. the templates for TFA fitting).

            fit_index(int):    The index of the dataset being fit within the
                list of datasets that will be fit for this lightcurve.

            result:    The result variable for the parent update for this
                fit.

            num_extra_predictors(int):    How many extra predictors are
                there.

        Returns:
            dict:
                The same structure as fit_results, but ready to pass to
                self._save_result().
        """

        if fit_results[0] is None:
            if num_extra_predictors:
                for predictor in result.dtype.names[-num_extra_predictors:]:
                    result[predictor][0][fit_index] = scipy.nan

            fit_results = dict(
                corrected_values=scipy.full(raw_values.shape,
                                            scipy.nan,
                                            dtype=raw_values.dtype),
                fit_residual=scipy.nan,
                non_rejected_points=0,
            )

        else:
            if num_extra_predictors:
                for predictor, amplitude in zip(
                        result.dtype.names[-num_extra_predictors:],
                        fit_results[0][-num_extra_predictors:]
                ):
                    result[predictor][0][fit_index] = amplitude
                corrected_values = (
                    raw_values
                    -
                    scipy.dot(fit_results[0][:-num_extra_predictors],
                              predictors[:-num_extra_predictors])
                )
            else:
                corrected_values = (raw_values
                                    -
                                    scipy.dot(fit_results[0], predictors))

            fit_results = dict(
                corrected_values=corrected_values,
                fit_residual=fit_results[1]**0.5,
                non_rejected_points=fit_results[2],
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
        return fit_results

    def __init__(self, fit_datasets, **iterative_fit_config):
        """
        Configure the fitting.

        Args:
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


            iterative_fit_config:    Any other arguments to pass directly to
                iterative_fit().

        Returns:
            None
        """

        self.fit_datasets = fit_datasets
        self.iterative_fit_config = iterative_fit_config
#pylint: enable=too-few-public-methods
