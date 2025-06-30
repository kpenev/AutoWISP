"""Define base class for all LC de-trend algorithms."""

import logging

import numpy


# Intended to serve as base class only
# pylint: disable=too-few-public-methods
class Correction:
    """
    Functionality and interface shared by all LC de-trending corrections.

    Attributes:
        fit_datasets:    See __init__().

        iterative_fit_config(dict):    Configuration to use for iterative
            fitting. See iterative_fit() for details.
    """

    _logger = logging.getLogger(__name__)

    @staticmethod
    def _get_config_key_prefix(fit_target):
        """Return the prefix of the pipeline key for storing configuration."""

        print("Fit target: " + repr(fit_target))
        return fit_target[2].rsplit(".", 1)[0] + ".cfg."

    def _get_io_iterative_fit_config(self, pipeline_key_prefix):
        """
        Return the iterative fit portion of the configuration to save in the LC.

        Args:
            pipeline_key_prefix(str):    The part of the pipeline key specifying
                which configuration is being defined (i.e. everything except the
                last item in the key).

        Returns:
            [()]:
                A list of tuples of the configuration options contained in
                :attr:`iterative_fit_config`.
        """

        return [
            (
                pipeline_key_prefix + "error_avg",
                self.iterative_fit_config["error_avg"].encode("ascii"),
            )
        ] + [
            (pipeline_key_prefix + cfg_key, self.iterative_fit_config[cfg_key])
            for cfg_key in ["rej_level", "max_rej_iter"]
        ]

    def _save_result(
        self,
        *,
        fit_index,
        corrected_values,
        fit_residual,
        non_rejected_points,
        fit_points,
        configuration,
        light_curve,
    ):
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

        original_key, substitutions, destination_key = self.fit_datasets[
            fit_index
        ]
        if self._fixed_substitutions is not None:
            substitutions = self._fixed_substitutions
            self._fixed_substitutions = None

        light_curve.add_corrected_dataset(
            original_key=original_key,
            corrected_key=destination_key,
            corrected_values=corrected_values,
            corrected_selection=fit_points,
            **substitutions,
        )
        config_key_prefix = destination_key.rsplit(".", 1)[0]

        configuration = tuple(
            configuration
            + [
                (config_key_prefix + ".fit_residual", fit_residual),
                (config_key_prefix + ".num_fit_points", non_rejected_points),
            ]
        )
        light_curve.add_configurations(
            component=config_key_prefix,
            configurations=(configuration,),
            config_indices=numpy.zeros(
                shape=(fit_points.size,), dtype=numpy.uint
            ),
            **substitutions,
        )

    @staticmethod
    def _process_fit(
        *,
        fit_results,
        raw_values,
        predictors,
        fit_index,
        result,
        num_extra_predictors,
    ):
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
                    result[predictor][0][fit_index] = numpy.nan

            fit_results = {
                "corrected_values": numpy.full(
                    raw_values.shape, numpy.nan, dtype=raw_values.dtype
                ),
                "fit_residual": numpy.nan,
                "non_rejected_points": 0,
            }

        else:
            if num_extra_predictors:
                for predictor, amplitude in zip(
                    result.dtype.names[-num_extra_predictors:],
                    fit_results[0][-num_extra_predictors:],
                ):
                    result[predictor][0][fit_index] = amplitude
                corrected_values = raw_values - numpy.dot(
                    fit_results[0][:-num_extra_predictors],
                    predictors[:-num_extra_predictors],
                )
            else:
                corrected_values = raw_values - numpy.dot(
                    fit_results[0], predictors
                )

            fit_results = {
                "corrected_values": corrected_values,
                "fit_residual": fit_results[1] ** 0.5,
                "non_rejected_points": fit_results[2],
            }

        result["rms"][0][fit_index] = numpy.sqrt(
            numpy.nanmean(numpy.power(fit_results["corrected_values"], 2))
        )
        result["num_finite"][0][fit_index] = numpy.isfinite(
            fit_results["corrected_values"]
        ).sum()
        return fit_results

    @staticmethod
    def get_result_dtype(num_photometries, extra_predictors=None, id_size=1):
        """Return the data type for the result of __call__."""

        return [
            (
                "ID",
                numpy.uint64 if id_size == 1 else (numpy.uint64, id_size),
            ),
            ("mag", numpy.float64),
            ("xi", numpy.float64),
            ("eta", numpy.float64),
            ("rms", (numpy.float64, (num_photometries,))),
            ("num_finite", (numpy.uint, (num_photometries,))),
        ] + [
            (predictor_name, numpy.float64)
            for predictor_name in (
                []
                if extra_predictors is None
                else (
                    extra_predictors.keys()
                    if isinstance(extra_predictors, dict)
                    else extra_predictors.dtype.names
                )
            )
        ]

    def _fix_substitutions(
        self,
        *,
        light_curve,
        photometry_mode,
        fit_points,
        substitutions,
        in_place=False,
    ):
        """Fix magfit iteration in substitutions if negative."""

        self._logger.debug("Fixing LC substitutions: %s", repr(substitutions))
        if substitutions.get("magfit_iteration", 0) >= 0:
            return substitutions
        if not in_place:
            substitutions = dict(substitutions)
        substitutions[
            "magfit_iteration"
        ] += light_curve.get_num_magfit_iterations(
            photometry_mode, fit_points, **substitutions
        )
        assert self._fixed_substitutions is None
        self._fixed_substitutions = substitutions
        return substitutions

    def _get_fit_data(
        self, light_curve, get_fit_dataset, fit_target, fit_points
    ):
        """Return the lightcurve points to detrend."""

        substitutions = self._fix_substitutions(
            light_curve=light_curve,
            photometry_mode=fit_target[0].split(".", 1)[0],
            fit_points=fit_points,
            substitutions=fit_target[1],
        )
        self._logger.debug(
            "Fitting %s (%s) for %s ",
            fit_target[0],
            repr(substitutions),
            light_curve.filename,
        )
        return get_fit_dataset(light_curve, fit_target[0], **substitutions)

    def __init__(self, fit_datasets, mark_progress, **iterative_fit_config):
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
        self.mark_progress = mark_progress
        self._fixed_substitutions = None


# pylint: enable=too-few-public-methods
