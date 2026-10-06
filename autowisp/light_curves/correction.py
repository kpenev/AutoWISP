"""Define base class for all LC de-trend algorithms."""

import logging

import numpy

from autowisp.diagnostics.diagnostic_types import (
    parse_photometry_literal,
    shapefit_photometry,
)
from autowisp.exceptions import ConfigurationError


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

    @staticmethod
    def _read_exclusions(exclusion_fname, num_id_values):
        """
        Return the observations an exclusion list names.

        Args:
            exclusion_fname(str or None):    See ``qc_exclude_file`` argument
                to __init__().

            num_id_values(int):    How many values identify an observation.

        Returns:
            dict:
                ``{photometry: [(str, ...), ...]}``: the values identifying
                each listed observation, as written, under the photometry id
                its line names, or under None for a line naming none, which
                applies to every photometry.
        """

        result = {}
        if exclusion_fname is None:
            return result
        with open(exclusion_fname, encoding="utf-8") as exclusion_list:
            for line in exclusion_list:
                values = line.split()
                if not values:
                    continue
                if len(values) == num_id_values:
                    photometry = None
                elif len(values) == num_id_values + 1:
                    photometry = parse_photometry_literal(values[-1])
                    if photometry is None:
                        raise ConfigurationError(
                            f"The exclusion list {exclusion_fname!r} gives "
                            f"{values[-1]!r} as the photometry of an "
                            "observation, where it takes 'shapefit' or 'ap' "
                            "followed by an aperture index."
                        )
                    del values[-1]
                else:
                    raise ConfigurationError(
                        f"The exclusion list {exclusion_fname!r} gives "
                        f"{' '.join(values)!r} as an observation, but "
                        f"observations are identified by {num_id_values} "
                        "values, optionally followed by a photometry."
                    )
                result.setdefault(photometry, []).append(values)
        return result

    def _get_observation_ids(self, light_curve, substitutions):
        """Return the observation IDs from the given light curve."""

        return light_curve.read_data_array(
            {
                str(i): (dset_key, substitutions)
                for i, dset_key in enumerate(self._observation_id)
            }
        )

    def _find_qc_excluded(self, light_curve, fit_target, num_points):
        """
        Flag the points of a lightcurve to leave out of the fit.

        Args:
            light_curve(LightCurveFile):    The lightcurve being corrected.

            fit_target(tuple):    The entry of :attr:`fit_datasets` being
                fit: the pipeline key of the input dataset, the substitutions
                resolving it, and the pipeline key of the output dataset. The
                first two give the photometry, and the substitutions also
                resolve the observation ID datasets.

            num_points(int):    The number of points in the lightcurve.

        Returns:
            numpy.array(dtype=bool):
                For each point, whether its observation is in the exclusion
                list for the photometry being fit.
        """

        if not self._qc_exclusions:
            return numpy.zeros(num_points, dtype=bool)
        return self._is_qc_excluded(
            self._get_observation_ids(light_curve, fit_target[1]),
            *fit_target[:2],
        )

    @staticmethod
    def _get_photometry(dset_key, substitutions):
        """
        Return the photometry id of a fit dataset, as exclusion lists name it.

        Args:
            dset_key(str):    The pipeline key of the dataset.

            substitutions(dict):    The substitutions resolving it.

        Returns:
            int:    :data:`shapefit_photometry` for the shape fit, and the
                aperture index for aperture photometry.
        """

        if dset_key.startswith("shapefit."):
            return shapefit_photometry
        if dset_key.startswith("apphot."):
            return substitutions["aperture_index"]
        raise ConfigurationError(
            f"Cannot tell which photometry {dset_key!r} is, so the exclusions "
            "listed per photometry cannot be applied to it."
        )

    def _match_exclusions(self, observation_ids, photometry):
        """
        Flag the observations listed under one photometry of the list.

        Args:
            observation_ids(structured array):    The observations to check,
                as returned by _get_observation_ids().

            photometry(int or None):    The photometry the list names them
                under, None for the lines naming none.

        Returns:
            numpy.array(dtype=bool):
                For each observation, whether it is listed there.
        """

        if photometry not in self._sorted_exclusions:
            listed = self._qc_exclusions[photometry]
            exclusions = numpy.empty(len(listed), dtype=observation_ids.dtype)
            for name, values in zip(observation_ids.dtype.names, zip(*listed)):
                # Lightcurves give strings as bytes in object columns.
                if observation_ids.dtype[name] == object:
                    exclusions[name] = [
                        value.encode("ascii") for value in values
                    ]
                else:
                    exclusions[name] = numpy.array(values).astype(
                        observation_ids.dtype[name]
                    )
            self._sorted_exclusions[photometry] = numpy.sort(exclusions)

        sorted_exclusions = self._sorted_exclusions[photometry]
        matched_indices = numpy.searchsorted(sorted_exclusions, observation_ids)
        matched_indices[matched_indices == sorted_exclusions.size] = 0
        return sorted_exclusions[matched_indices] == observation_ids

    def _is_qc_excluded(self, observation_ids, dset_key, substitutions):
        """
        Flag the observations the exclusion list leaves out of one dataset.

        Those listed with no photometry, and those listed under the
        photometry of the dataset.

        Args:
            observation_ids(structured array):    The observations to check,
                as returned by _get_observation_ids().

            dset_key(str):    The pipeline key of the input dataset being fit.

            substitutions(dict):    The substitutions resolving it.

        Returns:
            numpy.array(dtype=bool):
                For each observation, whether it is in the exclusion list for
                the photometry of the dataset.
        """

        result = numpy.zeros(observation_ids.shape, dtype=bool)
        for photometry in self._qc_exclusions:
            if photometry is None or photometry == self._get_photometry(
                dset_key, substitutions
            ):
                result |= self._match_exclusions(observation_ids, photometry)
        return result

    # Keyword-only, each named at every call, and bundling them would only
    # move the list somewhere less visible.
    def _save_result(  # pylint: disable=too-many-arguments
        self,
        *,
        fit_index,
        corrected_values,
        fit_residual,
        non_rejected_points,
        fit_points,
        qc_excluded,
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

            qc_excluded(bool array):    Flags indicating for each entry in the
                input (uncorrected) dataset, whether it was left out of the
                fit. See _find_qc_excluded().

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
        light_curve.add_corrected_dataset(
            original_key=original_key,
            corrected_key=config_key_prefix + ".qc_included",
            corrected_values=numpy.logical_not(qc_excluded[fit_points]),
            corrected_selection=fit_points,
            **substitutions,
        )

        configuration = tuple(
            configuration
            + [
                (config_key_prefix + ".fit_residual", fit_residual),
                (config_key_prefix + ".num_fit_points", non_rejected_points),
                (
                    config_key_prefix + ".cfg.exclusion_rule",
                    self._exclusion_rule.encode("utf-8"),
                ),
                (
                    config_key_prefix + ".num_qc_excluded_points",
                    numpy.uint(qc_excluded[fit_points].sum()),
                ),
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

    # Keyword-only, each named at every call.
    @staticmethod
    def _process_fit(  # pylint: disable=too-many-arguments
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
        """
        Return the lightcurve points to detrend.

        Args:
            light_curve(LightCurveFile):    The lightcurve being corrected.

            get_fit_dataset(callable):    See EPDCorrection.__call__().

            fit_target((str, dict)):    The dataset key and substitutions
                identifying a unique dataset in the lightcurve to fit.

            fit_points(bool array):    Flags selecting the points to return.

        Returns:
            numpy.array:
                The selected points of the dataset to apply the correction to.

            numpy.array:
                The selected points of the dataset to derive the correction
                from: a separate array, even when ``get_fit_dataset`` returns
                a single dataset to use for both.
        """

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
        raw_values = get_fit_dataset(
            light_curve, fit_target[0], **substitutions
        )
        if isinstance(raw_values, tuple):
            raw_values, fit_data = raw_values
        else:
            fit_data = raw_values
        return raw_values[fit_points], fit_data[fit_points]

    def __init__(
        self,
        fit_datasets,
        mark_progress,
        *,
        observation_id=None,
        qc_exclude_file=None,
        exclusion_rule=None,
        **iterative_fit_config,
    ):
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

            observation_id([str]):    The pipeline keys of the datasets whose
                values identify an observation. Only needed to match
                observations across lightcurves or with ``qc_exclude_file``.

            qc_exclude_file(str or None):    The observations to leave out of
                the fit while still correcting them: one per line, each given
                by the values of the ``observation_id`` datasets separated by
                white space, blank lines ignored. A line may end with the
                photometry it applies to, ``shapefit`` or ``ap`` followed by
                an aperture index; without one it applies to every
                photometry. None excludes nothing.

            exclusion_rule(str or None):    The rule that produced
                ``qc_exclude_file``, recorded with the fit. None if the list
                was not produced by a rule.

            iterative_fit_config:    Any other arguments to pass directly to
                iterative_fit().

        Returns:
            None
        """

        self.fit_datasets = fit_datasets
        self.iterative_fit_config = iterative_fit_config
        self.mark_progress = mark_progress
        self._fixed_substitutions = None
        self._observation_id = observation_id
        self._exclusion_rule = exclusion_rule or ""
        self._qc_exclusions = self._read_exclusions(
            qc_exclude_file, len(observation_id or ())
        )
        # Sorted for matching, per photometry, when first needed: the dtype
        # comes from the lightcurves' observation ids.
        self._sorted_exclusions = {}


# pylint: enable=too-few-public-methods
