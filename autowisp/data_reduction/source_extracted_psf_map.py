"""Define callable class representing PSF maps from source extraction."""

import pandas
import numpy

from autowisp.fit_expression import Interface as FitTermsInterface


# Instances are intended to work as functions, so no methods other than __call__
# pylint: disable=too-few-public-methods
class SourceExtractedPSFMap:
    """Evaluate smoothed PSF maps based on sorce extraction."""

    def __init__(self, psf_parameters, terms_expression, coefficients):
        """
        Create the map for the given parameters per the given coefficients.

        Args:
            psf_parameters([str]):    The names of the mapped PSF parameters.

            terms_expression(str):    An expression that expands to the fitting
                terms the map depends on.

            coefficients(2-D numpy.array):    The coefficients defining the map.
                The outer (0-th) index should select the PSF parameter.

        Returns:
            None
        """

        self._psf_parameters = psf_parameters
        self._get_predictors = FitTermsInterface(terms_expression)
        self._coefficients = coefficients

    def __call__(self, source_data):
        """
        Evaluate the map for the given sources.

        Args:
            source_data(numpy structured array or pandas DataFrame):    The
                x, y and catalogue information for the sources for which to
                evaluate the map. The names of the fields should exactly match
                the names used when creating the map.

        Returns:
            numpy structured array:
                The PSF parameters predicted by the map for the input sources.
                The fields will have the names of the PSF parameters.
        """

        print("Source data: " + repr(source_data))
        assert (
            isinstance(source_data, pandas.DataFrame)
            or len(source_data.shape) == 1
        )
        result = numpy.empty(
            len(source_data),
            dtype=[
                (param_name, numpy.float64)
                for param_name in self._psf_parameters
            ],
        )

        predictors = self._get_predictors(source_data)
        for coefficients, param_name in zip(
            self._coefficients, self._psf_parameters
        ):
            result[param_name] = numpy.dot(coefficients, predictors)

        return result


# pylint: enable=too-few-public-methods
