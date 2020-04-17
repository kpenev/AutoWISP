"""Define a function implementing reconstructive detrending."""

import numpy
from scipy.optimize import minimize

from superphot_pipeline.light_curves.reconstructive_correction_transit import\
    ReconstructiveCorrectionTransit

def apply_reconstructive_correction_transit(lc_fname,
                                            correct,
                                            *,
                                            transit_model,
                                            transit_parameters,
                                            fit_parameter_flags,
                                            num_limbdark_coef):
    """
    Perform a reconstructive EPD on a lightcurve assuming it contains a transit.

    The corrected lightcurve, preserving the best-fit transit is saved in the
    lightcurve just like for non-reconstructive EPD.

    Args:
        transit_model:    Object which supports the transit model intefrace of
            pytransit.

        transit_parameters(scipy float array):    The full array of parameters
            required by the transit model's evaluate() method.

        fit_parameter_flags(scipy bool array):    Flags indicating parameters
            whose values should be fit for (by having a corresponding entry of
            True). Must match exactly the shape of transit_parameters.

        num_limbdark_coef(int):    How many of the transit parameters are limb
            darkening coefficinets? Those need to be passed to the model
            separately.

        correct(Correction):    Instance of one of the correction algarithms to
            make adaptive.

    Returns:
        (scipy array, scipy array):
            * The best fit transit parameters

            * The return value of ReconstructiveEPDTransit.__call__() for the
              best-fit transit parameters.
    """

    #This is intended to server as a callable.
    #pylint: disable=too-few-public-methods
    class MinimizeFunction:
        """Suitable callable for scipy.optimize.minimize()."""

        def __init__(self):
            """Create the EPD object."""

            self.epd = ReconstructiveCorrectionTransit(
                transit_model,
                correct,
                fit_amplitude=False,
            )
            self.transit_parameters = numpy.copy(transit_parameters)

        def __call__(self, fit_params):
            """
            Return the RMS residual of the EPD after removing a transit model.

            Args:
                fit_params(scipy array):    The values of the mutable model
                    parameters for the current minimization function evaluation.

            Returns:
                float:
                    RMS of the residuals after EPD correctiong around the
                    transit model with the given parameters.
            """

            self.transit_parameters[fit_parameter_flags] = fit_params
            return self.epd(lc_fname,
                            self.transit_parameters[0],
                            self.transit_parameters[1 : num_limbdark_coef + 1],
                            *self.transit_parameters[num_limbdark_coef + 1 : ],
                            save=False)['rms']
    #pylint: enable=too-few-public-methods

    rms_function = MinimizeFunction()
    best_fit_transit = numpy.copy(transit_parameters)

    if fit_parameter_flags.any():
        minimize_result = minimize(rms_function,
                                   transit_parameters[fit_parameter_flags])
        assert minimize_result.success
        best_fit_transit[fit_parameter_flags] = minimize_result.x

    return (
        best_fit_transit,
        rms_function.epd(lc_fname,
                         best_fit_transit[0],
                         best_fit_transit[1: num_limbdark_coef + 1],
                         *best_fit_transit[num_limbdark_coef + 1 : ])
    )
