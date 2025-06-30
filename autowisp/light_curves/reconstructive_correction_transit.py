"""Definet class performing reconstructive detrending on LCs with transits."""

from .transit_model import magnitude_change
from .light_curve_file import LightCurveFile


class ReconstructiveCorrectionTransit:
    """
    Class for corrections that protect known or suspected transit signals.

    Attributes:
        transit_model:    See same name argument to __init__()

        fit_amplitude:    See same name argument to __init__()

        transit_parameters(2-tuple):     The positional and keyword arguments to
            pass to the transit model's evaluate() method.
    """

    def __init__(
        self,
        transit_model,
        correction,
        fit_amplitude=True,
        transit_parameters=None,
    ):
        """
        Configure the fitting.

        Args:
            transit_model:    If not None, this should be one of the models
                implemented in pytransit.

            correction(Correction):    An instance of one of the correction
                classes.

            fit_amplitude(bool):    Should the amplitude of the model be
                fit along with the EPD correction coefficients? If not, the
                amplitude of the signal is assumed known.

            transit_parameters(2-tiple):    Positional and keyword arguments to
                pass to the transit model. Overwritten by `__call__`. The first
                entry should be iterable and specifies positional arguments, the
                second entry is dict and specifies keyword arguments.

        Returns:
            None
        """

        self.correction = correction
        self.transit_model = transit_model
        self.transit_parameters = transit_parameters
        self.fit_amplitude = fit_amplitude

    def get_fit_data(self, light_curve, dset_key, **substitutions):
        """To be used as the get_fit_data argument to parent's __call__."""

        raw_magnitudes = light_curve.get_dataset(dset_key, **substitutions)

        if self.transit_model is None or self.fit_amplitude:
            return raw_magnitudes

        fit_magnitudes = raw_magnitudes - magnitude_change(
            light_curve,
            self.transit_model,
            *self.transit_parameters[0],
            **self.transit_parameters[1],
        )
        return raw_magnitudes, fit_magnitudes

    # The call signature is deliberately different than the underlying class.
    # pylint: disable=arguments-differ
    def __call__(
        self,
        lc_fname,
        *transit_parameters_pos,
        save=True,
        **transit_parameters_kw,
    ):
        """
        Perform reconstructive EPD on a light curve, given transit parameters.

        Args:
            lc_fname(str):    The filename of the lightcurve to fit.

            save(bool):   See same name orgument to EPDCorrection.__call__().

            transit_parameters_pos:    Positional arguments to be passed to the
                transit model's evaluate() method.

            transit_parameters_kw:    Keyword arguments to be passed to the
                transit model's evaluate() method.

        Returns:
            See EPDCorrection.__call__()
        """

        if self.fit_amplitude:
            with LightCurveFile(lc_fname, "r") as light_curve:
                extra_predictors = {
                    "transit": magnitude_change(
                        light_curve,
                        self.transit_model,
                        *transit_parameters_pos,
                        **transit_parameters_kw,
                    )
                }
        else:
            self.transit_parameters = (
                transit_parameters_pos,
                transit_parameters_kw,
            )
            extra_predictors = None

        return self.correction(
            lc_fname, self.get_fit_data, extra_predictors, save
        )

    # pylint: enable=arguments-differ
