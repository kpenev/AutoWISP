"""Definet class performing reconstructive EPD on LCs with transits."""

from superphot_pipeline.light_curves.transit_model import magnitude_change
from superphot_pipeline.light_curves.epd_correction import EPDCorrection
from superphot_pipeline import LightCurveFile

class ReconstructiveEPDTransit(EPDCorrection):
    """
    Class for EPD corrections that protect known on suspected transit signals.

    Attributes:
        transit_model:    See same name argument to __init__()

        fit_amplitude:    See same name argument to __init__()

        transit_parameters(2-tuple):     The positional and keyword arguments to
            pass to the transit model's evaluate() method.
    """

    def __init__(self,
                 transit_model,
                 *epd_args,
                 fit_amplitude=True,
                 **epd_kwargs):
        """
        Configure the fitting.

        Args:
            transit_model:    If not None, this should be one of the models
                implemented in pytransit.

            fit_amplitude(bool):    Should the amplitude of the model be
                fit along with the EPD correction coefficients? If not, the
                amplitude of the signal is assumed known.

            epd_args:    Passed directly as positional arguments to parent class
                __init__().

            epd_kwargs:    Passed directly as keyword arguments to parent class
                __init__().

        Returns:
            None
        """

        super().__init__(*epd_args, **epd_kwargs)
        self.transit_model = transit_model
        self.transit_parameters = None
        self.fit_amplitude = fit_amplitude

    def get_fit_data(self, light_curve, dset_key, **substitutions):
        """To be used as the get_fit_data argument to parent's __call__."""

        raw_magnitudes = light_curve.get_dataset(dset_key, **substitutions)

        if self.transit_model is None or self.fit_amplitude:
            return raw_magnitudes

        return (
            raw_magnitudes,
            raw_magnitudes - magnitude_change(light_curve,
                                              self.transit_model,
                                              *self.transit_parameters[0],
                                              **self.transit_parameters[1])
        )

    #The call signature is deliberately different than the underlying class.
    #pylint: disable=arguments-differ
    def __call__(self,
                 lc_fname,
                 *transit_parameters_pos,
                 save=True,
                 **transit_parameters_kw):
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
            with LightCurveFile(lc_fname, 'r') as light_curve:
                extra_predictors = dict(
                    transit=magnitude_change(light_curve,
                                             self.transit_model,
                                             *transit_parameters_pos,
                                             **transit_parameters_kw)
                )
        else:
            self.transit_parameters = (transit_parameters_pos,
                                       transit_parameters_kw)
            extra_predictors = None

        return super().__call__(lc_fname, self.get_fit_data, extra_predictors, save)
    #pylint: enable=arguments-differ
