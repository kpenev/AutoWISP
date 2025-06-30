#!/usr/bin/env python3

"""
Predict LC magnitudes given transit model parameters.

Basically a convenient interface around pytransit, specific to the pipeline's
lightcurve format.
"""

import numpy


def magnitude_change(light_curve, transit_model, *model_args, **model_kwargs):
    """
    Evaluate the given model at the exposure times contained in the lightcurve.

    Args:
        light_curve(LightCurveFile):    The lightcurve to model.

        transit_model:    One of the pytransit transit models (or something that
            supports the same interface).

        model_args:    Passed directly as positional arguments to
            transit_model.evaluate().

        model_kwargs:    Passed directly as kewyord argumets to
            transit_model.evaluate().

    Returns:
        array:
            The change in magnitude of the given star due to the transit (i.e.
            zere for all LC pointst out-of-transit).
    """

    transit_model.set_data(
        light_curve.read_data_array({"BJD": ("skypos.BJD", {})})["BJD"]
    )
    return -2.5 * numpy.log10(
        transit_model.evaluate(*model_args, **model_kwargs)
    )
