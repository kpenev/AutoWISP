"""Implement magnitude fitting using linear regression."""

from superphot_pipeline.magnitude_fitting.base import MagnitudeFit

class LinearMagnitudeFit(MagnitudeFit):
    """Differential photometry correction using linear regression."""

    def _fit(self, fit_data):
        pass

    def _apply_fit(self, phot, coefficients):
        pass

    def __init__(self, *, config, **kwargs):

        super().__init__(config=config, **kwargs)
