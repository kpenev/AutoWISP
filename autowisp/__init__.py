"""A general purpose AstroWISP based photometry pipeline."""

from autowisp.evaluator import Evaluator

try:
    from autowisp.processor import Processor
    from autowisp.data_reduction.data_reduction_file import\
        DataReductionFile
    from autowisp.light_curves.light_curve_file import LightCurveFile
    from autowisp.light_curves.epd_correction import EPDCorrection
    from autowisp.light_curves.tfa_correction import TFACorrection
    from autowisp.source_finder import SourceFinder
    from autowisp.piecewise_bicubic_psf_map import PiecewiseBicubicPSFMap
except:
    pass
