"""Define class for working with piecewise bucubic PSF maps."""

import numpy
from astropy.io import fits
import superphot

from superphot_pipeline import fit_expression
from superphot_pipeline.data_reduction import DataReductionFile

class PiecewiseBicubicPSFMap:
    """Fit and use piecewise bicubic PSF maps."""


    def __init__(self,
                 shape_terms_expression,
                 *,
                 background_annulus=(6.0, 7.0),
                 require_convergence=True,
                 **fit_star_shape_config):
        """
        Create a map involving the specified smooth dependence terms.

        Args:
            shape_terms_expression(str):    An expression specifying the terms
                PSF/PRF parameters are allowed to dependend on. May involve
                arbitrary catalogque position and other external variables. See
                fit_expression.Interface for details.

            background_annulus(float, float):    The inner radius and difference
                between inner and outer radius of the annulus around each source
                where background is measured. See superphot.BackgroundExtractor
                for details.

            require_convergence(bool):    See same name argument to
                superphot.FitStarShape.fit()

            fit_star_shape_config:    Any required configuration by
                superphot.FitStarShape.__init__()

        Returns:
            None
        """

        self.configuration = dict(shape_terms_expression=shape_terms_expression,
                                  background_annulus=background_annulus,
                                  require_convergence=require_convergence)

        self._eval_shape_terms = fit_expression.Interface(
            shape_terms_expression
        )
        self._star_shape_fitter = superphot.FitStarShape(
            **fit_star_shape_config
        )


    #Breaking up seems to make things worse
    #pylint: disable=too-many-locals
    def fit(self, fits_fnames, sources, output_dr_fnames=None):
        """
        Find the best fit PSF/PRF map for the given images (simultaneous fit).

        Args:
            fits_fnames(str iterable):    The filenames of the FITS files to
                fit.

            sources(numpy structured array iterable):    For each image specify
                the sources on which to base the PSF/PRF map. Sources are
                supplied as a numpy structured array that should define all
                variables required to evaluate the shape term expression
                specified on __init__(). It must also include ``'x'`` and
                ``'y'`` fields, even if those are not required to evaluate the
                terms.

            output_fnames(str iterable):    If not None, should specify one data
                reduction file for each input frame where the fit results are
                saved.

        Returns:
            None
        """

        assert len(fits_fnames) == len(sources)
        assert (output_dr_fnames is None
                or
                len(output_dr_fnames) == len(fits_fnames))

        opened_frames = [fits.open(fname, 'readonly') for fname in fits_fnames]
        value_index = 1 if opened_frames[0][0].header['NAXIS'] == 0 else 0
        error_index, mask_index = value_index + 1, value_index + 2
        for frame in opened_frames:
            assert frame[error_index].header['IMAGETYP'] == 'error'
            assert frame[mask_index].header['IMAGETYP'] == 'mask'

        measure_backgrounds = [
            superphot.BackgroundExtractor(
                frame[value_index].data.astype(numpy.float64, copy=False),
                self.configuration['background_annulus'][0],
                sum(self.configuration['background_annulus'])
            )
            for frame in opened_frames
        ]

        for get_bg, frame_sources in zip(measure_backgrounds, sources):
            get_bg(numpy.copy(frame_sources['x']),
                   numpy.copy(frame_sources['y']))

        fit_result = self._star_shape_fitter.fit(
            [
                (
                    frame[value_index].data.astype(numpy.float64, copy=False),
                    frame[error_index].data.astype(numpy.float64, copy=False),
                    frame[mask_index].data.astype(numpy.float64, copy=False),
                    frame_sources,
                    self._eval_shape_terms(frame_sources).T
                )
                for frame, frame_sources in zip(opened_frames, sources)
            ],
            measure_backgrounds,
            require_convergence=False
        )

        if output_dr_fnames:
            for image_index, dr_fname in enumerate(output_dr_fnames):
                with DataReductionFile(dr_fname, 'a') as dr_file:
                    dr_file.add_star_shape_fit(
                        fit_terms_expression=self.configuration[
                            'shape_terms_expression'
                        ],
                        shape_fit_result_tree=fit_result,
                        num_sources=sources[image_index].size,
                        image_index=image_index,
                        fit_variables=self._eval_shape_terms.get_var_names()
                    )
    #pylint: enable=too-many-locals
