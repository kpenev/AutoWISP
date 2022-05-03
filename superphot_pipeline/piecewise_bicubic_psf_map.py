"""Define class for working with piecewise bucubic PSF maps."""

import numpy
from astropy.io import fits
import superphot
from superphot.utils import flux_from_magnitude

from superphot_pipeline import fit_expression
from superphot_pipeline.data_reduction import DataReductionFile

class PiecewiseBicubicPSFMap:
    """Fit and use piecewise bicubic PSF maps."""


    def __init__(self, dr_fname=None):
        """Create a map, loading it from a DR file if one is specified."""

        self.configuration = None
        self._superphot_map = None
        self._eval_shape_terms = None
        if dr_fname is not None:
            self.load(dr_fname)


    #Breaking up seems to make things worse
    #pylint: disable=too-many-locals
    def fit(self,
            fits_fnames,
            sources,
            shape_terms_expression,
            *,
            background_annulus=(6.0, 7.0),
            require_convergence=True,
            output_dr_fnames=None,
            dr_path_substitutions=dict(
                background_version=0,
                shapefit_version=0,
                srcproj_version=0,
            ),
            **fit_star_shape_config):
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

            output_dr_fnames(str iterable):    If not None, should specify one
                data reduction file for each input frame where the fit results
                are saved.

            background_version(int), shapefit_version(int) srcproj_version(int):
                The version numbers to use when saving projected
                photometry sources, background extraciton, and PSF/PRF fitting
                results.

            fit_star_shape_config:    Any required configuration by
                superphot.FitStarShape.__init__()

        Returns:
            None
        """

        self.configuration = dict(shape_terms_expression=shape_terms_expression,
                                  background_annulus=background_annulus,
                                  require_convergence=require_convergence,
                                  **fit_star_shape_config)

        self._eval_shape_terms = fit_expression.Interface(
            shape_terms_expression
        )
        star_shape_fitter = superphot.FitStarShape(
            **fit_star_shape_config
        )

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

        shape_fit_result_tree  = star_shape_fitter.fit(
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

        self._superphot_map = superphot.PiecewiseBicubicPSFMap(
            shape_fit_result_tree
        )

        if output_dr_fnames:
            for image_index, dr_fname in enumerate(output_dr_fnames):
                with DataReductionFile(dr_fname, 'a') as dr_file:
                    dr_file.add_star_shape_fit(
                        fit_terms_expression=self.configuration[
                            'shape_terms_expression'
                        ],
                        shape_fit_result_tree=shape_fit_result_tree,
                        num_sources=sources[image_index].size,
                        image_index=image_index,
                        fit_variables=self._eval_shape_terms.get_var_names(),
                        background_version=background_version,
                        shapefit_version=shapefit_version,
                        srcproj_version=srcproj_version
                    )
    #pylint: enable=too-many-locals

    def load(self, dr_fname, return_sources=False):
        """Read the PSF/PRF map from the given data reduction file."""

        dummy_tool = superphot.SubPixPhot()
        io_tree = superphot.SuperPhotIOTree(dummy_tool)
        self.configuration = dict()

        with DataReductionFile(dr_fname, 'r') as dr_file:
            (
                apphot_data,
                self.configuration['shape_terms_expression']
            ) = dr_file.get_aperture_photometry_inputs(
                shapefit_version=0,
                srcproj_version=0,
                background_version=0
            )
            io_tree.set_aperture_photometry_inputs(**apphot_data)

        self._superphot_map = superphot.PiecewiseBicubicPSFMap(io_tree)

        self.configuration['magnitude_1adu'] = apphot_data['magnitude_1adu']
        self.configuration['grid'] = apphot_data['star_shape_grid']
        self._eval_shape_terms = fit_expression.Interface(
            self.configuration['shape_terms_expression']
        )

        if return_sources:
            sources = numpy.copy(apphot_data['source_data']).astype(
                numpy.dtype(apphot_data['source_data'].dtype.fields)
            )
            print('Source data: ' + repr(apphot_data['source_data'][:3]))
            sources.dtype.names = tuple('flux' if field == 'mag' else field
                                        for field in sources.dtype.names)
            print('Source data: ' + repr(apphot_data['source_data'][:3]))

            sources['flux'] = flux_from_magnitude(
                apphot_data['source_data']['mag'],
                self.configuration['magnitude_1adu']
            )
            return sources

        return None


    def __call__(self, source):
        """
        Evaluate the map for the given source.

        Args:
            source(numpy structured value):    Should specify all variables
                needed by the map.

        Returns:
            superphot.PiecewiseBicubicPSF:
                The PSF of the given source.
        """

        assert source.size == 1

        return self._superphot_map(self._eval_shape_terms(source).flatten())
