"""A collection of overscan correction methods (see Calibrator class docs)."""

from abc import ABC, abstractmethod
import numpy

from superphot_pipeline.pipeline_exceptions import ConvergenceError
from superphot_pipeline import Processor
from superphot_pipeline.image_calibration.fits_util import assemble_channels

git_id = '$Id$'

#pylint: disable=too-few-public-methods
#It still makes sense to make a class with two methods (including __call__).

class Base(ABC, Processor):
    """The minimal intefrace that must be provided by overscan methods."""

    @abstractmethod
    def document_in_fits_header(self, header):
        """Document last overscan correction by updating given FITS header."""

    #Intentional
    #pylint: disable=arguments-differ
    @abstractmethod
    def __call__(self,
                 *,
                 raw_fname,
                 raw_hdu,
                 overscans,
                 image_area,
                 gain,
                 split_channels=None):
        """
        Return the overscan correction and its variance for the given image.

        Args:
            raw_fname:    The raw image(s) for which to find the
                overscan correction. Should be dictionary of filenames if
                processing multi-channel dataset that is split in different
                files.

            raw_hdu:    The HDU to read from the raw file. Int or dict indexed
                by channel name.

            overscans:    A list of the areas on the image to use when
                determining the overscan correction. Each area is specified as
                ``dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)``
                For multi-channel data the coordinates are for the un-split
                image.

            image_area:    The area in raw_image for which to calculate the
                overscan correction. The format is the same as a single
                overscan area.

            gain:    The value of the gain to assume for the raw image
                (electrons/ADU).

            split_channels:    See :class:`Calibrator`.

        Returns:
            dict: Dictionary with items:

                * correction:    A 2-D numpy array with the same resolution
                    as the image_area giving the correction to subtract from
                    each pixel.

                * variance:    An estimate of the variance in the
                    overscan_correction entries (in ADU).
        """
    #pylint: enable=arguments-differ

class Median(Base):
    """
    Correction is median of all overscan pixels with iterative outlier reject.

    The correction is computed as the median of all overscan pixels. After that,
    pixels that are too far from the median are excluded and the process starts
    from the beginning until no pixels are rejected or the maximum number of
    rejection iterations is reached.

    Public attributes exactly match the  :meth:`__init__` arguments.
    """

    def __init__(self,
                 reject_threshold=5.0,
                 max_reject_iterations=10,
                 min_pixels=100,
                 require_convergence=False):
        """
        Create a median ovescan correction method.

        Args:
            reject_threshold:    Pixels that differ by more than
                reject_threshold standard deviations from the median are
                rejected at each iteration.

            max_reject_iterations:    The maximum number of outlier rejection
                iterations to perform. If this limit is reached, either the
                latest result is accepted, or an exception is raised depending
                on **require_convergence**.

            require_convergence:    If this is False and the maximum number of
                rejection iterations is reached, the last median computed is the
                accepted result. If this is True, hitting the
                max_reject_iterations limit throws an exception.

            min_pixels:    If iterative rejection drives the number of
                acceptable pixels below this value an exception is raised.

        Returns:
            None

        Notes:
            Initializes the following private attributes to None, which indicate
            the state of the last overscan correction calculation:

            **_last_num_reject_iter**:    The number of rejection iterations
                used by the last overscan correction calculation.

            **_last_num_pixels**:    The number of unrejected pixels the last
                overscan correction was based on.

            **_last_converged**:    Did the last overscan calculation converge?

        """

        self.reject_threshold = reject_threshold
        self.max_reject_iterations = max_reject_iterations
        self.min_pixels = min_pixels
        self.require_convergence = require_convergence

        self._last_num_reject_iter = None
        self._last_num_pixels = None
        self._last_converged = None

    #pylint: disable=anomalous-backslash-in-string
    #Triggers on doxygen commands.
    def document_in_fits_header(self, header):
        """
        Document the last calculated overscan correction to header.

        Notes:
            Adds the following keywords to the header::

                OVSCNMTD = Iterative rejection median
                           / Overscan correction method

                OVSCREJM = ###
                           / Maximum number of allowed overscan rejection
                           iterations.

                OVSCMINP = ###
                           / Minimum number of pixels to base correction on

                OVSCREJI = ###
                           / Number of overscan rejection iterations applied

                OVSCNPIX = ###
                           / Actual number of pixels used to calc overscan

                OVSCCONV = T/F
                           / Did the last overscan correction converge

        Args:
            header:    The FITS header to add the keywords to.

        Returns:
            None
        """
    #pylint: enable=anomalous-backslash-in-string

        header['OVSCNMTD'] = ('Iterative rejection median',
                              'Overscan correction method')

        header['OVSCREJM'] = (
            self.max_reject_iterations,
            'Maximum number of allowed overscan rejection iterations.'
        )

        header['OVSCMINP'] = (self.min_pixels,
                              'Minimum number of pixels to base correction on')

        header['OVSCREJI'] = (self._last_num_reject_iter,
                              'Number of overscan rejection iterations applied')

        header['OVSCNPIX'] = (self._last_num_pixels,
                              'Actual number of pixels used to calc overscan')

        header['OVSCCONV'] = (self._last_converged,
                              'Did the last overscan correction converge')

    def __call__(self,
                 *,
                 raw_image,
                 overscans,
                 image_area,
                 gain,
                 split_channels=None):
        """
        See Base.__call__
        """

        if split_channels is None:
            split_channels = {None: slice(None)}

        def get_overscan_pixel_values():
            """
            Return a numpy array of the pixel values to base correctiono on.

            Args:
                None

            Retruns:
                numpy.array:
                    The values of the pixels to use when calculating the
                    overscan correction. Even if overscan areasoverlap only a
                    single copy of each pixel is included.
            """

            need_area = {'xmin': numpy.inf,
                         'ymin': numpy.inf,
                         'xmax': 0,
                         'ymax': 0}
            num_overscan_pixels = 0
            for overscan_area in overscans:
                for coord in 'xy':
                    need_area[coord + 'min'] = min(
                        need_area[coord + 'min'],
                        overscan_area[coord + 'min']
                    )
                    need_area[coord + 'max'] = max(
                        need_area[coord + 'max'],
                        overscan_area[coord + 'max']
                    )
                    num_overscan_pixels += (
                        (overscan_area['ymax'] - overscan_area['ymin'])
                        *
                        (overscan_area['xmax'] - overscan_area['xmin'])
                    )


            not_included = numpy.full(raw_image.shape, True)

            overscan_values = numpy.empty(num_overscan_pixels)
            new_value_start = 0

            for area in overscans:
                overscan_area = {
                    key: area[key] - need_area[key[0] + 'min']
                    for key in area.keys()
                }
                new_pixels = raw_image[
                    overscan_area['ymin'] : overscan_area['ymax'],
                    overscan_area['xmin'] : overscan_area['xmax'],
                ][
                    not_included[
                        overscan_area['ymin'] : overscan_area['ymax'],
                        overscan_area['xmin'] : overscan_area['xmax'],
                    ]
                ]
                overscan_values[
                    new_value_start : new_value_start + new_pixels.size
                ] = new_pixels

                new_value_start += new_pixels.size

                not_included[
                    overscan_area['ymin'] : overscan_area['ymax'],
                    overscan_area['xmin'] : overscan_area['xmax'],
                ] = False

            return overscan_values

        overscan_values = get_overscan_pixel_values()
        self._last_num_reject_iter = 0
        num_rejected = 1
        while (
                num_rejected > 0
                and
                self._last_num_reject_iter <= self.max_reject_iterations
                and
                overscan_values.size >= self.min_pixels
        ):
            start_num_values = overscan_values.size
            correction = numpy.median(overscan_values)
            median_deviations = numpy.square(overscan_values - correction)
            deviation_scale = median_deviations.sum() / (start_num_values - 1)
            overscan_values = overscan_values[
                median_deviations
                <=
                self.reject_threshold**2 * deviation_scale
            ]
            num_rejected = start_num_values - overscan_values.size
            self._last_num_reject_iter += 1

        if overscan_values.size < self.min_pixels:
            raise ConvergenceError(
                'Median overscan: Too few pixels remain '
                f'({overscan_values.size:d}) after '
                f'{self._last_num_reject_iter:d} rejection iterations.'
            )
        if num_rejected > 0 and self.require_convergence:
            assert self._last_num_reject_iter > self.max_reject_iterations
            raise ConvergenceError(
                'Median overscan correction iterative rejection exceeded the '
                f'maximum number ({self.max_reject_iterations:d}) of iteratons '
                'allowed'
            )

        self._last_num_pixels = overscan_values.size
        self._last_converged = True

        image_shape = (image_area['ymax'] - image_area['ymin'],
                       image_area['xmax'] - image_area['xmin'])
        return {
            'correction': numpy.full(image_shape, correction),
            'variance': numpy.full(image_shape,
                                   deviation_scale / overscan_values.size)
        }

#pylint: enable=too-few-public-methods
