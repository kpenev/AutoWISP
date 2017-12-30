"""Define a class for creating fake raw images."""

from astropy.io import fits

import numpy

git_id = '$Id$'

class FakeRawImage:
    """
    Create fake raw images with all bells and whistles.

    Currently implemented:
        * bias, dark and flat instrumental effects

        * bias and/or dark overscan areas

        * hot pixels (simply set high dark current)

    To do:
        * poisson noise

        * dead pixels and/or columns (can partially be emulated be setting
          zero flat field).

        * cosmic ray hits

        * stars - using SuperPhot's fake image tools

        * charge overflow: partial (i.e. anti-blooming gates or full)

        * non-linearity
    """

    def __init__(self, full_resolution, image_area, gain=1.0):
        """
        Start creating a fake image with the given parameters.

        Args:
            full_resolution:    The full resolution of the image to create,
                including the light sensitive area, but also overscan areas etc.

            image_area:    The light sensitivy part of the image. The format is:
                `dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)`

            gain:    The gain to assume for the A to D converter in electrons
                per ADU. Setting a non-finite value (+-infinity or NaN) disables
                poisson noise.
        """

        self._pixels = numpy.zeros((full_resolution['y'], full_resolution['x']))
        self._image_offset = dict(x=image_area['xmin'], y=image_area['ymin'])
        self._image = self._pixels[image_area['ymin'] : image_area['ymax'],
                                   image_area['xmin'] : image_area['xmax']]
        self._gain = gain
        self._dark_rate = 0.0
        self._flat = 1.0

    def add_bias(self, bias, units='ADU'):
        """
        Add a bias level to the full image.

        Args:
            bias:    The noiseless bias level to add. Should be a single value,
                a single row or column matching or a 2-D image with the y index
                being first. The row, column or the image should matchthe full
                reselotion of the fake image, not just the image area.

            units:    Is the bias level specified in 'electrons' or in amplifier
                units ('ADU').

        Returns:
            None
        """

        assert units in ['ADU', 'electrons']

        self._pixels += bias * (1.0 if units == 'ADU' else self._gain)

    def set_dark(self, rate, areas, units='ADU'):
        """
        Define the rate at which dark current accumulates.

        Args:
            rate:    The noiseless rate per unit time at which dark current
                accumulates. See `bias` argument of `add_bias` for details on
                the possible formats.

            areas:    Areas specified using the same format as the `image_area`
                argument of __init__ specifying the areas which accumulate dark
                current but no light.

            units:    Is the dark rate specified in 'ADU' or 'electrons' per
                unit time.

        Returns:
            None
        """


        dark_rate_multiplier = (1.0 if units == 'ADU' else self._gain)

        self._dark_rate = numpy.zeros(self._pixels.shape())
        image_y_res, image_x_res = self._image.shape
        self._dark_rate[
            self._image_offset['y'] : self._image_offset['y'] + image_y_res,
            self._image_offset['x'] : self._image_offset['x'] + image_x_res,
        ] = dark_rate_multiplier

        for dark_area in areas:
            self._dark_rate[
                dark_area['ymin'] : dark_area['ymax'],
                dark_area['xmin'] : dark_area['xmax']
            ] = dark_rate_multiplier

        self._dark_rate *= rate



    def set_flat_field(self, flat):
        """
        Define the sensitivity map of the fake imaging system.

        Args:
            flat:    The noiseless map of the throughput of the system times the
                sensitivy of each pixel. Should have the same resolution as the
                image area (not the full image).

        Returns:
            None
        """

        self._flat = flat
