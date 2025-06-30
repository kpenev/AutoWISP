"""Define a class for creating fake raw images."""

import numpy

git_id = "$Id$"


# TODO: dead pixels and/or columns
# (currently can partially be emulated be setting zero flat field)
# TODO: cosmic ray hits
# TODO: charge overflow: partial (i.e. anti-blooming gates) or full
# TODO: non-linearity
class FakeRawImage:
    """
    Create fake raw images with all bells and whistles.

    Currently implemented:
        * sky & stars

        * bias, dark and flat instrumental effects

        * bias and/or dark overscan areas

        * hot pixels (simply set high dark current)

        * discretization noise

        * poisson noise

    Examples:

        >>> from autowisp.fake_image import FakeRawImage
        >>> import numpy
        >>>
        >>> #Create a 1044x1024 image with the first 10 pixels in x being a bias
        >>> #area and the next 10 being a dark area.
        >>> image = FakeRawImage(full_resolution=dict(x=1044, y=1024),
        >>>                      image_area=dict(xmin=20,
        >>>                                      xmax=1044,
        >>>                                      ymin=0,
        >>>                                      ymax=1024))
        >>>
        >>> #Bias level is 12.5 ADU
        >>> image.add_bias(12.5)
        >>>
        >>> #Dark current is 2.3 ADU/s except for a hot column at x=100 with 10x
        >>> #the dark current.
        >>> dark = numpy.full((1044, 1024), 12.5)
        >>> dark[:, 100] = 125.0
        >>> image.set_dark(
        >>>     rate=dark,
        >>>     areas=[dict(xmin=10, xmax=20, ymin=0, ymax=1024)]
        >>> )
        >>>
        >>> #Define a flat field which is a quadratic function in both x and y.
        >>> x, y = numpy.meshgrid(numpy.arange(1024), numpy.arange(1024))
        >>> flat = (2.0 - ((x - 512.0) / 512.0)**2) / 2.0
        >>> image.set_flat_field(flat)
        >>>
        >>> #Add simple stars
        >>> star = numpy.array([[0.25, 0.50, 0.25],
        >>>                     [0.50, 1.00, 0.50],
        >>>                     [0.25, 0.50, 0.25]])
        >>> sky_flux = numpy.zeros((1024, 1024))
        >>> for star_x in numpy.arange(50.0, 1024.0 - 50.0, 50.0):
        >>>     for star_y in numpy.arange(50.0, 1024.0 - 50.0, 50.0):
        >>>         sky_flux[star_y - 1 : star_y + 2,
        >>>                  star_x - 1 : star_x + 2] = star
        >>> image.set_sky(sky_flux)
        >>>
        >>> #Get image with the given parameters with 30s exposure
        >>> exp1 = image(5)
    """

    def __init__(self, full_resolution, image_area, gain=1.0):
        """
        Start creating a fake image with the given parameters.

        Args:
            full_resolution:    The full resolution of the image to create,
                including the light sensitive area, but also overscan areas etc.
                Should be dict(x=<int>, y=<int>).

            image_area:    The light sensitivy part of the image. The format is:
                `dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)`

            gain:    The gain to assume for the A to D converter in electrons
                per ADU. Setting a non-finite value (+-infinity or NaN) disables
                poisson noise.
        """

        self._pixels = numpy.zeros((full_resolution["y"], full_resolution["x"]))
        self._image_offset = {"x": image_area["xmin"], "y": image_area["ymin"]}
        self._image = self._pixels[
            image_area["ymin"] : image_area["ymax"],
            image_area["xmin"] : image_area["xmax"],
        ]
        self._gain = gain
        self._dark_rate = 0.0
        self._flat = 1.0
        self._sky = 0.0

    def add_bias(self, bias, units="ADU"):
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

        assert units in ["ADU", "electrons"]

        self._pixels += bias * (1.0 if units == "ADU" else 1.0 / self._gain)

    def set_dark(self, rate, areas, units="ADU"):
        """
        Define the rate at which dark current accumulates.

        Args:
            rate:    The noiseless rate per unit time at which dark current
                accumulates. See `bias` argument of `add_bias` for details on
                the possible formats.

            areas:    List of areas specified using the same format as the
                `image_area` argument of __init__ specifying the areas which
                accumulate dark current but no light.

            units:    Is the dark rate specified in 'ADU' or 'electrons' per
                unit time.

        Returns:
            None
        """

        dark_rate_multiplier = 1.0 if units == "ADU" else 1.0 / self._gain

        self._dark_rate = numpy.zeros(self._pixels.shape)
        image_y_res, image_x_res = self._image.shape
        self._dark_rate[
            self._image_offset["y"] : self._image_offset["y"] + image_y_res,
            self._image_offset["x"] : self._image_offset["x"] + image_x_res,
        ] = dark_rate_multiplier

        for dark_area in areas:
            self._dark_rate[
                dark_area["ymin"] : dark_area["ymax"],
                dark_area["xmin"] : dark_area["xmax"],
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

        assert flat.shape == self._image.shape
        self._flat = flat

    def set_sky(self, sky_flux, units="ADU"):
        """
        Define the flux arriving from the sky (with or without stars).

        Args:
            sky_flux:    The image that a perfect imaging system (no bias, dark
                of flat) would see. Should only cover the imaging area.

            units:    Is the sky flux specified in 'ADU' or 'electrons' per
                unit time.

        Returns:
            None
        """

        assert sky_flux.shape == self._image.shape
        self._sky = sky_flux * (1.0 if units == "ADU" else 1.0 / self._gain)

    def __call__(self, exposure):
        """
        Simulate an exposure of the given duration.

        Args:
            exposure:    The amount of time to expose for in units consistent
                with the units used for the rates specified.

        Returns:
            2-D numpy array:
                The simulated exposure image sprinkled with random poisson noise
                if gain is finite.
        """

        image = self._pixels + self._dark_rate * exposure

        x_res, y_res = self._image.shape
        image[
            self._image_offset["y"] : self._image_offset["y"] + y_res,
            self._image_offset["x"] : self._image_offset["x"] + x_res,
        ] = (
            self._sky * self._flat * exposure
        )

        if numpy.isfinite(self._gain):
            image = (
                numpy.random.poisson(numpy.around(image * self._gain))
                / self._gain
            )

        return numpy.around(image).astype("int")
