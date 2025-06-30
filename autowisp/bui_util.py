"""View for displaying FITS images."""

from io import BytesIO
from base64 import b64encode
from os.path import exists

from PIL import Image
from matplotlib import colors
import numpy
from astropy.io import fits
from astropy.visualization import ZScaleInterval


def _log_transform(pixel_values, parameter=1000.0):
    """Perform the same log-transform as DS9."""

    return numpy.log(parameter * pixel_values + 1) / numpy.log(parameter)


def _pow_transform(pixel_values, parameter=1000.0):
    """Perform the same pow transfom as DS9."""

    return (numpy.power(parameter, pixel_values) - 1.0) / parameter


def _sqrt_transform(pixel_values):
    """Use square root of the pixel values as intensity."""

    return numpy.sqrt(pixel_values)


def _square_transform(pixel_values):
    """Use the square of the pixel values as intensity."""

    return numpy.square(pixel_values)


def _asinh_transform(pixel_values):
    """The asinh transform of DS9."""

    return numpy.arcsinh(10.0 * pixel_values) / 3.0


def _sinh_transform(pixel_values):
    """The sinh transform of DS9."""

    return numpy.sinh(3.0 * pixel_values) / 10.0


def encode_fits(fits_fname, values_range, values_transform):
    """Display transformed & scaled FITS image to user."""

    if not exists(fits_fname):
        raise RuntimeError(
            f"Requested FITS file ({fits_fname}) does not exist!"
        )
    png_stream = BytesIO()
    with fits.open(fits_fname, "readonly") as frame:
        if values_range == "zscale":
            limits = ZScaleInterval().get_limits(frame[1].data)
        elif values_range == "minmax":
            limits = frame[1].data.min(), frame[1].data.max()
        else:
            limits = tuple(int(lim.strip()) for lim in values_range.split(","))
        pixel_values = colors.Normalize(*limits, True)(frame[1].data)
        if values_transform is not None and values_transform != "None":
            transform_args = values_transform.split("-")
            transform = globals()["_" + transform_args.pop(0) + "_transform"]
            transform_args = [float(arg) for arg in transform_args]
            pixel_values = transform(pixel_values, *transform_args)
        scaled_pixels = (pixel_values * 255).astype("uint8")
        image = Image.fromarray(scaled_pixels)
        # apply_zoom = AffineTransform((1.0/zoom, 0, 0, 0, 1.0/zoom, 0.0))
        # image.transform(
        #    size=(int(image.size[0] * zoom), int(image.size[1] * zoom)),
        #    method=apply_zoom
        # ).save(
        #    png_stream,
        #    'png'
        # )
        image.save(png_stream, "png")

    return {
        "image": b64encode(png_stream.getvalue()).decode("utf-8"),
        "transform_list": [
            entry[1:].split("_", 1)[0]
            for entry in globals()
            if (entry[0] == "_" and entry.endswith("_transform"))
        ],
    }


def hex_color(color_tuple):
    """Return string of hex color give tuple of 0-1 float values."""

    return "#" + "".join(
        [f"{int(numpy.round(c * 255)):02x}" for c in color_tuple[:3]]
    )
