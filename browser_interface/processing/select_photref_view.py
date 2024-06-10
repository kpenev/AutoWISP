"""Implement the view for selecting single photometric reference."""

from io import BytesIO
from base64 import b64encode

import numpy
from PIL import Image
from PIL.ImageTransform import AffineTransform
from django.shortcuts import render
from matplotlib import colors

from astropy.io import fits
from astropy.visualization import ZScaleInterval


def log_scale(pixel_values, exponent=1000):
    """Perform the same log-transform as DS9."""

    return numpy.log(exponent * pixel_values + 1) / numpy.log(exponent)


def select_photref(request,
                   values_range='zscale',
                   values_transform=None,
                   zoom=1.0):
    """
    A view for reviewing calibrated frames to select photometric rerference.
    """

    png_stream = BytesIO()
    with fits.open(
        '/Users/kpenev/tmp/10-483557_2_R.fits.fz',
        'readonly'
    ) as frame:
        if values_range == 'zscale':
            limits = ZScaleInterval().get_limits(frame[1].data)
        elif values_range == 'minmax':
            limits = frame[1].data.min(), frame[1].data.max()
        else:
            limits = tuple(int(lim.strip()) for lim in values_range.split(','))
        pixel_values = colors.Normalize(
            *limits,
            True
        )(frame[1].data)
        if values_transform is not None and values_transform != 'None':
            transform_args = values_transform.split('-')
            transform = globals().get(transform_args.pop(0) + '_scale')
            transform_args = [float(arg) for arg in transform_args]
            pixel_values = transform(pixel_values, *transform_args)
        print(f'Limits: {limits!r}')
        scaled_pixels = (
            pixel_values
            *
            255
        ).astype('uint8')
        image = Image.fromarray(scaled_pixels)
        apply_zoom = AffineTransform((1.0/zoom, 0, 0, 0, 1.0/zoom, 0.0))
        image.transform(
            size=(int(image.size[0] * zoom), int(image.size[1] * zoom)),
            method=apply_zoom
        ).save(
            png_stream,
            'png'
        )
    return render(
        request,
        'processing/select_photref.html',
        {
            'image': b64encode(png_stream.getvalue()).decode('utf-8'),
            'range': values_range,
            'transform': values_transform
        }
    )
