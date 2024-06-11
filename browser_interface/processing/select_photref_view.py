"""Implement the view for selecting single photometric reference."""

from io import BytesIO
from base64 import b64encode

import numpy
from PIL import Image
from PIL.ImageTransform import AffineTransform
from django.views import View
from django.shortcuts import render
from matplotlib import colors

from astropy.io import fits
from astropy.visualization import ZScaleInterval

from autowisp.database.processing import ProcessingManager
from autowisp.database.interface import Session
from autowisp.database.user_interface import get_processing_sequence



class SelectPhotRef(View):
    """
    A view for reviewing calibrated frames to select photometric rerference.
    """

    @staticmethod
    def _log_transform(pixel_values, parameter=1000.0):
        """Perform the same log-transform as DS9."""

        return numpy.log(parameter * pixel_values + 1) / numpy.log(parameter)


    @staticmethod
    def _pow_transform(pixel_values, parameter=1000.0):
        """Perform the same pow transfom as DS9."""

        return (numpy.power(parameter, pixel_values) - 1.0) / parameter


    @staticmethod
    def _sqrt_transform(pixel_values):
        """Use square root of the pixel values as intensity."""

        return numpy.sqrt(pixel_values)


    @staticmethod
    def _square_transform(pixel_values):
        """Use the square of the pixel values as intensity."""

        return numpy.square(pixel_values)


    @staticmethod
    def _asinh_transform(pixel_values):
        """The asinh transform of DS9."""

        return numpy.arcsinh(10.0 * pixel_values) / 3.0


    @staticmethod
    def _sinh_transform(pixel_values):
        """The sinh transform of DS9."""

        return numpy.sinh(3.0 * pixel_values) / 10.0


    @staticmethod
    def _get_missing_photref(request):
        """Add all frame sets missing photometric reference to the session."""

        processing = ProcessingManager(dummy=True)
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            pending_photref = processing.get_pending(
                db_session,
                [entry for entry in get_processing_sequence(db_session)
                 if entry[0].name == 'fit_magnitudes'],
            )
            for (step_id, imtype_id), pending_images in pending_photref.items():
                by_photref = processing.group_pending_by_conditions(
                    pending_images,
                    db_session,
                    match_observing_session=False,
                    step_id=step_id,
                    masters_only=False
                )


    def get(self,
            request,
            values_range='zscale',
            values_transform=None,
            zoom=1.0):
        """Display the interface for reviewing canditate reference frames."""

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
                limits = tuple(int(lim.strip())
                               for lim in values_range.split(','))
            pixel_values = colors.Normalize(
                *limits,
                True
            )(frame[1].data)
            if values_transform is not None and values_transform != 'None':
                transform_args = values_transform.split('-')
                transform = getattr(self,
                                    '_' + transform_args.pop(0) + '_transform')
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
                'transform': values_transform,
                'transform_list': [
                    entry[1:].split('_', 1)[0]
                    for entry in vars(SelectPhotRef)
                    if(
                        entry[0] == '_'
                        and
                        entry.endswith('_transform')
                    )
                ]
            }
        )
