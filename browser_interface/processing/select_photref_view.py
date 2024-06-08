"""Implement the view for selecting single photometric reference."""

from io import BytesIO
from base64 import b64encode

import numpy
from PIL import Image
from django.shortcuts import render

def select_photref(request):
    """
    A view for reviewing calibrated frames to select photometric rerference.
    """

    png_stream = BytesIO()
    Image.fromarray(
        numpy.random.randint(0, 255, size=(500, 500), dtype='uint8')
    ).save(png_stream, 'png')
    return render(
        request,
        'processing/select_photref.html',
        {'image': b64encode(png_stream.getvalue()).decode('utf-8')}
    )
