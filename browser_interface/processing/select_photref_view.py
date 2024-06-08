"""Implement the view for selecting single photometric reference."""

from io import BytesIO
from base64 import b64encode

from matplotlib import pyplot
from django.shortcuts import render

def select_photref(request):
    """
    A view for reviewing calibrated frames to select photometric rerference.
    """

    pyplot.plot(range(10), range(10), 'ok')
    png_stream = BytesIO()
    pyplot.savefig(png_stream, format='png')
    return render(
        request,
        'processing/select_photref.html',
        {'image': b64encode(png_stream.getvalue())}
    )
