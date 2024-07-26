"""Views for displaying diagnostics for the calibration steps."""

from django.http import HttpResponseRedirect
from django.urls import reverse

def display_diagnostics(request, step, imtype):
    return HttpResponseRedirect(reverse('processing:progress'))
