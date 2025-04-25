"""The views showing the status of the processing."""
from subprocess import Popen
from sys import executable  # Import the Python interpreter path

from django.http import HttpResponseRedirect
from django.urls import reverse
#from django.contrib import messages
#from django.template import loader

from autowisp.database import image_processing

#This module should collect all views
#pylint: disable=unused-import
from .log_views import review, review_single
from .select_raw_view import SelectRawImages
from .progress_view import progress
from .select_photref_views import\
    select_photref_target,\
    select_photref_image,\
    record_photref_selection
from .tune_starfind_views import\
    select_starfind_batch,\
    tune_starfind,\
    find_stars,\
    project_catalog,\
    save_starfind_config
from .detrending_diagnostics_views import\
    display_detrending_diagnostics,\
    refresh_detrending_diagnostics,\
    update_detrending_diagnostics_plot,\
    download_detrending_diagnostics_plot
from .display_fits_util import update_fits_display
#pylint: enable=unused-import


def start_processing(_request):
    """Run the pipeline to complete any pending processing tasks."""

    #We don't want processing to stop when this goes out of scope.
    #pylint: disable=consider-using-with
    Popen([executable, image_processing.__file__],  # Use the Python interpreter
          start_new_session=True,
          encoding='ascii')
    #pylint: enable=consider-using-with
    return HttpResponseRedirect(reverse('processing:progress'))
