"""The views showing the status of the processing."""

from subprocess import Popen
from sys import executable  # Import the Python interpreter path
import os

from django.shortcuts import redirect

# from django.contrib import messages
# from django.template import loader

from autowisp import run_pipeline

# This module should collect all views
# pylint: disable=unused-import
from .log_views import review, review_single
from .select_raw_view import SelectRawImages
from .progress_view import progress
from .select_photref_views import (
    select_photref_target,
    select_photref_image,
    record_photref_selection,
)
from .tune_starfind_views import (
    select_starfind_batch,
    tune_starfind,
    find_stars,
    project_catalog,
    save_starfind_config,
)
from .detrending_diagnostics_views import (
    display_detrending_diagnostics,
    refresh_detrending_diagnostics,
    update_detrending_diagnostics_plot,
    download_detrending_diagnostics_plot,
)
from .display_fits_util import update_fits_display

# pylint: enable=unused-import


def start_processing(request):
    """Run the pipeline to complete any pending processing tasks."""

    print('Starting')
    # We don't want processing to stop when this goes out of scope.
    # pylint: disable=consider-using-with
    Popen(
        [
            'pythonw' if os.name == 'nt' else executable,
            run_pipeline.__file__,
            request.session["project_db_path"],
        ],  # Use the Python interpreter
        start_new_session=True,
        encoding="ascii",
    )
    print('Started')
    # pylint: enable=consider-using-with
    return redirect("processing:progress", await_start=0)
