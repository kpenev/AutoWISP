"""The views showing the status of the processing."""

import subprocess
from sys import executable
import os
import sys
from traceback import format_exc

from django.shortcuts import redirect
import platformdirs

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
    cmd = [
        executable,
        run_pipeline.__file__,
        request.session["project_home"],
    ]
    # We don't want processing to stop when this goes out of scope.
    # pylint: disable=consider-using-with
    sys.stdout.flush()
    sys.stderr.flush()
    with open(
        os.path.join(
            platformdirs.user_data_dir("autowisp"), "run_pipeline.out"
        ),
        "w",
        encoding="utf-8",
        buffering=1,
    ) as outf:
        subprocess.Popen(
        cmd,
        start_new_session=(os.name == "posix"),
        stdout=outf,
        stderr=outf,
        )
    print('Started')
    # pylint: enable=consider-using-with
    return redirect("processing:progress", await_start=0)
