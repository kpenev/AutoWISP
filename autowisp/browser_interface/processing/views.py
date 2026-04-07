"""The views showing the status of the processing."""

import subprocess
import threading
from sys import executable
import os
import sys

from django.shortcuts import redirect
import platformdirs

# from django.contrib import messages
# from django.template import loader

from autowisp import run_pipeline
from autowisp.database.interface import get_project_home

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
from .display_fits_util import update_fits_display

# pylint: enable=unused-import
def start_processing(request):
    """Run the pipeline to complete any pending processing tasks."""
    cmd = [
        executable,
        run_pipeline.__file__,
        get_project_home(),
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
        proc = subprocess.Popen(
            cmd,
            start_new_session=(os.name == "posix"),
            stdout=outf,
            stderr=outf,
        )
    if os.name == "posix":
        # Reap the child when it exits so it does not become a zombie.
        # On Linux the double-fork in run_pipeline.py orphans the
        # grandchild to init (which reaps it), but the intermediate
        # child still needs to be waited on here.  On macOS there is
        # no double-fork so this thread is the only reaper.
        threading.Thread(target=proc.wait, daemon=True).start()
    print('Started')
    # pylint: enable=consider-using-with
    return redirect("processing:progress", await_start=0)
