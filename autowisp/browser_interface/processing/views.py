"""The views showing the status of the processing."""

import subprocess
import threading
from sys import executable
import os
import sys
import logging

from django.shortcuts import redirect
import platformdirs

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
from .error_views import (
    error_list,
    error_detail_view,
    toggle_error_resolved,
    delete_error_view,
    download_crash_report,
)

# pylint: enable=unused-import


def start_processing(request):
    """Run the pipeline to complete any pending processing tasks."""
    cmd = [
        executable,
        run_pipeline.__file__,
        get_project_home(),
    ]

    selected_steps = []
    selected_step_imtypes = []

    if request.method == "POST":
        raw_tokens = request.POST.getlist("steps")

        # Store which steps were selected
        request.session["selected_step_tokens"] = list(raw_tokens)

        seen_base = set()

        for token in raw_tokens:
            if not token:
                continue
            parts = token.rsplit("_", 1)
            if len(parts) == 2:
                base_step, imtype = parts
                selected_step_imtypes.append(f"{base_step}:{imtype}")
                if base_step not in seen_base:
                    selected_steps.append(base_step)
                    seen_base.add(base_step)
            else:
                if token not in seen_base:
                    selected_steps.append(token)
                    seen_base.add(token)

    logging.info(
        "start_processing: steps=%s step_imtypes=%s",
        selected_steps,
        selected_step_imtypes,
    )

    if selected_steps:
        cmd.extend(["--steps", *selected_steps])
    if selected_step_imtypes:
        cmd.extend(["--step-imtypes", *selected_step_imtypes])

    logging.info("start_processing: cmd=%s", cmd)

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
    print("Started")
    return redirect("processing:progress", await_start=0)
