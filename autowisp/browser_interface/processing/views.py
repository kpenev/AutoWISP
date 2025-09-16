"""The views showing the status of the processing."""

# from subprocess import Popen
import subprocess
from sys import executable  # Import the Python interpreter path
import os
import sys
from traceback import format_exc

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
    cmd = [
        executable,  # os.path.join(sys.exec_prefix, 'pythonw.exe') if os.name == 'nt' else
        run_pipeline.__file__,
        request.session["project_db_path"],
    ]
    with open('C:\\WISP\\out\\file_out.txt', mode='a', encoding='utf-8') as f:
        f.write('Starting processing...\n')
        f.write(repr(cmd) + '\n')
    # We don't want processing to stop when this goes out of scope.
    # pylint: disable=consider-using-with
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        flags = 0
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS # | subprocess.CREATE_NO_WINDOW
        with open('C:\\WISP\\out\\file_subprocess.txt', mode='a', encoding='utf-8') as fsub:
            proc = subprocess.Popen(
                # [
                #     'pythonw' if os.name == 'nt' else executable,    # 'pythonw' if os.name == 'nt' else
                #     run_pipeline.__file__,
                #     request.session["project_db_path"],
                # ],  # Use the Python interpreter
                cmd,
                start_new_session=(os.name == "posix"),
                creationflags=(flags),
                stdout=fsub,
                stderr=fsub,
            )
        with open('C:\\WISP\\out\\file_out.txt', mode='a', encoding='utf-8') as f:
            f.write(f'Started subprocess with PID {proc.pid}\n')
    except Exception as e:
        with open('C:\\WISP\\out\\file_out.txt', mode='a', encoding='utf-8') as f:
            f.write(f'Failed to start processing: {format_exc()}\n')
        raise
    with open('C:\\WISP\\out\\file_out.txt', mode='a', encoding='utf-8') as f:
        f.write('Started processing successfully.\n')
    print('Started')
    # pylint: enable=consider-using-with
    return redirect("processing:progress", await_start=0)
