"""The views showing the status of the processing."""
from subprocess import run, Popen, PIPE, TimeoutExpired

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages

from superphot_pipeline.database.interface import Session
from superphot_pipeline.database import processing
from superphot_pipeline.database.user_interface import\
    get_processing_sequence,\
    get_progress,\
    list_channels
from superphot_pipeline.database import add_raw_images_gui

_running_pipeline = None

def progress(request):
    """Display the current processing progress."""

    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        channels = sorted(list_channels(db_session))
        channel_index = {channel: i for i, channel in enumerate(channels)}
        processing_sequence = get_processing_sequence(db_session)

        formatted = [
            [step.name, imtype.name, [[0, 0, []] for _ in channels]]
            for step, imtype in processing_sequence
        ]
        for (step, imtype), destination in zip(processing_sequence, formatted):
            final, pending, by_status = get_progress(step.id,
                                                     imtype.id,
                                                     0,
                                                     db_session)
            assert len(final) <= len(channels)
            for channel, _, count in final:
                destination[2][channel_index[channel]][0] = (count or 0)

            for channel, count in pending:
                destination[2][channel_index[channel]][1] = (count or 0)

            for channel, status, count in by_status:
                destination[2][channel_index[channel]][2].append(
                    (status, (count or 0))
                )

    global _running_pipeline
    if _running_pipeline is None:
        running=False
        refresh_seconds=0
    else:
        try:
            stdout, stderr = _running_pipeline.communicate(timeout=1)
            if _running_pipeline.returncode:
                messages.error(
                    request,
                    'Processing failed (return code '
                    f'{_running_pipeline.returncode}).'
                )
                messages.error(request, 'Stdout:\n' + stdout)
                messages.error(request, 'Stderr:\n' + stderr)
                refresh_seconds = 0
                running=False
                _running_pipeline = None
            else:
                messages.info(
                    request,
                    f'Processing finished.'
                )
                messages.info(request, 'Stdout:\n' + stdout)
                messages.info(request, 'Stderr:\n' + stderr)
                refresh_seconds = 0
                running=False
                _running_pipeline = None
        except TimeoutExpired:
            running=True
            refresh_seconds=10

    return render(
        request,
        'processing/progress.html',
        {
            'channels': channels,
            'progress': formatted,
            'refresh_seconds': refresh_seconds,
            'running': running
        }
    )


def add_raw_images(request, files_or_dir):
    """Add new raw images to the database for processing."""

    select_input_result = run([add_raw_images_gui.__file__, files_or_dir],
                              check=False,
                              capture_output=True,
                              encoding='ascii')

    if select_input_result.returncode:
        messages.error(
            request,
            'Selecting images failed (return code '
            f'{select_input_result.returncode}).'
        )
        messages.error(
            request,
            'Stdout:'
            +
            select_input_result.stdout
        )
        messages.error(
            request,
            'Stderr:'
            +
            select_input_result.stderr
        )

    return HttpResponseRedirect(reverse('processing:progress'))


def start_processing(request):
    """Run the pipeline to complete any pending processing tasks."""

    global _running_pipeline
    _running_pipeline = Popen([processing.__file__],
                              stdout=PIPE,
                              stderr=PIPE,
                              encoding='ascii')
    return HttpResponseRedirect(reverse('processing:progress'))
# Create your views here.
