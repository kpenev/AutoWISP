"""The views showing the status of the processing."""
from os import path, scandir
from subprocess import run, Popen, PIPE, TimeoutExpired
import fnmatch
import re
import json

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.views import View
from django.template import loader

from superphot_pipeline.database.interface import Session
from superphot_pipeline.database import processing
from superphot_pipeline.database.user_interface import\
    get_processing_sequence,\
    get_progress,\
    list_channels
from superphot_pipeline.file_utilities import find_fits_fnames
from superphot_pipeline.database.processing import ProcessingManager

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
            [step.name.split('_'), imtype.name, [[0, 0, []] for _ in channels]]
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


class SelectRawImages(View):
    """A view for selecting raw images to add for processing."""

    @staticmethod
    def _get_context(config, search_dir):
        """Return te context required by the file selection template."""

        print(f'Config: {config!r}')
        result = {}
        filename_check = config.get('filename_filter', r'.*\.fits(.fz)?\Z')
        result['filename_filter'] = filename_check
        result['filename_filter_type'] = config.get('filefilter_type',
                                                    'Regular Expression')
        if result['filename_filter_type'] != 'Regular Expression':
            filename_check = fnmatch.translate(filename_check)
        try:
            filename_check = re.compile(filename_check)
        except re.error:
            filename_check = re.compile('')

        dirname_check = config.get('dirname_filter', r'[^.]')
        result['dirname_filter'] = dirname_check
        result['dirname_filter_type'] = config.get('dirfilter_type',
                                                   'Regular Expression')
        if result['dirname_filter_type'] != 'Regular Expression':
            dirname_check = fnmatch.translate(dirname_check)
        try:
            dirname_check = re.compile(dirname_check)
        except re.error:
            print(f'Invalid REX: {dirname_check!r}')
            dirname_check = re.compile('')

        if search_dir is None:
            search_dir = config.get("currentdir", path.expanduser('~'))
            if 'enter_dir' in config:
                search_dir = path.join(search_dir, config['enter_dir'])
        result['file_list'] = []
        result['dir_list'] = []
        with scandir(search_dir) as dir_entries:
            for entry in dir_entries:
                if entry.is_dir():
                    if dirname_check.match(entry.name):
                        result['dir_list'].append(entry.name)
                elif filename_check.match(entry.name):
                    result['file_list'].append(entry.name)

        result['file_list'].sort()
        result['dir_list'].sort()

        head = path.abspath(search_dir)
        parent_dir_list = [('/', 'Computer')]
        while head and head != '/':
            parent_dir_list.insert(1, (head, path.basename(head)))
            head = path.dirname(head)

        result['parent_dir_list'] = parent_dir_list

        print(f'Context: {result!r}')
        return result


    def get(self, request, dirname=None):
        """Display the interface for selecting files."""

        return render(
            request,
            'processing/select_raw_images.html',
            self._get_context(request.GET, dirname)
        )


    def post(self, request, *_args, **_kwargs):
        """Respond to user changing file selection configuration."""

        print(f'POST: {request.POST!r}')
        dir_name = request.POST['currentdir']
        image_list = []
        selected = request.POST['selected']
        if isinstance(selected, str):
            selected = [selected]
        for item_name in selected:
            full_path = path.join(dir_name, item_name)
            if path.isdir(full_path):
                print(f'Adding images under: {full_path!r}')
                image_list.extend(find_fits_fnames(full_path))
            else:
                print(f'Adding single image: {full_path!r}')
                assert path.isfile(full_path)
                image_list.append(full_path)

        try:
            ProcessingManager().add_raw_images(image_list)
        except OSError:
            return HttpResponseRedirect(
                reverse('processing:select_raw_images')
            )


        return HttpResponseRedirect(
            reverse('processing:progress')
        )


def start_processing(request):
    """Run the pipeline to complete any pending processing tasks."""

    global _running_pipeline
    _running_pipeline = Popen([processing.__file__],
                              stdout=PIPE,
                              stderr=PIPE,
                              encoding='ascii')
    return HttpResponseRedirect(reverse('processing:progress'))
# Create your views here.
