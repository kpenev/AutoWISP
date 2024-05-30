"""The views showing the status of the processing."""
from os import path, scandir
from subprocess import Popen, PIPE
import fnmatch
import re
from socket import getfqdn
import logging

from sqlalchemy import select, func, and_
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
#from django.contrib import messages
from django.views import View
#from django.template import loader
from psutil import pid_exists

from autowisp.database.interface import Session
from autowisp.database import processing
from autowisp.database.user_interface import\
    get_processing_sequence,\
    get_progress,\
    list_channels
from autowisp.file_utilities import find_fits_fnames
from autowisp.database.processing import ProcessingManager
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    ImageProcessingProgress,\
    ProcessingSequence,\
    ImageType,\
    Step
#pylint: enable=no-name-in-module


_datetime_fmt = '%Y%m%d %H:%i:%s'


def progress(request):
    """Display the current processing progress."""

    context = {
        'running': False,
        'refresh_seconds': 0
    }
    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        context['channels'] = sorted(list_channels(db_session))
        channel_index = {channel: i
                         for i, channel in enumerate(context['channels'])}
        processing_sequence = get_processing_sequence(db_session)

        context['progress'] = [
            [
                step.name.split('_'),
                imtype.name,
                [[0, 0, []] for _ in context['channels']],
                []
            ]
            for step, imtype in processing_sequence
        ]
        for (step, imtype), destination in zip(processing_sequence,
                                               context['progress']):
            final, pending, by_status = get_progress(step.id,
                                                     imtype.id,
                                                     0,
                                                     db_session)
            assert len(final) <= len(context['channels'])
            for channel, _, count in final:
                destination[2][channel_index[channel]][0] = (count or 0)

            for channel, count in pending:
                destination[2][channel_index[channel]][1] = (count or 0)

            for channel, status, count in by_status:
                destination[2][channel_index[channel]][2].append(
                    (status, (count or 0))
                )
            destination[3] = db_session.execute(
                select(
                    ImageProcessingProgress.id,
                    func.date_format(ImageProcessingProgress.started,
                                     _datetime_fmt),
                    func.date_format(ImageProcessingProgress.finished,
                                     _datetime_fmt)
                ).where(
                    ImageProcessingProgress.step_id == step.id,
                    ImageProcessingProgress.image_type_id == imtype.id
                )
            ).all()

    for check_pid in db_session.scalars(
        select(
            ImageProcessingProgress.process_id
        ).where(
            #pylint: disable=singleton-comparison
            ImageProcessingProgress.finished == None,
            #pylint: enable=singleton-comparison
            ImageProcessingProgress.host == getfqdn()
        ).group_by(
            ImageProcessingProgress.process_id
        )
    ):
        if pid_exists(check_pid):
            context['running'] = True
            context['refresh_seconds'] = 5

    return render(request, 'processing/progress.html', context)


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


def start_processing(_request):
    """Run the pipeline to complete any pending processing tasks."""

    #We don't want processing to stop when this goes out of scope.
    #pylint: disable=consider-using-with
    Popen([processing.__file__],
          start_new_session=True,
          encoding='ascii')
    #pylint: enable=consider-using-with
    return HttpResponseRedirect(reverse('processing:progress'))


def review(request,
           selected_processing_id,
           min_log_level='WARNING'):
    """
    A view for going through pipeline logs and diagnostics.

    Args:
        progress_id(int):    The progress ID for which to display logs and/or
            diagnostics.
    """

    context = {'selected_processing_id': selected_processing_id,
               'min_log_level': min_log_level}
    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        selected_progress = db_session.execute(
            select(
                ImageProcessingProgress.id,
                ImageProcessingProgress.step_id,
                ImageProcessingProgress.image_type_id,
                func.date_format(ImageProcessingProgress.started,
                                 _datetime_fmt),
                func.date_format(ImageProcessingProgress.finished,
                                 _datetime_fmt),
            ).where(
                ImageProcessingProgress.id == selected_processing_id,
            )
        ).one()

        context['reviewable'] = db_session.execute(
            select(
                ImageProcessingProgress.id,
                func.date_format(ImageProcessingProgress.started,
                                 _datetime_fmt),
                func.date_format(ImageProcessingProgress.finished,
                                 _datetime_fmt)
            ).where(
                (
                    ImageProcessingProgress.step_id
                    ==
                    selected_progress[1]
                ),
                (
                    ImageProcessingProgress.image_type_id
                    ==
                    selected_progress[2]
                )
            )
        ).all()
        context['selected_info'] = selected_progress
        context['pipeline_steps'] = db_session.execute(
            select(
                Step.id,
                func.replace(Step.name, '_', ' '),
                ImageProcessingProgress.id
            ).join(
                ImageProcessingProgress
            ).group_by(
                Step.id,
                Step.name,
            )
        ).all()
        context['image_types'] = db_session.execute(
            select(
                ProcessingSequence.image_type_id,
                ImageType.name,
                ImageProcessingProgress.id
            ).select_from(
                ProcessingSequence
            ).join(
                ImageType
            ).join(
                ImageProcessingProgress,
                and_(
                    (
                        ImageProcessingProgress.step_id
                        ==
                        ProcessingSequence.step_id
                    ),
                    (
                        ImageProcessingProgress.image_type_id
                        ==
                        ProcessingSequence.image_type_id
                    )
                )
            ).where(
                ProcessingSequence.step_id == selected_progress[1]
            ).group_by(
                ProcessingSequence.image_type_id,
                ImageType.name
            )
        ).all()

    return render(request, 'processing/review.html', context)


def review_single(request,
                  selected_processing_id,
                  what,
                  sub_process=0,
                  min_log_level=None):
    """A view that shows only one type of output from a processing step."""

    context = {'selected_processing_id': selected_processing_id,
               'what': what,
               'min_log_level': min_log_level,
               'selected_subp': sub_process}

    log_output_fnames = ProcessingManager(dummy=True).find_processing_outputs(
        selected_processing_id
    )
    context['sub_processes'] = range(1, len(log_output_fnames[1][0]) + 1)
    assert len(log_output_fnames[1][0]) == len(log_output_fnames[1][1])

    if sub_process == 0:
        log_output_fnames = log_output_fnames[0]
    else:
        log_output_fnames = tuple(flist[sub_process - 1]
                                  for flist in log_output_fnames[1])

    if what == 'out':
        context['reviewing'] = 'standard output/error'
        if 'out' in what:
            with open(log_output_fnames[1], 'r', encoding='utf8') as outfile:
                context['messages'] = [['debug', outfile.read()]]

    if what == 'log':
        min_log_level = getattr(logging, min_log_level.upper())
        context['reviewing'] = 'log'
        context['messages'] = []
        log_msg_start_rex = re.compile('(DEBUG|INFO|WARNING|ERROR|CRITICAL) ')
        with open(log_output_fnames[0], 'r', encoding='utf-8') as log_f:
            skip=True
            for line in log_f:
                if log_msg_start_rex.match(line):
                    level, message = line.split(maxsplit=1)
                    skip = getattr(logging, level.upper()) < min_log_level
                    if not skip:
                        context['messages'].append([level, message])
                else:
                    if not skip:
                        context['messages'][-1][1] += line

    return render(request, 'processing/review_single.html', context)
