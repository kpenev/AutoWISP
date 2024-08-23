"""The views related to reviewing logs."""

import re
import logging

from django.shortcuts import render
from sqlalchemy import select, func, and_

from autowisp.database.interface import Session
from autowisp.database.image_processing import ImageProcessingManager
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    ImageProcessingProgress,\
    Step,\
    ImageType,\
    ProcessingSequence
#pylint: enable=no-name-in-module

datetime_fmt = '%Y%m%d %H:%M:%S'


def review(request,
           selected_processing_id,
           min_log_level='WARNING'):
    """
    A view for going through pipeline logs and diagnostics.

    Args:
        selected_processing_id(int):    The progress ID for which to display
            logs and/or diagnostics.

        min_log_level(str):    Only log messages of this level and higher are
            displayed.
    """

    context = {'selected_processing_id': selected_processing_id,
               'min_log_level': min_log_level}
    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        selected_progress = db_session.scalar(
            select(
                ImageProcessingProgress
            ).where(
                ImageProcessingProgress.id == selected_processing_id,
            )
        )
        selected_progress = (
            selected_progress.id,
            selected_progress.step_id,
            selected_progress.image_type_id,
            selected_progress.started.strftime(datetime_fmt),
            (
                '-' if selected_progress.finished is None
                else selected_progress.finished.strftime(datetime_fmt)
            )
        )

        context['reviewable'] = [
            (
                record[0],
                record[1].strftime(datetime_fmt),
                '-' if record[2] is None else record[2].strftime(datetime_fmt)
            )
            for record in db_session.execute(
                select(
                    ImageProcessingProgress.id,
                    ImageProcessingProgress.started,
                    ImageProcessingProgress.finished,
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
        ]
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

    log_output_fnames = ImageProcessingManager(
        dummy=True
    ).find_processing_outputs(
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
