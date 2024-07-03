"""Define the view displaying the current processing progress."""

from socket import getfqdn
from datetime import datetime

from sqlalchemy import select, func
from psutil import pid_exists
from django.shortcuts import render

from autowisp.database.interface import Session
from autowisp.database.user_interface import\
    get_processing_sequence,\
    get_progress,\
    list_channels
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import ImageProcessingProgress
#pylint: enable=no-name-in-module

from .log_views import datetime_fmt


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
            destination[3] = [
                (
                    record[0],
                    record[1].strftime(datetime_fmt) if record[1] else '-',
                    record[2].strftime(datetime_fmt) if record[2] else '-'
                )
                for record in db_session.execute(
                    select(
                        ImageProcessingProgress.id,
                        ImageProcessingProgress.started,
                        ImageProcessingProgress.finished,
                    ).where(
                        ImageProcessingProgress.step_id == step.id,
                        ImageProcessingProgress.image_type_id == imtype.id
                    )
                ).all()
            ]

        for check_running in db_session.scalars(
            select(
                ImageProcessingProgress
            ).where(
                #pylint: disable=singleton-comparison
                ImageProcessingProgress.finished == None,
                #pylint: enable=singleton-comparison
                ImageProcessingProgress.host == getfqdn()
            )
        ):
            if pid_exists(check_running.process_id):
                context['running'] = True
                context['refresh_seconds'] = 5
            else:
                print('Marking {check_running} as finished')
                check_running.finished = datetime.now()

    return render(request, 'processing/progress.html', context)
