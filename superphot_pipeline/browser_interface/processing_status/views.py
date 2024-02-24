"""The views showing the status of the processing."""
from django.shortcuts import render

from superphot_pipeline.database.interface import Session
from superphot_pipeline.database.user_interface import\
    get_processing_sequence,\
    get_progress,\
    list_channels

def progress(request):
    """Display the current processing progress."""

    with Session.begin() as db_session:
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
    print('Formatted: ' + repr(formatted))

    return render(
        request,
        'processing_status/progress.html',
        {
            'channels': channels,
            'progress': formatted
        }
    )
# Create your views here.
