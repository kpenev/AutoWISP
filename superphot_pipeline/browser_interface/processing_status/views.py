"""The views showing the status of the processing."""
from django.shortcuts import render

from superphot_pipeline.database.interface import Session
from superphot_pipeline.database.user_interface import\
    get_processing_sequence,\
    get_progress

def progress(request):
    """Display the current processing progress."""

    with Session.begin() as db_session:
        processing_sequence = get_processing_sequence(db_session)
        progress_data = [
            (step.name, imtype.name, [])
            for step, imtype in processing_sequence
        ]
        for i, (step, image_type) in enumerate(processing_sequence):
            progress = get_progress(step.id, image_type.id, 0, db_session)
            for channel, final, pending, by_status in progress:
                by_status.remove(final)
                progress_data[i][2].append(
                    (channel, final[1], pending, by_status)
                )

    return render(request,
                  'processing_status/progress.html',
                  {'progress_data': progress_data})

# Create your views here.
