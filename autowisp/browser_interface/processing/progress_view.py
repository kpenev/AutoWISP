"""Define the view displaying the current processing progress."""

import logging
from socket import getfqdn
import os
from datetime import datetime

from sqlalchemy import select, sql
from psutil import pid_exists
from django.shortcuts import render

from autowisp.database.interface import start_db_session
from autowisp.database.user_interface import (
    get_processing_sequence,
    get_progress,
    list_channels,
)

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import ImageProcessingProgress, PipelineRun

# pylint: enable=no-name-in-module

from .log_views import datetime_fmt

logger = logging.getLogger(__name__)


def progress(request, await_start=-1):
    """Display the current processing progress."""

    print(f"Generating progress page with await start: {await_start}")
    context = {"await_start": await_start + 1}
    if 0 <= await_start < 10:
        context = {
            "await_start": await_start + 1,
            "running": True,
            "refresh_seconds": 6,
        }
    else:
        context = {
            "await_start": -1,
            "running": False,
            "refresh_seconds": 0,
        }
    with start_db_session() as db_session:
        context["channels"] = sorted(list_channels(db_session))
        channel_index = {
            channel: i for i, channel in enumerate(context["channels"])
        }
        processing_sequence = get_processing_sequence(db_session)

        context["progress"] = [
            [
                step.name.split("_"),
                imtype.name,
                [[0, 0, 0, []] for _ in context["channels"]],
                [],
            ]
            for step, imtype in processing_sequence
        ]
        for (step, imtype), destination in zip(
            processing_sequence, context["progress"]
        ):
            final, pending, by_status = get_progress(
                step, imtype.id, 0, db_session
            )
            for channel, status, count in final:
                destination[2][channel_index[channel]][
                    0 if status > 0 else 1
                ] = (count or 0)

            for channel, count in pending:
                destination[2][channel_index[channel]][2] = count or 0

            for channel, status, count in by_status:
                destination[2][channel_index[channel]][3].append(
                    (status, (count or 0))
                )
            destination[3] = [
                (
                    record[0],
                    record[1].strftime(datetime_fmt) if record[1] else "-",
                    record[2].strftime(datetime_fmt) if record[2] else "-",
                )
                for record in db_session.execute(
                    select(
                        ImageProcessingProgress.id,
                        ImageProcessingProgress.started,
                        ImageProcessingProgress.finished,
                    ).where(
                        ImageProcessingProgress.step_id == step.id,
                        ImageProcessingProgress.image_type_id == imtype.id,
                    )
                ).all()
            ]

        for check_running in db_session.scalars(
            select(PipelineRun).filter_by(finished=None, host=getfqdn())
        ).all():
            elapsed_time = datetime.now() - check_running.started
            if (
                pid_exists(check_running.process_id)
                and check_running.process_id != os.getpid()
                or (elapsed_time.days < 0 and elapsed_time.seconds <= 60)
            ):
                logger.info(
                    "Calibration process with ID %s still exists.",
                    check_running.process_id,
                )
                context["running"] = True
                context["refresh_seconds"] = 5
                context["await_start"] = -1
            else:
                logger.info("Marking %s as finished", check_running)
                check_running.finished = (
                    sql.func.now()  # pylint: disable=not-callable
                )

    return render(request, "processing/progress.html", context)
