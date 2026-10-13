"""Define the view displaying the current processing progress."""

import logging

from sqlalchemy import select
from django.shortcuts import render

from autowisp.database.interface import start_db_session
from autowisp.database.pipeline_runs import find_active_runs
from autowisp.database.user_interface import (
    get_processing_sequence,
    get_progress,
    list_channels,
)
from autowisp.error_render import (
    error_counts_by_step,
    open_error_count_for_steps,
)

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import ImageProcessingProgress

# pylint: enable=no-name-in-module

from autowisp.browser_interface.diagnostics.quantities import (
    get_quantile_names,
)
from .log_views import datetime_fmt

logger = logging.getLogger(__name__)


def progress(request, await_start=-1):  # pylint: disable=too-many-locals
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

        error_by_step = error_counts_by_step(db_session)
        # Drives the colour of the Start Processing button: open pipeline +
        # step errors (BUI errors excluded) mean "errors pending".
        context["gate_error_count"] = open_error_count_for_steps([], db_session)
        context["progress"] = [
            [
                step.name.split("_"),
                imtype.name,
                [[0, 0, 0, []] for _ in context["channels"]],
                [],
                error_by_step.get(step.name, 0),
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

        # Only this machine's runs: the BUI shows what it can start or wait
        # for here.
        running_here = find_active_runs(db_session)[0]
        if running_here:
            logger.info(
                "Pipeline processes still running: %s.",
                ", ".join(str(run.process_id) for run in running_here),
            )
            context["running"] = True
            context["refresh_seconds"] = 5
            context["await_start"] = -1

    # Get selected steps from session if processing
    selected_tokens = set()
    if context["running"] and "selected_step_tokens" in request.session:
        selected_tokens = set(request.session["selected_step_tokens"])

    context["selected_tokens"] = selected_tokens

    # The calibration bar links to every quantile recorded, each its own
    # section of one plot, there being no longer a single name standing
    # for the family. Empty until one is recorded, since the URL converter
    # cannot build a section list out of nothing -- which is the same as
    # the other bars, whose links lead nowhere until there is something to
    # show.
    with start_db_session() as db_session:
        context["quantile_quantities"] = ",".join(
            get_quantile_names(db_session)
        )

    return render(request, "processing/progress.html", context)
