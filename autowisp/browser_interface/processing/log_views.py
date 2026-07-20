"""The views related to reviewing logs."""

import re
import logging

from django.shortcuts import render
from sqlalchemy import select, func, and_

from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    ImageProcessingProgress,
    LightCurveProcessingProgress,
    Step,
    ImageType,
    ProcessingSequence,
)

# pylint: enable=no-name-in-module

datetime_fmt = "%Y%m%d %H:%M:%S"


def _get_progress_type(request):
    """Return the requested progress model and its URL identifier."""

    if request.GET.get("processing_type") == "lightcurve":
        return LightCurveProcessingProgress, "lightcurve"
    return ImageProcessingProgress, "image"


def review(request, selected_processing_id, min_log_level="WARNING"):
    """
    A view for going through pipeline logs and diagnostics.

    Args:
        selected_processing_id(int):    The progress ID for which to display
            logs and/or diagnostics.

        min_log_level(str):    Only log messages of this level and higher are
            displayed.
    """

    progress_class, processing_type = _get_progress_type(request)
    context = {
        "selected_processing_id": selected_processing_id,
        "min_log_level": min_log_level,
        "processing_type": processing_type,
    }
    with start_db_session() as db_session:
        selected_progress = db_session.scalar(
            select(progress_class).where(
                progress_class.id == selected_processing_id,
            )
        )
        target_column = (
            LightCurveProcessingProgress.single_photref_id
            if progress_class is LightCurveProcessingProgress
            else ImageProcessingProgress.image_type_id
        )
        target_id = getattr(selected_progress, target_column.key)
        image_type_id = (
            db_session.scalar(
                select(ProcessingSequence.image_type_id).where(
                    ProcessingSequence.step_id == selected_progress.step_id
                )
            )
            if progress_class is LightCurveProcessingProgress
            else target_id
        )
        selected_progress = (
            selected_progress.id,
            selected_progress.step_id,
            image_type_id,
            selected_progress.started.strftime(datetime_fmt),
            (
                "-"
                if selected_progress.finished is None
                else selected_progress.finished.strftime(datetime_fmt)
            ),
        )

        context["reviewable"] = [
            (
                record[0],
                record[1].strftime(datetime_fmt),
                "-" if record[2] is None else record[2].strftime(datetime_fmt),
            )
            for record in db_session.execute(
                select(
                    progress_class.id,
                    progress_class.started,
                    progress_class.finished,
                ).where(
                    progress_class.step_id == selected_progress[1],
                    target_column == target_id,
                )
            ).all()
        ]
        context["selected_info"] = selected_progress
        image_steps = db_session.execute(
            select(
                Step.id,
                func.replace(Step.name, "_", " "),
                ImageProcessingProgress.id,
            )
            .join(ImageProcessingProgress)
            .group_by(
                Step.id,
                Step.name,
            )
        ).all()
        lightcurve_steps = db_session.execute(
            select(
                Step.id,
                func.replace(Step.name, "_", " "),
                func.max(LightCurveProcessingProgress.id),
            )
            .join(LightCurveProcessingProgress)
            .group_by(Step.id, Step.name)
        ).all()
        context["pipeline_steps"] = sorted(
            [(*step, "image") for step in image_steps]
            + [(*step, "lightcurve") for step in lightcurve_steps]
        )

        if progress_class is LightCurveProcessingProgress:
            image_type = db_session.execute(
                select(ImageType.id, ImageType.name)
                .select_from(ProcessingSequence)
                .join(ImageType)
                .where(ProcessingSequence.step_id == selected_progress[1])
            ).one()
            context["image_types"] = [
                (*image_type, selected_processing_id, processing_type)
            ]
        else:
            context["image_types"] = [
                (*image_type, processing_type)
                for image_type in db_session.execute(
                    select(
                        ProcessingSequence.image_type_id,
                        ImageType.name,
                        ImageProcessingProgress.id,
                    )
                    .select_from(ProcessingSequence)
                    .join(ImageType)
                    .join(
                        ImageProcessingProgress,
                        and_(
                            ImageProcessingProgress.step_id
                            == ProcessingSequence.step_id,
                            ImageProcessingProgress.image_type_id
                            == ProcessingSequence.image_type_id,
                        ),
                    )
                    .where(
                        ProcessingSequence.step_id == selected_progress[1]
                    )
                    .group_by(
                        ProcessingSequence.image_type_id,
                        ImageType.name,
                    )
                ).all()
            ]

    return render(request, "processing/review.html", context)


def review_single(
    request, selected_processing_id, what, sub_process=0, min_log_level=None
):
    """A view that shows only one type of output from a processing step."""

    progress_class, processing_type = _get_progress_type(request)
    context = {
        "selected_processing_id": selected_processing_id,
        "what": what,
        "min_log_level": min_log_level,
        "selected_subp": sub_process,
        "processing_type": processing_type,
    }

    with start_db_session() as db_session:
        processing_progress = db_session.scalar(
            select(progress_class).where(
                progress_class.id == selected_processing_id
            )
        )
        log_output_fnames = ImageProcessingManager(
            pipeline_run_id=None
        ).find_processing_outputs(processing_progress, db_session)
    context["sub_processes"] = range(1, len(log_output_fnames[1][0]) + 1)
    assert len(log_output_fnames[1][0]) == len(log_output_fnames[1][1])

    if sub_process == 0:
        log_output_fnames = log_output_fnames[0]
    else:
        log_output_fnames = tuple(
            flist[sub_process - 1] for flist in log_output_fnames[1]
        )

    if what == "out":
        context["reviewing"] = "standard output/error"
        if "out" in what:
            with open(log_output_fnames[1], "r", encoding="utf8") as outfile:
                context["messages"] = [["debug", outfile.read()]]

    if what == "log":
        min_log_level = getattr(logging, min_log_level.upper())
        context["reviewing"] = "log"
        context["messages"] = []
        log_msg_start_rex = re.compile("(DEBUG|INFO|WARNING|ERROR|CRITICAL) ")
        with open(log_output_fnames[0], "r", encoding="utf-8") as log_f:
            skip = True
            for line in log_f:
                if log_msg_start_rex.match(line):
                    level, message = line.split(maxsplit=1)
                    skip = getattr(logging, level.upper()) < min_log_level
                    if not skip:
                        context["messages"].append([level, message])
                else:
                    if not skip:
                        context["messages"][-1][1] += line

    return render(request, "processing/review_single.html", context)
