"""Functions for diagnostics of the calibrate step."""

import os
import logging
import numpy
from sqlalchemy import select

from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.fits_utilities import read_image_components

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Step,
    ImageType,
    ObservingSession,
    ProcessingSequence,
)

# pylint: enable=no-name-in-module

_logger = logging.getLogger(__name__)


def _get_image_percentiles_jd(cal_fname, percentiles):
    """Return the pixel value percentiles and JD for the given image."""

    (  # pylint: disable=unbalanced-tuple-unpacking
        pixel_data,
        mask,
        cal_header,
    ) = read_image_components(cal_fname, read_error=False)

    jd = cal_header.get("JD-OBS", cal_header.get("JD_OBS"))
    if jd is None:
        _logger.warning("No JD-OBS in %s, skipping", cal_fname)
        return None, None

    good = (
        pixel_data[mask == 0]  # pylint: disable=unsubscriptable-object
        if mask is not None
        else pixel_data
    )
    if good.size == 0:
        return None, None

    pct_values = numpy.percentile(good, percentiles)
    return jd, pct_values


def collect_calibrated_percentiles(  # pylint: disable=too-many-locals
    observing_session_label, percentiles=(10, 50, 90, 99.9)
):
    """
    Compute pixel-value percentiles of calibrated images vs time (JD).

    For every successfully calibrated object image in the observing session,
    the requested percentiles of the unmasked pixel values are computed
    (separately for each color channel) and returned alongside the
    mid-exposure Julian Date.

    Args:
        observing_session_label(str):    The ``label`` of the
            :class:`ObservingSession` whose calibrated images should be
            analysed.

        percentiles(iterable):    The percentiles of pixel values to
            compute.

    Returns:
        dict:
            Keyed by ``(channel_name, percentile)`` tuples, each value is
            a pair of numpy arrays ``(times, values)``.
    """

    processing = ImageProcessingManager(pipeline_run_id=None)

    collected = {}

    with start_db_session() as db_session:
        session = db_session.scalar(
            select(ObservingSession).where(
                ObservingSession.label == observing_session_label
            )
        )
        if session is None:
            raise ValueError(
                f"No observing session with label "
                f"{observing_session_label!r}"
            )

        calibrate_steps = db_session.execute(
            select(Step, ImageType)
            .select_from(ProcessingSequence)
            .join(Step, ProcessingSequence.step_id == Step.id)
            .join(
                ImageType,
                ProcessingSequence.image_type_id == ImageType.id,
            )
            .where(Step.name == "calibrate")
        ).all()

        processing.set_pending(db_session, calibrate_steps, invert=True)

        for step, imtype in calibrate_steps:
            for image, channel, _ in processing.pending.get(
                (step.id, imtype.id), []
            ):
                if image.observing_session_id != session.id:
                    continue

                processing.evaluate_expressions_image(image, db_session)
                cal_fname = processing.get_step_input(
                    image, channel, "calibrated"
                )

                if not os.path.exists(cal_fname):
                    _logger.debug("Calibrated file not found: %s", cal_fname)
                    continue

                jd, pct_values = _get_image_percentiles_jd(
                    cal_fname, percentiles
                )
                if jd is None:
                    continue
                for pctl, val in zip(percentiles, pct_values):
                    key = (channel, pctl)
                    if key not in collected:
                        collected[key] = ([], [])
                    collected[key][0].append(jd)
                    collected[key][1].append(val)

    return {
        key: (numpy.array(times), numpy.array(values))
        for key, (times, values) in collected.items()
    }


def plot_calibrated_percentiles(
    observing_session_label, axes, percentiles=(10, 50, 90, 99.9)
):
    """
    Plot pixel-value percentiles of calibrated images vs time (JD).

    Collects percentile data via :func:`collect_calibrated_percentiles`
    and draws the results on the given axes.

    Args:
        observing_session_label(str):    The ``label`` of the
            :class:`ObservingSession` whose calibrated images should be
            analysed.

        axes(matplotlib.axes.Axes):    The axes to draw the percentile
            curves on.

        percentiles(iterable):    The percentiles of pixel values to plot.

    Returns:
        dict:
            Keyed by ``(channel_name, percentile)`` tuples, each value is
            a pair of numpy arrays ``(times, values)``.
    """

    result = collect_calibrated_percentiles(
        observing_session_label, percentiles
    )
    _plot(result, axes)
    return result


def _plot(result, axes):
    """Draw the percentile curves on *axes*."""

    channel_colors = {"R": "red", "G": "green", "B": "blue"}
    fallback_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    percentile_styles = {10: ":", 50: "-", 90: "--", 99.9: "-."}

    channel_idx = {}
    for channel, percentile in sorted(
        result, key=lambda k: (k[0] is None, str(k[0]), k[1])
    ):
        if channel not in channel_idx:
            channel_idx[channel] = len(channel_idx)
        color = channel_colors.get(
            channel[0],
            fallback_colors[channel_idx[channel] % len(fallback_colors)],
        )
        ch_label = channel if channel is not None else "mono"

        times, values = result[(channel, percentile)]
        if times.size == 0:
            continue
        order = numpy.argsort(times)
        axes.plot(
            times[order],
            values[order],
            linestyle=percentile_styles[percentile],
            color=color,
            label=f"{ch_label} p{percentile:g}",
        )

    axes.set_xlabel("JD")
    axes.set_ylabel("Pixel value")
    axes.legend()


if __name__ == "__main__":
    from autowisp.database.interface import set_project_home
    from matplotlib import pyplot

    set_project_home(
        os.path.expanduser("~/tmp/test_data/test_calib_diagnostic")
    )
    plot_calibrated_percentiles("G10124500_139", pyplot.gca())
    pyplot.show()
    pyplot.cla()
    plot_calibrated_percentiles("dark", pyplot.gca())
    pyplot.show()
    pyplot.cla()
    plot_calibrated_percentiles("skyflat", pyplot.gca())
    pyplot.show()
