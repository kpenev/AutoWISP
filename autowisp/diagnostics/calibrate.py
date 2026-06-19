"""Functions for diagnostics of the calibrate step."""

import os
import logging
import numpy

_logger = logging.getLogger(__name__)
MEDIAN_RB_RATIO_DIAGNOSTIC = "median_rb_ratio"
LOCAL_SKY_GB_RATIO_DIAGNOSTIC = "local_sky_gb_ratio"
LOCAL_SKY_BRIGHTNESS_MINMAX_DIAGNOSTIC = (
    "local_sky_brightness_minmax_frac"
)


def _get_image_percentiles_jd(cal_fname, percentiles):
    """Return the pixel value percentiles and JD for the given image."""

    from autowisp.fits_utilities import read_image_components

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


def _block_sky_medians(channel_pixels, channel_masks, grid_size, bright_clip):
    """Return star-suppressed block medians for available color channels."""

    required = ("R", "G", "B")
    grouped = {key: [] for key in required}
    grouped_masks = {key: [] for key in required}
    channels = {}
    masks = {}
    for channel_name, pixels in channel_pixels.items():
        if not channel_name:
            continue
        channel_key = channel_name[0].upper()
        if channel_key in grouped:
            grouped[channel_key].append(numpy.asarray(pixels, dtype=float))
            grouped_masks[channel_key].append(
                numpy.asarray(channel_masks[channel_name])
            )
    for channel_key in required:
        if not grouped[channel_key]:
            return None
        channels[channel_key] = numpy.mean(grouped[channel_key], axis=0)
        masks[channel_key] = numpy.max(grouped_masks[channel_key], axis=0)

    num_y = min(channel.shape[0] for channel in channels.values())
    num_x = min(channel.shape[1] for channel in channels.values())
    medians = []
    for y_index in range(grid_size):
        y_start = y_index * num_y // grid_size
        y_end = (y_index + 1) * num_y // grid_size
        for x_index in range(grid_size):
            x_start = x_index * num_x // grid_size
            x_end = (x_index + 1) * num_x // grid_size
            block_values = {}
            good = None
            for channel_key in required:
                values = channels[channel_key][
                    y_start:y_end, x_start:x_end
                ].ravel()
                mask = masks[channel_key][
                    y_start:y_end, x_start:x_end
                ].ravel()
                finite = numpy.isfinite(values) & (values > 0) & (mask == 0)
                good = finite if good is None else good & finite
                block_values[channel_key] = values
            if good is None or good.sum() < 100:
                continue

            sky = good.copy()
            for channel_key in required:
                values = block_values[channel_key][good]
                limit = numpy.quantile(values, bright_clip)
                sky &= block_values[channel_key] <= limit
            if sky.sum() < 100:
                continue

            medians.append(
                {
                    channel_key: float(
                        numpy.median(block_values[channel_key][sky])
                    )
                    for channel_key in required
                }
            )

    return medians


def get_local_sky_diagnostics(
    channel_pixels,
    channel_masks,
    *,
    grid_size=8,
    bright_clip=0.8,
):
    """Return local, star-suppressed sky diagnostics."""

    block_medians = _block_sky_medians(
        channel_pixels, channel_masks, grid_size, bright_clip
    )
    if not block_medians:
        return {}

    rb_ratios = []
    gb_ratios = []
    brightnesses = []
    for medians in block_medians:
        blue = medians["B"]
        if blue <= 0:
            continue
        rb_ratios.append(medians["R"] / blue)
        gb_ratios.append(medians["G"] / blue)
        brightnesses.append(
            (medians["R"] + medians["G"] + medians["B"]) / 3.0
        )
    if not rb_ratios or not gb_ratios or not brightnesses:
        return {}

    red_channel = next(
        (
            channel_name
            for channel_name in channel_pixels
            if channel_name and channel_name[0].upper() == "R"
        ),
        None,
    )
    if red_channel is None:
        return {}

    diagnostics = [
        (MEDIAN_RB_RATIO_DIAGNOSTIC, float(numpy.median(rb_ratios))),
        (LOCAL_SKY_GB_RATIO_DIAGNOSTIC, float(numpy.median(gb_ratios))),
    ]
    median_brightness = float(numpy.median(brightnesses))
    if median_brightness > 0:
        diagnostics.append(
            (
                LOCAL_SKY_BRIGHTNESS_MINMAX_DIAGNOSTIC,
                (
                    float(numpy.max(brightnesses))
                    - float(numpy.min(brightnesses))
                )
                / median_brightness,
            )
        )

    return {red_channel: diagnostics}


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

    from sqlalchemy import select

    from autowisp.database.interface import start_db_session
    from autowisp.database.image_processing import ImageProcessingManager

    # False positive
    # pylint: disable=no-name-in-module,import-outside-toplevel
    from autowisp.database.data_model import (
        Step,
        ImageType,
        ObservingSession,
        ProcessingSequence,
    )

    # pylint: enable=no-name-in-module,import-outside-toplevel

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
