"""Sky-background diagnostics shared by calibration and browser views."""

import numpy


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

            # Keep the dimmer pixels in each block so stars do not dominate
            # the local sky estimate.
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

    # Attach cross-channel diagnostics to a real processed channel so the
    # existing image-diagnostics save path persists them.
    diagnostics = [
        ("median_rb_ratio", float(numpy.median(rb_ratios))),
        ("local_sky_gb_ratio", float(numpy.median(gb_ratios))),
    ]
    median_brightness = float(numpy.median(brightnesses))
    if median_brightness > 0:
        diagnostics.append(
            (
                "local_sky_brightness_minmax_frac",
                (
                    float(numpy.max(brightnesses))
                    - float(numpy.min(brightnesses))
                )
                / median_brightness,
            )
        )

    return {red_channel: diagnostics}
