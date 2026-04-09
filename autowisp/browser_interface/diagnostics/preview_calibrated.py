"""View for previewing a calibrated FITS image by image ID and channel."""

from base64 import b64encode
from io import BytesIO

import numpy
from PIL import Image as PILImage
from sqlalchemy import select

from django.shortcuts import render
from django.http import JsonResponse
from django.urls import reverse

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import Image, ImageDiagnostics, DiagnosticType

# pylint: enable=no-name-in-module
from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.bui_util import encode_fits
from autowisp.browser_interface.processing.display_fits_util import (
    update_fits_display,
)
from autowisp.fits_utilities import read_image_components


def _get_dr_fname(image_id, color_channel):
    """Return the data-reduction file path for the given image and channel."""

    with start_db_session() as db_session:
        image = db_session.get(Image, image_id)
        processing = ImageProcessingManager(pipeline_run_id=None)
        processing.evaluate_expressions_image(image, db_session)
        return processing.get_product_fname(image_id, color_channel, "dr")


def _read_source_positions(dr_file, src_type):
    """Return x, y positions for srcextract or srcproj sources from DR file."""

    dataset_key = (
        "srcextract.sources" if src_type == "srcextract" else "srcproj.columns"
    )
    x = dr_file.get_dataset(
        dataset_key,
        **{f"{src_type}_column_name": "x", f"{src_type}_version": 0},
    )
    y = dr_file.get_dataset(
        dataset_key,
        **{f"{src_type}_column_name": "y", f"{src_type}_version": 0},
    )
    return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]


def _read_matched_positions(dr_file):
    """Return extracted positions of stars matched to catalog entries."""

    matched = dr_file.get_dataset("skytoframe.matched", skytoframe_version=0)
    extracted = _read_source_positions(dr_file, "srcextract")
    return [extracted[i] for i in matched[:, 1]]


def get_image_overlay(_, image_id, color_channel, overlay_type):
    """Return JSON star positions for the requested overlay type."""

    readers = {
        "extracted_stars": lambda dr: _read_source_positions(dr, "srcextract"),
        "projected_stars": lambda dr: _read_source_positions(dr, "srcproj"),
        "matched_stars": _read_matched_positions,
    }
    if overlay_type not in readers:
        return JsonResponse(
            {"stars": [], "message": f"Unknown overlay type: {overlay_type}"}
        )
    try:
        dr_fname = _get_dr_fname(image_id, color_channel)
        with DataReductionFile(fname=dr_fname, mode="r") as dr_file:
            stars = readers[overlay_type](dr_file)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"stars": [], "message": str(exc)})
    return JsonResponse({"stars": stars})


def _generate_quantile_overlay_png(fits_fname, threshold):
    """
    Highlight pixels in the image below threshold

    Return PNG bytes: cyan at 50% alpha below threshold, transparent above.
    """

    print(
        f"Generating quantile overlay for {fits_fname!r} with "
        f"threshold: {threshold!r}"
    )
    pixel_data = read_image_components(
        fits_fname, read_error=False, read_mask=False, read_header=False
    )[0]
    height, width = pixel_data.shape
    overlay = numpy.zeros((height, width, 4), dtype=numpy.uint8)
    below = pixel_data < threshold
    overlay[below] = [252, 141, 89, 128]
    overlay[numpy.logical_not(below)] = [145, 191, 219, 128]
    print(f"{numpy.sum(below)}/{pixel_data.size} pixels below threshold")
    png_stream = BytesIO()
    PILImage.fromarray(overlay, mode="RGBA").save(png_stream, "png")
    return png_stream.getvalue()


def _get_overlay_choices(image_id, color_channel, db_session):
    """Return the list of (value, label) overlay choices for this image."""

    fixed_choices = [
        ("extracted_stars", "Extracted Stars"),
        ("projected_stars", "Projected Stars"),
        ("matched_stars", "Matched Stars"),
        ("below_background", "Below Background"),
    ]

    quantile_names = db_session.scalars(
        select(DiagnosticType.name)
        .join(
            ImageDiagnostics,
            ImageDiagnostics.diagnostic_id == DiagnosticType.id,
        )
        .where(
            ImageDiagnostics.image_id == image_id,
            ImageDiagnostics.channel == color_channel,
            DiagnosticType.name.like("pixel_q%"),
        )
        .order_by(DiagnosticType.name)
    ).all()

    quantile_choices = [
        (
            "below_" + name,
            "Below 0." + name[len("pixel_q") :] + " quantile",
        )
        for name in quantile_names
    ]

    return fixed_choices + quantile_choices


def preview_calibrated_image(request, image_id, color_channel):
    """Display the calibrated image for the given image and color channel."""

    update_fits_display(request)
    overlay = request.session["fits_display"].get("overlay")

    with start_db_session() as db_session:
        image = db_session.get(Image, image_id)
        processing = ImageProcessingManager(pipeline_run_id=None)
        processing.evaluate_expressions_image(image, db_session)
        fits_fname = processing.get_product_fname(
            image_id, color_channel, "calibrated"
        )
        overlay_choices = _get_overlay_choices(
            image_id, color_channel, db_session
        )
        overlay_diagnostic = None
        if overlay == "below_background":
            overlay_diagnostic = "bg_center"
        elif overlay and overlay.startswith("below_pixel_q"):
            overlay_diagnostic = overlay[len("below_") :]
        overlay_threshold = None
        if overlay_diagnostic is not None:
            overlay_threshold = db_session.scalar(
                select(ImageDiagnostics.value)
                .join(
                    DiagnosticType,
                    DiagnosticType.id == ImageDiagnostics.diagnostic_id,
                )
                .where(
                    ImageDiagnostics.image_id == image_id,
                    ImageDiagnostics.channel == color_channel,
                    DiagnosticType.name == overlay_diagnostic,
                )
            )

    overlay_url = None
    overlay_image = None
    if overlay in ("extracted_stars", "projected_stars", "matched_stars"):
        overlay_url = reverse(
            "diagnostics:get_image_overlay",
            kwargs={
                "image_id": image_id,
                "color_channel": color_channel,
                "overlay_type": overlay,
            },
        )
    elif overlay_threshold is not None:
        overlay_image = b64encode(
            _generate_quantile_overlay_png(fits_fname, overlay_threshold)
        ).decode("utf-8")

    context = {
        "image_id": image_id,
        "color_channel": color_channel,
        "overlay_choices": overlay_choices,
        "overlay_url": overlay_url,
        "overlay_image": overlay_image,
        "view_config": request.session.get("view_config", "undefined"),
    }
    context.update(request.session["fits_display"])
    context.update(
        encode_fits(
            fits_fname,
            request.session["fits_display"]["range"],
            request.session["fits_display"]["transform"],
        )
    )
    return render(request, "diagnostics/preview_calibrated_image.html", context)
