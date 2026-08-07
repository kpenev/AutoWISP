"""Implement the view for selecting single photometric reference."""

from io import StringIO
from os import path
import logging

# from PIL.ImageTransform import AffineTransform
from django.shortcuts import render, redirect
import numpy
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select
import pandas
from astropy.coordinates import SkyCoord
from astropy import units as astropy_units

from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.interface import start_db_session
from autowisp.database.photref_selection import (
    bind_images_to_photref,
    compute_photref_candidates,
)
from autowisp.evaluator import Evaluator

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
)

# pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits
from .display_fits_util import update_fits_display

_logger = logging.getLogger(__name__)


def get_photref_merit_info(photref_group, db_session, merit_function):
    """
    Return the diagnostics, ranking, and std for each photref_group image.

    Args:
        photref_group:    The group of images for which an independent
            reference needs to be selected: list of
            (_, image_id, channel) tuples.

        db_session:    Active SQLAlchemy session for DB queries.

        merit_function(str):    Expression to evaluate for ranking images.
            May reference any diagnostic column as ``qnt_<name>`` (rank) or
            ``std_<name>`` (standard deviation).

    Returns:
        pandas.DataFrame with one row per image/channel, columns for all
        available diagnostics by name, additionally the rank of the image
        in that diagnostic is added (qnt_<name>). Finally a 'merit' column
        contains the value of the specified merit function.
    """

    rows = []
    for _, image_id, channel in photref_group:
        row = {}
        for diag_name, diag_value in db_session.execute(
            select(DiagnosticType.name, ImageDiagnostics.value)
            .join(
                DiagnosticType,
                ImageDiagnostics.diagnostic_id == DiagnosticType.id,
            )
            .where(ImageDiagnostics.image_id == image_id)
            .where(ImageDiagnostics.channel == channel)
        ).all():
            row[diag_name] = diag_value
        rows.append(row)

    merit_info = pandas.DataFrame(rows)
    frame_quantities = list(merit_info.columns)
    for column in frame_quantities:
        merit_info["qnt_" + column] = merit_info[column].rank(pct=True)

    eval_merit = Evaluator(merit_info)
    for column in frame_quantities:
        eval_merit.symtable["std_" + column] = merit_info[column].std()
    merit_info["merit"] = eval_merit(merit_function)

    return merit_info


def _get_missing_photref(request):
    """Add all frame sets missing photometric reference to the session."""

    assert "need_photref" not in request.session
    processing = ImageProcessingManager(pipeline_run_id=None)
    with start_db_session() as db_session:
        result = compute_photref_candidates(processing, db_session)

    request.session["merit_function"] = (
        "1.0 / ((1.0 - qnt_s_center)**2 + qnt_bg_center**2)"
    )
    request.session["demo"] = result["demo"]
    # Preserve the original "last entry wins" behavior: the outer loop in
    # the previous implementation overwrote ``request.session["need_photref"]``
    # on every iteration, so only the final ``(step_id, image_type_id)``
    # entry's data survived.
    if result["candidates"]:
        last = result["candidates"][-1]
        request.session["need_photref"] = {
            "master_expressions": last["master_expressions"],
            "master_values": last["groups"],
        }
    request.session.modified = True


def _get_merit_data(request, target_index):
    """Add to the session the merit information for selecting single ref."""

    if "merit_info" not in request.session:
        request.session["merit_info"] = {}
    if str(target_index) not in request.session["merit_info"]:
        _logger.debug("Calculating merit for target %s", target_index)
        batch = request.session["need_photref"]["master_values"][target_index][
            2
        ]
        photref_group = [(entry[1], entry[2], entry[3]) for entry in batch]
        with start_db_session() as db_session:
            request.session["merit_info"][str(target_index)] = (
                get_photref_merit_info(
                    photref_group,
                    db_session,
                    request.session["merit_function"],
                )
                .sort_values(by="merit", ascending=False)
                .to_json()
            )
    request.session.modified = True


def create_svg(fig):
    """Save *fig* to an SVG string, close the figure, and return the string."""

    with StringIO() as buf:
        fig.savefig(buf, format="svg")
        svg = buf.getvalue()
    pyplot.close(fig)
    return svg


def _create_pointing_plots(  # pylint: disable=too-many-locals
    merit_data,
    image_index,
    max_photref_separation=0.2,
    zoom_threshold=3,
    **plot_cfg,
):
    """
    Create SVG plots for pointing: RA vs Dec scatter and separation histogram.

    Images within max_photref_separation * diagonal_fov of the current image
    are drawn with in_range_cfg, those outside with out_of_range_cfg, and the
    current image itself with this_img_cfg.  RA axis is inverted per
    astronomical convention.  The separation histogram includes a vertical line
    at the threshold.

    If the maximum separation among all images exceeds
    zoom_threshold * threshold_deg, an additional zoomed scatter plot is
    appended that restricts the view to images within that radius so the
    in-range clustering remains visible despite the wider spread.  Images in
    the zoomed plot are still coloured by the original threshold_deg.

    Returns:
        List of SVG strings, or empty list if the required diagnostics
        (ra_center, dec_center, diagonal_fov) are absent.
    """

    if not all(
        col in merit_data.columns
        for col in ["ra_center", "dec_center", "diagonal_fov"]
    ):
        return []

    plot_cfg.setdefault("in_range_cfg", {"c": "green", "s": 20, "zorder": 3})
    plot_cfg.setdefault("out_of_range_cfg", {"c": "red", "s": 20, "zorder": 2})
    plot_cfg.setdefault("this_img_cfg", {"c": "white", "s": 100, "zorder": 4})

    ra_vals = merit_data["ra_center"].values
    dec_vals = merit_data["dec_center"].values

    threshold_deg = (
        max_photref_separation * merit_data["diagonal_fov"].iloc[image_index]
    )
    separations = (
        SkyCoord(
            ra=ra_vals[image_index] * astropy_units.deg,
            dec=dec_vals[image_index] * astropy_units.deg,
            frame="icrs",
        )
        .separation(
            SkyCoord(
                ra=ra_vals * astropy_units.deg,
                dec=dec_vals * astropy_units.deg,
                frame="icrs",
            )
        )
        .to_value(astropy_units.deg)
    )

    masks = {"this_img": numpy.arange(len(ra_vals)) == image_index}
    masks["in_range"] = (separations <= threshold_deg) & ~masks["this_img"]
    masks["out_of_range"] = ~masks["in_range"] & ~masks["this_img"]

    def plot_scatter_pointing(ax, extra_mask=None):
        for cfg_key in ["out_of_range", "in_range", "this_img"]:
            plot_mask = (
                masks[cfg_key] & extra_mask
                if extra_mask is not None
                else masks[cfg_key]
            )
            if plot_mask.any():
                ax.scatter(
                    ra_vals[plot_mask],
                    dec_vals[plot_mask],
                    **plot_cfg[cfg_key + "_cfg"],
                )
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")

    result = []

    zoom_radius = zoom_threshold * threshold_deg
    for extra_mask in [None] + (
        [separations <= zoom_radius] if separations.max() > zoom_radius else []
    ):
        fig, ax = pyplot.subplots()
        plot_scatter_pointing(ax, extra_mask)
        fig.suptitle(
            "Pointing (RA vs Dec)"
            + (" (zoomed)" if extra_mask is not None else ""),
            fontsize=32,
        )
        result.append(create_svg(fig))

    fig, ax = pyplot.subplots()
    ax.hist(separations, bins="auto", linewidth=0, color="white")
    xmin, xmax = ax.get_xlim()
    if xmin <= threshold_deg <= xmax:
        ax.axvline(x=threshold_deg, linewidth=2, color="lime", linestyle="--")
    ax.set_xlabel("Separation (deg)")
    fig.suptitle("Separation from current image", fontsize=32)
    result.append(create_svg(fig))

    return result


def _create_merit_histograms(
    merit_data, image_index, max_photref_separation=0.2
):
    """Create SVG histograms of various merit metrics showing image in each."""

    matplotlib.use("svg")
    pyplot.style.use("dark_background")
    result = []

    result.extend(
        _create_pointing_plots(merit_data, image_index, max_photref_separation)
    )

    for column in merit_data.columns:
        if column.startswith("qnt_"):
            continue
        fig, ax = pyplot.subplots()
        ax.hist(merit_data[column], bins="auto", linewidth=0, color="white")
        ax.axvline(
            x=merit_data[column].iloc[image_index], linewidth=5, color="red"
        )
        if column == "merit":
            fig.suptitle("merit", fontsize=32)
        else:
            quantile = merit_data["qnt_" + column].iloc[image_index]
            fig.suptitle(column + f" ({quantile:.3f} quantile)", fontsize=32)
        result.append(create_svg(fig))
    return result


def select_photref_image(request, *, target_index, recalculate=False):
    """Display the interface for reviewing canditate reference frames."""

    assert request.method == "GET"
    if "need_photref" not in request.session:
        return redirect("processing:select_photref_target")
    _logger.debug("Image view with request: %s", repr(request))
    update_fits_display(request)
    image_index = request.session["fits_display"]["image_index"]
    if recalculate:
        _logger.debug("Deleting merit info")
        # Recalculating before any merit info was computed is legitimate;
        # ``del`` would raise KeyError on that session.
        request.session.pop("merit_info", None)
    _get_merit_data(request, target_index)
    _logger.debug(
        "Merit info keys: %s", repr(request.session["merit_info"].keys())
    )

    merit_data = pandas.read_json(
        StringIO(request.session["merit_info"][str(target_index)])
    )
    batch = request.session["need_photref"]["master_values"][target_index][2]
    fits_fname = batch[
        # False positive
        # pylint:disable=no-member
        merit_data.index[image_index]
        # pylint:enable=no-member
    ][0]

    max_photref_separation = 0.2
    try:
        processing_mgr = ImageProcessingManager(pipeline_run_id=None)
        with start_db_session() as db_session:
            first_image = db_session.get(Image, batch[0][2])
            processing_mgr.evaluate_expressions_image(first_image, db_session)
            fit_config = processing_mgr.get_config(
                matched_expressions=None,
                db_session=db_session,
                image_id=batch[0][2],
                channel=batch[0][3],
                step_name="fit_magnitudes",
            )[0]
            max_photref_separation = fit_config.get(
                "max_photref_separation", 0.2
            )
    except Exception:  # pylint: disable=broad-except
        pass

    context = {
        "target_index": target_index,
        # False positive
        # pylint: disable=no-member
        "num_images": merit_data.shape[0],
        # pylint: enable=no-member
        "histograms": _create_merit_histograms(
            merit_data, image_index, max_photref_separation
        ),
        "fits_fname": path.basename(fits_fname),
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
    return render(request, "processing/select_photref_image.html", context)


def select_photref_target(request, recalc=False):
    """Display view to select which of the missing photrefs to define."""

    if recalc:
        merit_function = (
            request.POST.get("merit-function")
            if request.method == "POST"
            else None
        )
        request.session.flush()
        if merit_function is not None:
            request.session["merit_function"] = merit_function
        return redirect("/processing/select_photref_target")
    if "need_photref" not in request.session:
        _get_missing_photref(request)

    _logger.debug(
        "Request master values: %s",
        repr(request.session["need_photref"]["master_values"]),
    )
    return render(
        request,
        "processing/select_photref_target.html",
        {
            "master_expressions": request.session["need_photref"][
                "master_expressions"
            ]
            + ["Num. Images"],
            "master_values": [
                target[0] + [len(target[2])]
                for target in request.session["need_photref"]["master_values"]
            ],
            "merit_function": request.session["merit_function"],
            "view_config": request.body,
        },
    )


def record_photref_selection(request, target_index, image_index):
    """Record a single photometric reference frame selected by the user."""

    # The selection is recorded by following a plain link, so the browser can
    # re-issue this GET (refresh, back button, double click). The keys popped
    # at the end are gone by then, hence both the guard and re-deriving the
    # merit info below instead of assuming the session still holds it.
    if "need_photref" not in request.session:
        return redirect("processing:select_photref_target")
    if request.session["demo"]:
        _logger.info("Demo only! Not saving selected reference!")
        return redirect("processing:select_photref_target")
    _get_merit_data(request, target_index)
    _logger.debug(
        "Merit info keys: %s", repr(request.session["merit_info"].keys())
    )
    merit_data = pandas.read_json(
        StringIO(request.session["merit_info"][str(target_index)])
    )
    batch = request.session["need_photref"]["master_values"][target_index][2]
    dr_fname = batch[
        # False positive
        # pylint:disable=no-member
        merit_data.index[image_index]
        # pylint:enable=no-member
    ][1]

    ImageProcessingManager(pipeline_run_id=None).add_masters(
        {
            "type": "single_photref",
            "filename": dr_fname,
            "preference_order": None,
            "disable": False,
        }
    )
    bind_images_to_photref(dr_fname, batch)

    # Force full re-derivation of the photref selection list on next page load
    request.session.pop("need_photref", None)
    request.session.pop("merit_info", None)
    request.session.modified = True

    return redirect("/processing/select_photref_target")
