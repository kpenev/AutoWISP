"""Implement the view for selecting single photometric reference."""

from io import StringIO
from os import path
import logging

# from PIL.ImageTransform import AffineTransform
from django.shortcuts import render, redirect
import numpy
import matplotlib
from matplotlib import pyplot
import pandas
from astropy.coordinates import SkyCoord
from astropy import units as astropy_units

from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.interface import start_db_session
from autowisp.database.photref_selection import (
    compute_photref_candidates,
    get_group_exclusions,
    get_merit_expressions,
    get_offered_candidates,
    get_photref_exclusions,
    rank_photref_candidates,
    record_single_photref,
)
from autowisp.diagnostics.expression_library import (
    get_expressions,
    photref_merit,
)
from autowisp.evaluator import nanrank

# false positive due to unusual importing
# pylint: disable-next=no-name-in-module
from autowisp.database.data_model import Image

from autowisp.bui_util import encode_fits
from .display_fits_util import update_fits_display

_logger = logging.getLogger(__name__)


def _get_merit(request, db_session):
    """
    Return the library expressions to rank by, and the one to rank by now.

    The one chosen on the page if it can still rank candidates, otherwise
    :data:`photref_merit`; None if that cannot either, e.g. deleted.
    """

    choices = get_merit_expressions(get_expressions(db_session))
    for merit in (request.session.get("photref_merit"), photref_merit):
        if merit in choices:
            return choices, merit
    return choices, None


def _get_missing_photref(request):
    """Add all frame sets missing photometric reference to the session."""

    assert "need_photref" not in request.session
    processing = ImageProcessingManager(pipeline_run_id=None)
    with start_db_session() as db_session:
        result = compute_photref_candidates(processing, db_session)

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


def _get_merit_data(request, batch):
    """
    Return the candidates for a group's reference, best first.

    Evaluated on every request rather than kept, so that a change to the
    merit expression or to the exclusion rule shows on the next one. Only
    the images offered as the reference are ranked: not those the magnitude
    fitting exclusion rule leaves out.

    Returns:
        tuple:
            pandas.DataFrame:    A row per candidate, best first, indexed by
                its position in *batch*, which is what a choice is mapped
                back through. A column per diagnostic, and one for the merit
                under its name.

            str or None:    The name of the merit ranked by, None if none.

            str:    What to say about the images not offered.
    """

    with start_db_session() as db_session:
        merit = _get_merit(request, db_session)[1]
        exclusions = get_group_exclusions(batch, db_session)
        offered = get_offered_candidates(batch, exclusions)
        positions, values = rank_photref_candidates(
            batch, merit, offered, db_session
        )
    return (
        pandas.DataFrame(values, index=positions),
        merit,
        _describe_held_back(exclusions, len(offered) - sum(offered)),
    )


def _describe_held_back(exclusions, num_held_back):
    """Return what to say about the images not offered as the reference."""

    if not exclusions:
        return ""
    if "error" in exclusions:
        return (
            "Every image is offered: the magfit exclusion rule "
            f"{exclusions['rule']} is refused. {exclusions['error']}"
        )
    if num_held_back == 0:
        return (
            f"Every image is offered: the magfit exclusion rule "
            f"{exclusions['rule']} excludes them all."
            if exclusions["excluded"]
            else ""
        )
    return (
        f"{num_held_back} image(s) the magfit exclusion rule "
        f"{exclusions['rule']} excludes are not offered."
    )


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
    merit_data, merit, image_index, max_photref_separation=0.2
):
    """
    Create SVG histograms of the candidates' merit and every diagnostic.

    After the pointing plots, which show what the image shown would leave
    unbound as the reference, the merit comes first. Each histogram marks
    the image shown, and the quantile in a diagnostic's title is among the
    candidates.
    """

    matplotlib.use("svg")
    pyplot.style.use("dark_background")
    result = []

    result.extend(
        _create_pointing_plots(merit_data, image_index, max_photref_separation)
    )

    for column in sorted(merit_data.columns, key=lambda name: name != merit):
        if merit_data[column].isna().all():
            continue
        fig, ax = pyplot.subplots()
        ax.hist(
            merit_data[column].dropna(), bins="auto", linewidth=0, color="white"
        )
        ax.axvline(
            x=merit_data[column].iloc[image_index], linewidth=5, color="red"
        )
        if column == merit:
            fig.suptitle(f"merit: {merit}", fontsize=32)
        else:
            quantile = nanrank(merit_data[column])[image_index]
            fig.suptitle(column + f" ({quantile:.3f} quantile)", fontsize=32)
        result.append(create_svg(fig))
    return result


def select_photref_image(request, *, target_index):
    """Display the interface for reviewing canditate reference frames."""

    assert request.method == "GET"
    if "need_photref" not in request.session:
        return redirect("processing:select_photref_target")
    _logger.debug("Image view with request: %s", repr(request))
    update_fits_display(request)
    image_index = request.session["fits_display"]["image_index"]
    batch = request.session["need_photref"]["master_values"][target_index][1]
    merit_data, merit, photref_note = _get_merit_data(request, batch)
    if merit is None:
        photref_note = " ".join(
            filter(
                None,
                [
                    photref_note,
                    f"Candidates are in order of time: the library has no "
                    f"{photref_merit} expression, and no other was chosen to "
                    "rank by.",
                ],
            )
        )
    fits_fname, dr_fname = batch[int(merit_data.index[image_index])][:2]

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
        "dr_fname": dr_fname,
        "num_images": merit_data.shape[0],
        "histograms": _create_merit_histograms(
            merit_data, merit, image_index, max_photref_separation
        ),
        "fits_fname": path.basename(fits_fname),
        "view_config": request.session.get("view_config", "undefined"),
        "photref_note": photref_note,
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

    if recalc or request.method == "POST":
        # Refresh posts the form too, so the merit chosen survives the flush
        # that makes the photref groups be derived again.
        merit = request.POST.get("merit")
        if recalc:
            request.session.flush()
        if merit:
            request.session["photref_merit"] = merit
        return redirect("processing:select_photref_target")
    if "need_photref" not in request.session:
        _get_missing_photref(request)

    _logger.debug(
        "Request master values: %s",
        repr(request.session["need_photref"]["master_values"]),
    )
    # Recomputed on every visit, which is how a selection just recorded
    # shows here: this page is where recording one returns to. Each group
    # still needing a reference says what the magfit rule would exclude of
    # it, to inform the choice; each reference selected, what every step's
    # rule excludes of the images bound to it.
    with start_db_session() as db_session:
        targets = [
            {
                "values": target[0] + [len(target[1])],
                "exclusions": get_group_exclusions(target[1], db_session),
            }
            for target in request.session["need_photref"]["master_values"]
        ]
        exclusion_reports = get_photref_exclusions(db_session)
        merit_choices, merit = _get_merit(request, db_session)
    return render(
        request,
        "processing/select_photref_target.html",
        {
            "master_expressions": request.session["need_photref"][
                "master_expressions"
            ]
            + ["Num. Images", "Excluded by magfit rule"],
            "targets": targets,
            "merit_choices": merit_choices,
            "merit": merit,
            "view_config": request.body,
            "exclusion_reports": [
                dict(report, photref_name=path.basename(report["photref"]))
                for report in exclusion_reports
            ],
        },
    )


def record_photref_selection(request):
    """
    Record the single photometric reference whose DR file is ``?photref=``.

    The frame is named by its DR file rather than by its position among the
    ranked candidates, since the ranking is evaluated again on every request.
    """

    # The selection is recorded by following a plain link, so the browser can
    # re-issue this GET (refresh, back button, double click). The groups are
    # dropped below, so by then no group holds it and nothing happens.
    dr_fname = request.GET.get("photref")
    groups = request.session.get("need_photref", {}).get("master_values", [])
    containing = [
        index
        for index, (_, batch) in enumerate(groups)
        if any(entry[1] == dr_fname for entry in batch)
    ]
    if not containing:
        return redirect("processing:select_photref_target")
    if request.session["demo"]:
        _logger.info("Demo only! Not saving selected reference!")
        return redirect("processing:select_photref_target")
    batch = groups[containing[0]][1]

    record_single_photref(dr_fname, batch)

    # Force full re-derivation of the photref selection list on next page load
    request.session.pop("need_photref", None)
    request.session.modified = True

    return redirect("/processing/select_photref_target")
