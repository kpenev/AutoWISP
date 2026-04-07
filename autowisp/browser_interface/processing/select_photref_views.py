"""Implement the view for selecting single photometric reference."""

from io import StringIO
from functools import reduce

# from PIL.ImageTransform import AffineTransform
from django.shortcuts import render, redirect
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select
import pandas
from astropy.coordinates import SkyCoord
from astropy import units as astropy_units

from autowisp.database.image_processing import (
    ImageProcessingManager,
    get_master_expression_ids,
    remove_failed_prerequisite,
)
from autowisp.database.interface import start_db_session
from autowisp.database.user_interface import get_processing_sequence
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.evaluator import Evaluator

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    MasterType,
    MasterFile,
    ConditionExpression,
    Step,
    Image,
    ImageDiagnostics,
    DiagnosticType,
    ImageMasterSelection,
)

# pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits
from .display_fits_util import update_fits_display


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


def _get_missing_photref(request):  # pylint: disable=too-many-locals
    """Add all frame sets missing photometric reference to the session."""

    assert "need_photref" not in request.session
    processing = ImageProcessingManager(pipeline_run_id=None)
    with start_db_session() as db_session:
        master_type_id = db_session.scalar(
            select(MasterType.id).filter_by(name="single_photref")
        )
        magfit_steps = [
            entry
            for entry in get_processing_sequence(db_session)
            if entry[0].name == "fit_magnitudes"
        ]
        processing.set_pending(db_session, magfit_steps)
        for step in magfit_steps:
            for pending in processing.pending[(step[0].id, step[1].id)]:
                processing.evaluate_expressions_image(pending[0], db_session)

        request.session["merit_function"] = (
            "1.0 / ((1.0 - qnt_s_center)**2 + qnt_bg_center**2)"
        )
        request.session["demo"] = False
        if not reduce(
            lambda x, y: bool(x) or bool(y), processing.pending.values(), False
        ):
            request.session["demo"] = True
            processing.set_pending(db_session, magfit_steps, True)

        astrom_step_id = db_session.scalar(
            select(Step.id).filter_by(name="solve_astrometry")
        )
        for (
            step_id,
            image_type_id,
        ), pending_images in processing.pending.items():
            remove_failed_prerequisite(
                pending_images, image_type_id, astrom_step_id, db_session
            )
            request.session["need_photref"] = {
                "master_expressions": [
                    db_session.scalar(
                        select(ConditionExpression.expression).filter_by(
                            id=expr_id
                        )
                    )
                    for expr_id in get_master_expression_ids(
                        step_id, image_type_id, db_session
                    )
                ],
                "master_values": [],
            }

            by_photref = processing.group_pending_by_conditions(
                pending_images,
                db_session,
                match_observing_session=False,
                step_id=step_id,
                masters_only=True,
            )
            for by_master_values, master_values in by_photref:
                if request.session["demo"]:
                    unbound_images = by_master_values
                else:
                    group_channel = by_master_values[0][1]
                    bound_image_ids = set(
                        db_session.scalars(
                            select(ImageMasterSelection.image_id).where(
                                ImageMasterSelection.master_type_id
                                == master_type_id,
                                ImageMasterSelection.channel == group_channel,
                                ImageMasterSelection.image_id.in_(
                                    [img.id for img, _, _ in by_master_values]
                                ),
                            )
                        ).all()
                    )
                    unbound_images = [
                        (img, ch, st)
                        for img, ch, st in by_master_values
                        if img.id not in bound_image_ids
                    ]
                if not unbound_images:
                    continue
                config = processing.get_config(
                    matched_expressions=None,
                    db_session=db_session,
                    image_id=unbound_images[0][0].id,
                    channel=unbound_images[0][1],
                    step_name="calculate_photref_merit",
                )[0]
                request.session["need_photref"]["master_values"].append(
                    (
                        list(master_values),
                        config,
                        [
                            (
                                processing.get_step_input(
                                    image, channel, "calibrated"
                                ),
                                processing.get_step_input(image, channel, "dr"),
                                image.id,
                                channel,
                            )
                            for image, channel, _ in unbound_images
                        ],
                    )
                )
    request.session.modified = True


def _get_merit_data(request, target_index):
    """Add to the session the merit information for selecting single ref."""

    if "merit_info" not in request.session:
        request.session["merit_info"] = {}
    if str(target_index) not in request.session["merit_info"]:
        print("Calculating merit for target " + str(target_index))
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


def _create_pointing_plots(
    merit_data,
    image_index,
    max_photref_separation=0.2,
    **plot_cfg,
):
    """
    Create SVG plots for pointing: RA vs Dec scatter and separation histogram.

    Images within max_photref_separation * diagonal_fov of the current image
    are drawn with in_range_cfg, those outside with out_of_range_cfg, and the
    current image itself with this_img_cfg.  RA axis is inverted per
    astronomical convention.  The separation histogram includes a vertical line
    at the threshold.

    Returns:
        List of SVG strings [scatter, separation_hist], or empty list if the
        required diagnostics (ra_center, dec_center, diagonal_fov) are absent.
    """

    if not all(
        col in merit_data.columns
        for col in ["ra_center", "dec_center", "diagonal_fov"]
    ):
        return []

    plot_cfg.setdefault("in_range_cfg", {"c": "green", "s": 20, "zorder": 3})
    plot_cfg.setdefault("out_of_range_cfg", {"c": "red", "s": 20, "zorder": 2})
    plot_cfg.setdefault("this_img_cfg", {"c": "white", "s": 100, "zorder": 4})

    ra_vals = merit_data["ra_center"].tolist()
    dec_vals = merit_data["dec_center"].tolist()

    plot_data = {
        "this_img": {"ra": ra_vals[image_index], "dec": dec_vals[image_index]},
    }
    threshold_deg = (
        max_photref_separation * merit_data["diagonal_fov"].iloc[image_index]
    )
    separations = (
        SkyCoord(
            ra=plot_data["this_img"]["ra"] * astropy_units.deg,
            dec=plot_data["this_img"]["dec"] * astropy_units.deg,
            frame="icrs",
        )
        .separation(
            SkyCoord(
                ra=merit_data["ra_center"].values * astropy_units.deg,
                dec=merit_data["dec_center"].values * astropy_units.deg,
                frame="icrs",
            )
        )
        .to_value(astropy_units.deg)
    )

    result = []

    plot_data["in_range"] = {
        "ra": [
            ra_vals[i]
            for i in range(len(ra_vals))
            if i != image_index and separations[i] <= threshold_deg
        ],
        "dec": [
            dec_vals[i]
            for i in range(len(dec_vals))
            if i != image_index and separations[i] <= threshold_deg
        ],
    }
    plot_data["out_of_range"] = {
        "ra": [
            ra_vals[i]
            for i in range(len(ra_vals))
            if i != image_index and separations[i] > threshold_deg
        ],
        "dec": [
            dec_vals[i]
            for i in range(len(dec_vals))
            if i != image_index and separations[i] > threshold_deg
        ],
    }

    with StringIO() as image_stream:
        fig, ax = pyplot.subplots()
        for mode in ["in_range", "out_of_range", "this_img"]:
            if mode == "this_img" or plot_data[mode]["ra"]:
                ax.scatter(
                    plot_data[mode]["ra"],
                    plot_data[mode]["dec"],
                    **plot_cfg[mode + "_cfg"],
                )
        ax.invert_xaxis()
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        pyplot.suptitle("Pointing (RA vs Dec)", fontsize=32)
        pyplot.savefig(image_stream, format="svg")
        result.append(image_stream.getvalue())
        pyplot.close(fig)

    with StringIO() as image_stream:
        fig, ax = pyplot.subplots()
        ax.hist(separations, bins="auto", linewidth=0, color="white")
        ax.axvline(x=threshold_deg, linewidth=2, color="lime", linestyle="--")
        ax.set_xlabel("Separation (deg)")
        pyplot.suptitle("Separation from current image", fontsize=32)
        pyplot.savefig(image_stream, format="svg")
        result.append(image_stream.getvalue())
        pyplot.close(fig)

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
        with StringIO() as image_stream:
            pyplot.hist(
                merit_data[column], bins="auto", linewidth=0, color="white"
            )
            pyplot.axvline(
                x=merit_data[column].iloc[image_index], linewidth=5, color="red"
            )
            if column == "merit":
                pyplot.suptitle("merit", fontsize=32)
            else:
                quantile = merit_data["qnt_" + column].iloc[image_index]
                pyplot.suptitle(
                    column + f" ({quantile:.3f} quantile)", fontsize=32
                )
            pyplot.savefig(image_stream, format="svg")
            result.append(image_stream.getvalue())
            pyplot.clf()
    return result


def select_photref_image(request, *, target_index, recalculate=False):
    """Display the interface for reviewing canditate reference frames."""

    assert request.method == "GET"
    if "need_photref" not in request.session:
        return redirect("processing:select_photref_target")
    print("Image view with request: " + repr(request))
    update_fits_display(request)
    image_index = request.session["fits_display"]["image_index"]
    if recalculate:
        print("Deleting merit info")
        del request.session["merit_info"]
    _get_merit_data(request, target_index)
    print("Merit info keys: " + repr(request.session["merit_info"].keys()))

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

    print(
        "Request master values: "
        + repr(request.session["need_photref"]["master_values"])
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


def _bind_images_to_photref(dr_fname, batch):  # pylint: disable=too-many-locals
    """
    Write ImageMasterSelection rows for batch images within distance of photref.

    Reads the fit_magnitudes config to get max_photref_separation (which may
    be conditional), then for each image in batch computes the angular
    separation between the image center and the photref center.  Images whose
    separation is within max_photref_separation * photref_diagonal_fov are
    bound to the photref via an upsert into ImageMasterSelection.

    Args:
        dr_fname(str):    Path to the photref DR file that was just registered.
        batch(list):    List of (calibrated_fname, dr_fname, image_id, channel)
            tuples — the candidate images from the same condition group.
    """

    with DataReductionFile(dr_fname, "r") as pf_dr:
        pf_header = pf_dr.get_frame_header()
    pf_rawfname = pf_header["RAWFNAME"]
    pf_channel = pf_header["CLRCHNL"]

    processing = ImageProcessingManager(pipeline_run_id=None)

    with start_db_session() as db_session:
        master_file = db_session.scalar(
            select(MasterFile).where(MasterFile.filename == dr_fname)
        )
        if master_file is None:
            return

        pf_image_id = db_session.scalar(
            select(Image.id).where(  # pylint:disable=no-member
                Image.raw_fname.like(  # pylint:disable=no-member
                    f"%/{pf_rawfname}.%"
                )
            )
        )
        if pf_image_id is None:
            return
        pf_diags = dict(
            db_session.execute(
                select(DiagnosticType.name, ImageDiagnostics.value)
                .join(
                    DiagnosticType,
                    ImageDiagnostics.diagnostic_id == DiagnosticType.id,
                )
                .where(
                    ImageDiagnostics.image_id == pf_image_id,
                    ImageDiagnostics.channel == pf_channel,
                    DiagnosticType.name.in_(
                        ["ra_center", "dec_center", "diagonal_fov"]
                    ),
                )
            ).all()
        )
        if not all(
            k in pf_diags for k in ["ra_center", "dec_center", "diagonal_fov"]
        ):
            return

        pf_center = SkyCoord(
            ra=pf_diags["ra_center"] * astropy_units.deg,
            dec=pf_diags["dec_center"] * astropy_units.deg,
            frame="icrs",
        )

        first_image_id, first_channel = batch[0][2], batch[0][3]
        first_image = db_session.get(Image, first_image_id)
        processing.evaluate_expressions_image(first_image, db_session)
        fit_config = processing.get_config(
            matched_expressions=None,
            db_session=db_session,
            image_id=first_image_id,
            channel=first_channel,
            step_name="fit_magnitudes",
        )[0]
        threshold_deg = (
            fit_config.get("max_photref_separation", 0.2)
            * pf_diags["diagonal_fov"]
        )

        for _, _, image_id, channel in batch:
            img_diags = dict(
                db_session.execute(
                    select(DiagnosticType.name, ImageDiagnostics.value)
                    .join(
                        DiagnosticType,
                        ImageDiagnostics.diagnostic_id == DiagnosticType.id,
                    )
                    .where(
                        ImageDiagnostics.image_id == image_id,
                        ImageDiagnostics.channel == channel,
                        DiagnosticType.name.in_(["ra_center", "dec_center"]),
                    )
                ).all()
            )
            if "ra_center" not in img_diags or "dec_center" not in img_diags:
                continue
            img_center = SkyCoord(
                ra=img_diags["ra_center"] * astropy_units.deg,
                dec=img_diags["dec_center"] * astropy_units.deg,
                frame="icrs",
            )
            if (
                pf_center.separation(img_center).to_value(astropy_units.deg)
                <= threshold_deg
            ):
                db_session.merge(
                    ImageMasterSelection(
                        image_id=image_id,
                        channel=channel,
                        master_type_id=master_file.type_id,
                        master_file_id=master_file.id,
                    )
                )


def record_photref_selection(request, target_index, image_index):
    """Record a single photometric reference frame selected by the user."""

    if request.session["demo"]:
        print("Demo only! Not saving selected reference!")
        return None
    print("Merit info keys: " + repr(request.session["merit_info"].keys()))
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
    _bind_images_to_photref(dr_fname, batch)

    # Force full re-derivation of the photref selection list on next page load
    request.session.pop("need_photref", None)
    request.session.pop("merit_info", None)
    request.session.modified = True

    return redirect("/processing/select_photref_target")
