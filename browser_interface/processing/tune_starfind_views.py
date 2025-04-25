"""Implement views for tuning source extraction."""

import json
import logging
from traceback import print_exc
from functools import reduce

from django.shortcuts import render, redirect
from django.http import JsonResponse
from sqlalchemy import select, sql

from autowisp import SourceFinder, Evaluator
from autowisp.database.interface import Session
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.astrometry import estimate_transformation
from autowisp.fits_utilities import get_primary_header
from autowisp.catalog import ensure_catalog
from autowisp.processing_steps.solve_astrometry import (
    construct_transformation,
    prepare_configuration,
)
from autowisp.processing_steps.manual_util import get_catalog_config

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Step,
    ImageType,
    ProcessingSequence,
    ConditionExpression,
    Condition,
    Configuration,
    Parameter,
)
from autowisp.database.data_model import provenance

# pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits

from .display_fits_util import update_fits_display


def _init_session(request, processing, db_session):
    """Set default django session entries first time the interface is opened"""

    if "starfind" in request.session:
        return
    assert (
        len(processing.configuration["telescope-serial-number"]["value"]) == 1
    )
    assert len(processing.configuration["camera-serial-number"]["value"]) == 1

    grouping_expressions = []
    for component in ["Telescope", "Camera"]:
        sn_expression = list(
            processing.configuration.get(component.lower() + "-serial-number")[
                "value"
            ].values()
        )[0]
        for instrument_type in db_session.scalars(
            select(getattr(provenance, component + "Type"))
        ).all():
            serial_numbers = set(
                instrument.serial_number
                for instrument in getattr(
                    instrument_type, component.lower() + "s"
                )
            )
            grouping_expressions.append(
                (
                    f"{sn_expression} in {serial_numbers!r}",
                    f"{instrument_type.make} {instrument_type.model} "
                    f"{component.lower()}s",
                )
            )
    grouping_expressions.extend(
        [
            ("CLRCHNL", "{value} channel"),
            (
                list(
                    processing.configuration.get("exposure-seconds")[
                        "value"
                    ].values()
                )[0],
                "{value}s exposure",
            ),
        ]
    )

    request.session["starfind"] = {"grouping_expressions": grouping_expressions}


def _get_pending(request):
    """Add to ``request.session`` all image/channel pending star finding ."""

    processing = ImageProcessingManager(dummy=True)

    # False positivie
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        _init_session(request, processing, db_session)
        if "pending" in request.session["starfind"]:
            return

        request.session["starfind"]["pending"] = {}
        find_star_steps = db_session.execute(
            select(Step, ImageType)
            .select_from(ProcessingSequence)
            .join(Step, ProcessingSequence.step_id == Step.id)
            .join(ImageType, ProcessingSequence.image_type_id == ImageType.id)
            .where(Step.name == "find_stars")
        ).all()

        processing.set_pending(db_session, find_star_steps)
        if not reduce(
            lambda x, y: bool(x) or bool(y), processing.pending.values(), False
        ):
            processing.set_pending(db_session, find_star_steps, True)
        for step, imtype in find_star_steps:
            grouping = {}
            for image, channel, _ in processing.pending[step.id, imtype.id]:
                processing.evaluate_expressions_image(image, db_session)
                evaluator = Evaluator(
                    processing.get_product_fname(
                        image.id, channel, "calibrated"
                    )
                )
                grouping_key = json.dumps(
                    [
                        evaluator(expr)
                        for expr, _ in request.session["starfind"][
                            "grouping_expressions"
                        ]
                    ]
                )
                if grouping_key not in grouping:
                    grouping[grouping_key] = []
                grouping[grouping_key].append(
                    (
                        image.id,
                        channel,
                        processing.get_step_input(image, channel, "calibrated"),
                    )
                )
            request.session["starfind"]["pending"][imtype.name] = sorted(
                grouping.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )


def _get_batch_description(grouping_values, grouping_expressions):
    """Return as human readable as possible discription of a batch."""

    return ", ".join(
        expr[1].format(value=value)
        for value, expr in zip(grouping_values, grouping_expressions)
        if not isinstance(value, bool) or value
    )


def select_starfind_batch(request, refresh=False):
    """Allow the user to select batch of images to tune star finding for."""

    if refresh:
        request.session.flush()
        return redirect("/processing/select_starfind_batch")

    _get_pending(request)

    if "fits_display" in request.session:
        del request.session["fits_display"]

    # False positivie
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        configured = set(
            notes.split(":", 1)[1].strip()
            for notes in db_session.scalars(
                select(Condition.notes).where(
                    Condition.notes.like("BUI tuned source extraction for: %")
                )
            ).all()
        )
        logging.info("Found configured: " + repr(configured))

    context = {"batches": []}
    for imtype_name, imtype_batches in request.session["starfind"][
        "pending"
    ].items():
        batch_info = []
        for grouping_values, batch in imtype_batches:
            batch_description = _get_batch_description(
                json.loads(grouping_values),
                request.session["starfind"]["grouping_expressions"],
            )
            batch_info.append(
                (
                    batch_description,
                    len(batch),
                    batch_description.strip() in configured,
                )
            )

        context["batches"].append((imtype_name, batch_info))
    return render(request, "processing/select_starfind_batch.html", context)


def tune_starfind(request, imtype, batch_index):
    """Provide view allowing user to tune starfinding for given image batch."""

    batch = request.session["starfind"]["pending"][imtype][batch_index]
    update_fits_display(request)
    image_index = request.session["fits_display"]["image_index"]
    context = encode_fits(
        batch[1][image_index][2],
        request.session["fits_display"]["range"],
        request.session["fits_display"]["transform"],
    )
    context["num_images"] = len(batch[1])
    context.update(request.session["fits_display"])
    context["image_index1"] = context["image_index"] + 1
    context["fits_fname"] = batch[1][image_index][2]
    context["imtype"] = imtype
    context["batch_index"] = batch_index

    return render(request, "processing/tune_starfind.html", context)


def find_stars(request, fits_fname):
    """Run source extraction and respond with the results."""

    starfind_config = json.loads(request.body.decode())

    stars = SourceFinder(
        tool=starfind_config["srcfind-tool"],
        brightness_threshold=float(starfind_config["brightness-threshold"]),
        filter_sources=starfind_config["filter-sources"],
        max_sources=int(starfind_config["max-sources"] or "0"),
        allow_overwrite=True,
        allow_dir_creation=True,
    )(fits_fname)
    request.session["extracted"] = {c: list(stars[c]) for c in "xy"}
    stars = {"stars": [{"x": s["x"], "y": s["y"]} for s in stars]}
    return JsonResponse(stars)


def project_catalog(request, fits_fname):
    """Solve for astrometry with current extracted stars and project catalog."""

    try:
        header = get_primary_header(fits_fname)
        evaluate = Evaluator(header)
        processing = ImageProcessingManager(dummy=True)
        with Session.begin() as db_session:
            config = prepare_configuration(
                processing.get_config(
                    matched_expressions=processing.get_matched_expressions(
                        evaluate
                    ),
                    db_session=db_session,
                    step_name="solve_astrometry",
                )[0],
                header,
            )
        fov_estimate = max(config["frame_fov_estimate"]).to_value("deg")
        config["frame_center_estimate"] = [
            evaluate(val).to_value("deg")
            for val in config["frame_center_estimate"]
        ]

        approx_trans = {
            coord + "_cent": value
            for coord, value in zip(
                ["ra", "dec"], config["frame_center_estimate"]
            )
        }

        logging.info("Extracted: " + repr(request.session["extracted"]))

        (approx_trans["trans_x"], approx_trans["trans_y"], status) = (
            estimate_transformation(
                dr_file=None,
                xy_extracted=request.session["extracted"],
                config={
                    "astrometry_order": config["tweak_order"][1],
                    "tweak_order_range": (
                        config["tweak_order"][0],
                        config["tweak_order"][1] + 1,
                    ),
                    "fov_range": (
                        fov_estimate / config["image_scale_factor"],
                        fov_estimate * config["image_scale_factor"],
                    ),
                    "ra_cent": config["frame_center_estimate"][0],
                    "dec_cent": config["frame_center_estimate"][1],
                    "anet_indices": config["anet_indices"],
                },
                header=header,
            )
        )
        if status != "success":
            return JsonResponse(
                {"stars": [], "message": "Projecting catalog sources failed!"}
            )
        approx_trans = construct_transformation(approx_trans)

        catalog = ensure_catalog(
            transformation=approx_trans,
            header=header,
            configuration=get_catalog_config(config, "astrometry"),
            return_metadata=False,
        )[0]
        projected = approx_trans(catalog)
        return JsonResponse(
            {"stars": [{"x": s["x"], "y": s["y"]} for s in projected]}
        )
    except:
        print_exc()
        raise


def save_starfind_config(request, imtype, batch_index):
    """Save the currently set extraction configuration to the database."""

    starfind_config = {
        param: request.POST[param]
        for param in request.POST
        if not param.endswith("token")
    }

    condition_values = json.loads(
        request.session["starfind"]["pending"][imtype][batch_index][0]
    )
    grouping_expressions = request.session["starfind"]["grouping_expressions"]
    assert len(condition_values) == len(grouping_expressions)
    # False positivie
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        param_ids = {
            param: db_session.scalar(select(Parameter.id).filter_by(name=param))
            for param in starfind_config
        }
        condition_id = db_session.scalar(
            select(sql.functions.max(Condition.id) + 1)
        )

        for expression, value in zip(grouping_expressions, condition_values):
            if isinstance(value, bool):
                if not value:
                    continue
                match_expression = expression[0]
            else:
                match_expression = f"{expression[0]} == {value!r}"
            db_expression = db_session.execute(
                select(ConditionExpression).filter_by(
                    expression=match_expression
                )
            ).scalar_one_or_none()
            if db_expression is None:
                db_expression = ConditionExpression(
                    expression=match_expression,
                    notes=expression[1].format(value=value),
                )
                db_session.add(db_expression)
                db_session.flush()
            db_session.add(
                Condition(
                    id=condition_id,
                    expression_id=db_expression.id,
                    notes=(
                        "BUI tuned source extraction for: "
                        + _get_batch_description(
                            condition_values, grouping_expressions
                        )
                    ),
                )
            )
        for param in starfind_config:
            db_session.add(
                Configuration(
                    parameter_id=param_ids[param],
                    condition_id=condition_id,
                    version=0,
                    value=starfind_config[param],
                )
            )
    return redirect("/processing/select_starfind_batch")
