"""Implement views for tuning source extraction."""

import json
import logging
from traceback import print_exc
from functools import reduce

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from sqlalchemy import select, sql

from autowisp.source_finder import SourceFinder, Evaluator
from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.astrometry import estimate_transformation
from autowisp.fits_utilities import get_primary_header
from autowisp.catalog import ensure_catalog, get_catalog_config
from autowisp.processing_steps.solve_astrometry import (
    construct_transformation,
    prepare_configuration,
)

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
    AlternateParameterName,
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
    id_expressions = {
        "Telescope": "TELSCPID",
        "Camera": "CAMERAID",
    }
    for component in ["Telescope", "Camera"]:
        sn_expression = id_expressions[component]
        for instrument_type in db_session.scalars(
            select(getattr(provenance, component + "Type"))
        ).all():
            serial_numbers = set(
                instrument.serial_number
                for instrument in getattr(
                    instrument_type, component.lower() + "s"
                )
            )
            if len(serial_numbers) == 1:
                serial_number = next(iter(serial_numbers))
                match_expression = f"{sn_expression} == {serial_number!r}"
            else:
                match_expression = f"{sn_expression} in {serial_numbers!r}"
            grouping_expressions.append(
                (
                    match_expression,
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

    processing = ImageProcessingManager(pipeline_run_id=None)

    with start_db_session() as db_session:
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
                # Ensure DB-backed header fields like CAMERAID/TELSCPID exist
                # for grouping expressions when they are missing in FITS.
                evaluator.symtable.update(
                    processing._get_extra_header(image)
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

    with start_db_session() as db_session:
        configured = set(
            notes.split(":", 1)[1].strip()
            for notes in db_session.scalars(
                select(Condition.notes).where(  # pylint: disable=no-member
                    Condition.notes.like(  # pylint: disable=no-member
                        "BUI tuned source extraction for: %"
                    )
                )
            ).all()
        )
        logging.info("Found configured: %r", configured)

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

    defaults = {
        "srcfind_tool": "fistar",
        "threshold_mode": "brightness-threshold",
        "brightness_threshold": "1000",
        "brightness_quantile": "0.999",
        "brightness_quantile_scale": "1.0",
        "filter_sources": "True",
        "srcextract_max_sources": "0",
    }

    try:
        evaluate = Evaluator(get_primary_header(context["fits_fname"]))
        processing = ImageProcessingManager(pipeline_run_id=None)
        with start_db_session() as db_session:
            config = processing.get_config(
                matched_expressions=processing.get_matched_expressions(
                    evaluate
                ),
                db_session=db_session,
                step_name="find_stars",
            )[0]

        brightness_threshold = config.get("brightness_threshold")
        defaults.update(
            {
                "srcfind_tool": str(config.get("srcfind_tool", "fistar")),
                "threshold_mode": (
                    "quantile"
                    if brightness_threshold is None
                    else "brightness-threshold"
                ),
                "brightness_threshold": (
                    "" if brightness_threshold is None else str(brightness_threshold)
                ),
                "brightness_quantile": str(
                    config.get("brightness_quantile", 0.999)
                ),
                "brightness_quantile_scale": str(
                    config.get("brightness_quantile_scale", 1.0)
                ),
                "filter_sources": str(
                    config.get("filter_sources", "True")
                ),
                "srcextract_max_sources": str(
                    config.get("srcextract_max_sources", 0)
                ),
            }
        )
    except Exception:  # pragma: no cover - keep tune UI available
        logging.exception(
            "Failed to load find_stars defaults from current configuration"
        )

    context.update(defaults)

    return render(request, "processing/tune_starfind.html", context)


def find_stars(request, fits_fname):
    """Run source extraction and respond with the results."""

    starfind_config = json.loads(request.body.decode())

    try:
        max_sources = int(starfind_config.get("max-sources") or "0")
        mode = starfind_config.get(
            "threshold-mode", "brightness-threshold"
        )

        find_stars_config = {
            "tool": starfind_config["srcfind-tool"],
            "filter_sources": starfind_config["filter-sources"],
            "max_sources": max_sources,
            "allow_overwrite": True,
            "allow_dir_creation": True,
        }

        if mode == "quantile":
            quantile = float(starfind_config["brightness-quantile"])
            quantile_scale = float(
                starfind_config["brightness-quantile-scale"]
            )
            if not 0.0 <= quantile <= 1.0:
                raise ValueError("Quantile must be between 0 and 1.")
            if quantile_scale <= 0.0:
                raise ValueError("Quantile scale must be positive.")
            find_stars_config.update(
                {
                    "brightness_threshold": None,
                    "brightness_quantile": quantile,
                    "brightness_quantile_scale": quantile_scale,
                }
            )
        else:
            threshold = float(starfind_config["brightness-threshold"])
            if threshold <= 0.0:
                raise ValueError("Brightness threshold must be positive.")
            find_stars_config["brightness_threshold"] = threshold

    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse(
            {
                "stars": [],
                "message": f"Invalid source extraction inputs: {error}",
            }
        )

    stars = SourceFinder(**find_stars_config)(fits_fname)
    request.session["extracted"] = {c: list(stars[c]) for c in "xy"}
    stars = {"stars": [{"x": s["x"], "y": s["y"]} for s in stars]}
    return JsonResponse(stars)


def project_catalog(request, fits_fname):
    """Solve for astrometry with current extracted stars and project catalog."""

    try:
        header = get_primary_header(fits_fname)
        evaluate = Evaluator(header)
        processing = ImageProcessingManager(pipeline_run_id=None)
        with start_db_session() as db_session:
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

        logging.info("Extracted: %r", request.session["extracted"])

        approx_trans, status = estimate_transformation(
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
                "anet_indices": config["anet_indices"],
                "anet_api_key": config["anet_api_key"],
                "x_cent": header["NAXIS1"] / 2,
                "y_cent": header["NAXIS2"] / 2,
            },
            header=header,
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

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    submitted_config = {
        param: request.POST[param]
        for param in request.POST
        if not param.endswith("token")
    }

    mode = submitted_config.get("threshold-mode", "brightness-threshold")
    if mode not in ["brightness-threshold", "quantile"]:
        if is_ajax:
            return JsonResponse(
                {"message": "Invalid threshold mode."},
                status=400,
            )
        return HttpResponse("Invalid threshold mode.", status=400)

    try:
        normalized_config = {
            "srcfind-tool": submitted_config["srcfind-tool"].strip(),
            "filter-sources": submitted_config["filter-sources"].strip(),
            "srcextract-max-sources": str(
                int(submitted_config.get("srcextract-max-sources", "0"))
            ),
        }

        if int(normalized_config["srcextract-max-sources"]) < 0:
            raise ValueError("Max sources must be non-negative.")

        if mode == "quantile":
            quantile = float(submitted_config["brightness-quantile"])
            quantile_scale = float(
                submitted_config["brightness-quantile-scale"]
            )
            if not 0.0 <= quantile <= 1.0:
                raise ValueError("Quantile must be between 0 and 1.")
            if quantile_scale <= 0.0:
                raise ValueError("Quantile scale must be positive.")
            normalized_config.update(
                {
                    "brightness-threshold": None,
                    "brightness-quantile": str(quantile),
                    "brightness-quantile-scale": str(quantile_scale),
                }
            )
        else:
            threshold = float(submitted_config["brightness-threshold"])
            if threshold <= 0.0:
                raise ValueError("Brightness threshold must be positive.")
            normalized_config["brightness-threshold"] = str(threshold)

    except (KeyError, TypeError, ValueError) as error:
        error_message = f"Invalid source extraction configuration: {error}"
        if is_ajax:
            return JsonResponse(
                {"message": error_message},
                status=400,
            )
        return HttpResponse(
            error_message,
            status=400,
        )

    condition_values = json.loads(
        request.session["starfind"]["pending"][imtype][batch_index][0]
    )
    grouping_expressions = request.session["starfind"]["grouping_expressions"]
    assert len(condition_values) == len(grouping_expressions)
    with start_db_session() as db_session:
        matched_expression_ids = set()
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
            matched_expression_ids.add(db_expression.id)

        existing_conditions = {}
        for cond_id, expr_id in db_session.execute(
            select(Condition.id, Condition.expression_id)
        ).all():
            if cond_id not in existing_conditions:
                existing_conditions[cond_id] = set()
            existing_conditions[cond_id].add(expr_id)

        matching_condition_ids = [
            cond_id
            for cond_id, expr_ids in existing_conditions.items()
            if expr_ids == matched_expression_ids
        ]
        if matching_condition_ids:
            condition_id = min(matching_condition_ids)
        else:
            condition_id = db_session.scalar(
                select(
                    sql.functions.max(Condition.id) + 1
                )
            )
            if condition_id is None:
                condition_id = 1

            condition_note = (
                "BUI tuned source extraction for: "
                + _get_batch_description(condition_values, grouping_expressions)
            )
            for expression_id in sorted(matched_expression_ids):
                db_session.add(
                    Condition(  # pylint: disable=not-callable
                        id=condition_id,
                        expression_id=expression_id,
                        notes=condition_note,
                    )
                )

        config_version = db_session.scalar(
            select(sql.functions.max(Configuration.version))
        )
        if config_version is None:
            config_version = 0

        param_ids = {}
        missing_params = []
        for param in normalized_config:
            param_id = db_session.scalar(
                select(Parameter.id).filter_by(name=param)
            )
            if param_id is None:
                alt_param = db_session.execute(
                    select(AlternateParameterName).filter_by(alt_name=param)
                ).scalar_one_or_none()
                if alt_param is not None:
                    param_id = alt_param.parameter.id
            if param_id is None:
                missing_params.append(param)
            else:
                param_ids[param] = param_id

        if missing_params:
            error_message = (
                "Missing parameters in pipeline database: "
                + ", ".join(sorted(missing_params))
            )
            if is_ajax:
                return JsonResponse(
                    {"message": error_message},
                    status=500,
                )
            return HttpResponse(error_message, status=500)

        for param in param_ids:
            db_config = db_session.execute(
                select(Configuration).filter_by(
                    parameter_id=param_ids[param],
                    condition_id=condition_id,
                    version=config_version,
                )
            ).scalar_one_or_none()
            if db_config is None:
                db_session.add(
                    Configuration(  # pylint: disable=not-callable
                        parameter_id=param_ids[param],
                        condition_id=condition_id,
                        version=config_version,
                        value=normalized_config[param],
                    )
                )
            else:
                db_config.value = normalized_config[param]

        if mode == "brightness-threshold":
            threshold_param_id = db_session.scalar(
                select(Parameter.id).filter_by(name="brightness-threshold")
            )
            if threshold_param_id is not None:
                db_threshold_config = db_session.execute(
                    select(Configuration).filter_by(
                        parameter_id=threshold_param_id,
                        condition_id=condition_id,
                        version=config_version,
                    )
                ).scalar_one_or_none()
                if db_threshold_config is not None:
                    db_threshold_config.value = normalized_config[
                        "brightness-threshold"
                    ]

    if is_ajax:
        return JsonResponse({"message": "Saved"})

    return redirect("/processing/select_starfind_batch")
