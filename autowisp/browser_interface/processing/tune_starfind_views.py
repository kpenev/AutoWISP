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
    Image,
)
# pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits

from .display_fits_util import update_fits_display

STARFIND_SESSION_VERSION = 4

STARFIND_DEFAULT_CONFIG = {
    "srcfind-tool": "fistar",
    "filter-sources": "True",
    "srcextract-max-sources": "4000",
    "brightness-threshold": "1000",
    "brightness-quantile": "0.999",
    "brightness-quantile-scale": "1.0",
}

STARFIND_MANAGED_PARAMS = tuple(STARFIND_DEFAULT_CONFIG)

STARFIND_CONFIG_KEYS = {
    "srcfind-tool": "srcfind_tool",
    "filter-sources": "filter_sources",
    "srcextract-max-sources": "srcextract_max_sources",
    "brightness-threshold": "brightness_threshold",
    "brightness-quantile": "brightness_quantile",
    "brightness-quantile-scale": "brightness_quantile_scale",
}

BATCH_DISPLAY_ORDER = (0, 1, 3, 2)


def _init_session(request, processing, db_session):
    """Set default django session entries first time the interface is opened"""

    if (
        "starfind" in request.session
        and request.session["starfind"].get("version")
        == STARFIND_SESSION_VERSION
    ):
        return
    assert (
        len(processing.configuration["telescope-serial-number"]["value"]) == 1
    )
    assert len(processing.configuration["camera-serial-number"]["value"]) == 1

    exposure_expression = list(
        processing.configuration.get("exposure-seconds")["value"].values()
    )[0]
    grouping_expressions = [
        {
            "expression": "INTSN",
            "display_expression": "INTSN",
            "description": "{value} telescope",
        },
        {
            "expression": "CAMSN",
            "display_expression": "CAMSN",
            "description": "{value} camera",
        },
        {
            "expression": exposure_expression,
            "display_expression": exposure_expression,
            "description": "{value}s exposure",
        },
        {
            "expression": "CLRCHNL",
            "display_expression": "CLRCHNL",
            "description": "{value} channel",
        },
    ]

    request.session["starfind"] = {
        "version": STARFIND_SESSION_VERSION,
        "grouping_expressions": grouping_expressions,
    }


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
                # Ensure DB-backed header fields like INTSN/CAMSN exist
                # for grouping expressions when they are missing in FITS.
                evaluator.symtable.update(
                    processing._get_extra_header(image)
                )
                grouping_expressions = request.session["starfind"][
                    "grouping_expressions"
                ]
                display_values = [
                    evaluator(expr["display_expression"])
                    for expr in grouping_expressions
                ]
                grouping_key = json.dumps(
                    {
                        "condition_values": display_values,
                        "display_values": display_values,
                    },
                    sort_keys=True,
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


def _parse_batch_key(grouping_key):
    """Return normalized batch key data from the session representation."""

    batch_key = json.loads(grouping_key)
    if isinstance(batch_key, list):
        return {
            "condition_values": batch_key,
            "display_values": batch_key,
        }
    return batch_key


def _get_exact_batch_keys(batch):
    """Return exact condition keys represented by a visible batch."""

    return [_parse_batch_key(batch[0])]


def _get_batch_description(batch_key, grouping_expressions):
    """Return as human readable as possible discription of a batch."""

    display_values = batch_key["display_values"]
    return ", ".join(
        grouping_expressions[index]["description"].format(
            value=display_values[index]
        )
        for index in BATCH_DISPLAY_ORDER
        if (
            not isinstance(display_values[index], bool)
            or display_values[index]
        )
    )


def _get_condition_expression(expression, value):
    """Return database expression matching the given batch value."""

    if isinstance(value, bool):
        if not value:
            return None
        return expression["expression"]

    return f"{expression['expression']} == {value!r}"


def _iter_condition_expressions(batch_key, grouping_expressions):
    """Iterate over condition expressions matching the given batch."""

    return filter(
        None,
        (
            _get_condition_expression(expression, value)
            for expression, value in zip(
                grouping_expressions, batch_key["condition_values"]
            )
        ),
    )


def _get_existing_condition_id(
    db_session, batch_key, grouping_expressions
):
    """Return condition ID matching the batch exactly if it already exists."""

    expression_ids = set()
    for expression in _iter_condition_expressions(
        batch_key, grouping_expressions
    ):
        expression_id = db_session.scalar(
            select(ConditionExpression.id).filter_by(expression=expression)
        )
        if expression_id is None:
            return None
        expression_ids.add(expression_id)

    existing_conditions = {}
    for cond_id, expr_id in db_session.execute(
        select(Condition.id, Condition.expression_id)
    ).all():
        existing_conditions.setdefault(cond_id, set()).add(expr_id)

    matching_condition_ids = [
        cond_id
        for cond_id, expr_ids in existing_conditions.items()
        if expr_ids == expression_ids
    ]
    return min(matching_condition_ids) if matching_condition_ids else None


def _get_or_create_condition_id(
    db_session, batch_key, grouping_expressions
):
    """Return condition ID for the batch, creating condition rows as needed."""

    condition_id = _get_existing_condition_id(
        db_session, batch_key, grouping_expressions
    )
    if condition_id is not None:
        return condition_id

    expression_ids = []
    for grouping_expression, condition_value, display_value in zip(
        grouping_expressions,
        batch_key["condition_values"],
        batch_key["display_values"],
    ):
        match_expression = _get_condition_expression(
            grouping_expression, condition_value
        )
        if match_expression is None:
            continue
        db_expression = db_session.execute(
            select(ConditionExpression).filter_by(expression=match_expression)
        ).scalar_one_or_none()
        if db_expression is None:
            db_expression = ConditionExpression(
                expression=match_expression,
                notes=grouping_expression["description"].format(
                    value=display_value
                ),
            )
            db_session.add(db_expression)
            db_session.flush()
        expression_ids.append(db_expression.id)

    condition_id = db_session.scalar(select(sql.functions.max(Condition.id) + 1))
    if condition_id is None:
        condition_id = 1

    condition_note = "BUI tuned source extraction for: " + _get_batch_description(
        batch_key, grouping_expressions
    )
    for expression_id in expression_ids:
        db_session.add(
            Condition(  # pylint: disable=not-callable
                id=condition_id,
                expression_id=expression_id,
                notes=condition_note,
            )
        )
    db_session.flush()

    return condition_id


def _values_match(value, default_value):
    """Return True iff two configuration values are equivalent."""

    if value is None or default_value is None:
        return value is None and default_value is None

    try:
        return float(value) == float(default_value)
    except (TypeError, ValueError):
        return str(value) == str(default_value)


def _get_param_ids(db_session, param_names):
    """Return parameter IDs keyed by submitted/browser parameter name."""

    param_ids = {}
    missing_params = []
    for param in param_names:
        param_id = db_session.scalar(select(Parameter.id).filter_by(name=param))
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

    return param_ids, missing_params


def _get_config_version(db_session):
    """Return current DB configuration version."""

    return db_session.scalar(select(sql.functions.max(Configuration.version))) or 0


def _get_saved_starfind_config(db_session, condition_id):
    """Return BUI-managed source extraction config saved for condition."""

    if condition_id is None:
        return {}

    config_version = _get_config_version(db_session)
    param_ids, _ = _get_param_ids(db_session, STARFIND_MANAGED_PARAMS)
    by_id = {param_id: param for param, param_id in param_ids.items()}
    result = {}
    for parameter_id, value in db_session.execute(
        select(Configuration.parameter_id, Configuration.value).filter_by(
            condition_id=condition_id,
            version=config_version,
        )
    ).all():
        if parameter_id in by_id:
            result[by_id[parameter_id]] = value

    return result


def _get_tune_defaults():
    """Return default template context for the tune source extraction page."""

    return {
        "srcfind_tool": STARFIND_DEFAULT_CONFIG["srcfind-tool"],
        "threshold_mode": "brightness-threshold",
        "brightness_threshold": STARFIND_DEFAULT_CONFIG[
            "brightness-threshold"
        ],
        "brightness_quantile": STARFIND_DEFAULT_CONFIG[
            "brightness-quantile"
        ],
        "brightness_quantile_scale": STARFIND_DEFAULT_CONFIG[
            "brightness-quantile-scale"
        ],
        "filter_sources": STARFIND_DEFAULT_CONFIG["filter-sources"],
        "srcextract_max_sources": STARFIND_DEFAULT_CONFIG[
            "srcextract-max-sources"
        ],
    }


def _context_from_starfind_config(saved_config):
    """Return template context for saved source extraction values."""

    config = dict(STARFIND_DEFAULT_CONFIG)
    config.update(saved_config)
    return {
        "srcfind_tool": str(config["srcfind-tool"]),
        "threshold_mode": (
            "quantile"
            if config["brightness-threshold"] is None
            else "brightness-threshold"
        ),
        "brightness_threshold": (
            ""
            if config["brightness-threshold"] is None
            else str(config["brightness-threshold"])
        ),
        "brightness_quantile": str(config["brightness-quantile"]),
        "brightness_quantile_scale": str(
            config["brightness-quantile-scale"]
        ),
        "filter_sources": str(config["filter-sources"]),
        "srcextract_max_sources": str(config["srcextract-max-sources"]),
    }


def _desired_config_from_submission(submitted_config):
    """Validate submitted values and return normalized desired config."""

    mode = submitted_config.get("threshold-mode", "brightness-threshold")
    if mode not in ["brightness-threshold", "quantile"]:
        raise ValueError("Invalid threshold mode.")

    desired_config = dict(STARFIND_DEFAULT_CONFIG)
    desired_config.update(
        {
            "srcfind-tool": submitted_config["srcfind-tool"].strip(),
            "filter-sources": submitted_config["filter-sources"].strip(),
            "srcextract-max-sources": str(
                int(
                    submitted_config.get(
                        "srcextract-max-sources",
                        STARFIND_DEFAULT_CONFIG["srcextract-max-sources"],
                    )
                )
            ),
        }
    )

    if int(desired_config["srcextract-max-sources"]) < 0:
        raise ValueError("Max sources must be non-negative.")

    if mode == "quantile":
        quantile = float(submitted_config["brightness-quantile"])
        quantile_scale = float(submitted_config["brightness-quantile-scale"])
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Quantile must be between 0 and 1.")
        if quantile_scale <= 0.0:
            raise ValueError("Quantile scale must be positive.")
        desired_config.update(
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
        desired_config["brightness-threshold"] = str(threshold)

    return desired_config


def _config_value_to_string(value):
    """Return a database-compatible string value for source extraction config."""

    if value is None:
        return None

    return str(value)


def _starfind_config_from_pipeline_config(pipeline_config):
    """Return BUI parameter names and values from parsed pipeline config."""

    return {
        param: _config_value_to_string(
            pipeline_config.get(
                config_key,
                STARFIND_DEFAULT_CONFIG[param],
            )
        )
        for param, config_key in STARFIND_CONFIG_KEYS.items()
    }


def _get_batch_entry_config(
    processing,
    db_session,
    batch_entry,
    *,
    exclude_batch_key=None,
    grouping_expressions=None,
):
    """Return effective find_stars config for a pending batch entry."""

    image_id, channel = batch_entry[:2]
    image = db_session.get(Image, image_id)
    processing.evaluate_expressions_image(image, db_session)
    matched_expressions = set(
        processing._evaluated_expressions[image_id][channel]["matched"]
    )

    if exclude_batch_key is not None:
        assert grouping_expressions is not None
        matched_expressions -= _get_condition_expression_ids(
            db_session,
            exclude_batch_key,
            grouping_expressions,
        )

    return _starfind_config_from_pipeline_config(
        processing.get_config(
            matched_expressions,
            db_session,
            step_name="find_stars",
        )[0]
    )


def _get_condition_expression_ids(
    db_session, batch_key, grouping_expressions
):
    """Return DB expression IDs for existing expressions in a batch key."""

    expression_ids = set()
    for expression in _iter_condition_expressions(
        batch_key, grouping_expressions
    ):
        expression_id = db_session.scalar(
            select(ConditionExpression.id).filter_by(expression=expression)
        )
        if expression_id is not None:
            expression_ids.add(expression_id)

    return expression_ids


def _get_submitted_params(desired_config):
    """Return mode-relevant parameters to persist from the submitted config."""

    params = [
        "srcfind-tool",
        "filter-sources",
        "srcextract-max-sources",
        "brightness-threshold",
    ]
    if desired_config["brightness-threshold"] is None:
        params.extend(
            ["brightness-quantile", "brightness-quantile-scale"]
        )

    return params


def _changed_config_from_baseline(desired_config, baseline_config):
    """Return submitted values that differ from inherited frame config."""

    submitted_params = set(_get_submitted_params(desired_config))
    return {
        param: value
        for param, value in desired_config.items()
        if param in submitted_params
        and not _values_match(value, baseline_config[param])
    }


def _missing_parameter_response(is_ajax, missing_params):
    """Return an error response for missing pipeline parameters."""

    error_message = (
        "Missing parameters in pipeline database: "
        + ", ".join(sorted(missing_params))
    )
    if is_ajax:
        return JsonResponse({"message": error_message}, status=500)
    return HttpResponse(error_message, status=500)


def select_starfind_batch(request, refresh=False):
    """Allow the user to select batch of images to tune star finding for."""

    if refresh:
        request.session.flush()
        return redirect("/processing/select_starfind_batch")

    _get_pending(request)

    if "fits_display" in request.session:
        del request.session["fits_display"]

    context = {"batches": []}
    grouping_expressions = request.session["starfind"]["grouping_expressions"]
    with start_db_session() as db_session:
        for imtype_name, imtype_batches in request.session["starfind"][
            "pending"
        ].items():
            batch_info = []
            for grouping_key, batch in imtype_batches:
                batch_key = _parse_batch_key(grouping_key)
                batch_description = _get_batch_description(
                    batch_key,
                    grouping_expressions,
                )
                saved_configs = []
                for exact_batch_key in _get_exact_batch_keys(
                    (grouping_key, batch)
                ):
                    condition_id = _get_existing_condition_id(
                        db_session, exact_batch_key, grouping_expressions
                    )
                    saved_configs.append(
                        _get_saved_starfind_config(db_session, condition_id)
                    )
                batch_info.append(
                    (
                        batch_description,
                        len(batch),
                        any(saved_configs),
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

    defaults = _get_tune_defaults()

    try:
        with start_db_session() as db_session:
            defaults.update(
                _context_from_starfind_config(
                    _get_batch_entry_config(
                        ImageProcessingManager(pipeline_run_id=None),
                        db_session,
                        batch[1][image_index],
                    )
                )
            )
    except Exception:  # pragma: no cover - keep tune UI available
        logging.exception(
            "Failed to load saved find_stars values for current batch"
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

    try:
        desired_config = _desired_config_from_submission(submitted_config)
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

    batch = request.session["starfind"]["pending"][imtype][batch_index]
    grouping_expressions = request.session["starfind"]["grouping_expressions"]
    exact_batch_keys = _get_exact_batch_keys(batch)
    assert all(
        len(batch_key["condition_values"]) == len(grouping_expressions)
        for batch_key in exact_batch_keys
    )

    with start_db_session() as db_session:
        param_ids, missing_params = _get_param_ids(
            db_session, STARFIND_MANAGED_PARAMS
        )

        if missing_params:
            return _missing_parameter_response(is_ajax, missing_params)

        processing = ImageProcessingManager(pipeline_run_id=None)
        for batch_key in exact_batch_keys:
            inherited_config = _get_batch_entry_config(
                processing,
                db_session,
                batch[1][0],
                exclude_batch_key=batch_key,
                grouping_expressions=grouping_expressions,
            )
            changed_config = _changed_config_from_baseline(
                desired_config, inherited_config
            )
            condition_id = _get_existing_condition_id(
                db_session, batch_key, grouping_expressions
            )
            if changed_config and condition_id is None:
                condition_id = _get_or_create_condition_id(
                    db_session, batch_key, grouping_expressions
                )

            if condition_id is not None:
                config_version = _get_config_version(db_session)
                for param in STARFIND_MANAGED_PARAMS:
                    db_config = db_session.execute(
                        select(Configuration).filter_by(
                            parameter_id=param_ids[param],
                            condition_id=condition_id,
                            version=config_version,
                        )
                    ).scalar_one_or_none()
                    if param in changed_config:
                        if db_config is None:
                            db_session.add(
                                Configuration(  # pylint: disable=not-callable
                                    parameter_id=param_ids[param],
                                    condition_id=condition_id,
                                    version=config_version,
                                    value=changed_config[param],
                                )
                            )
                        else:
                            db_config.value = changed_config[param]
                    elif db_config is not None:
                        db_session.delete(db_config)

                has_remaining_config = db_session.scalar(
                    select(Configuration.parameter_id).filter_by(
                        condition_id=condition_id,
                        version=config_version,
                    )
                )
                if has_remaining_config is None:
                    for db_condition in db_session.scalars(
                        select(Condition).filter_by(id=condition_id)
                    ):
                        db_session.delete(db_condition)

    if is_ajax:
        return JsonResponse({"message": "Saved"})

    return redirect("/processing/select_starfind_batch")
