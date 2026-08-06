# pylint: disable=too-many-lines
"""Define interface to the pipeline database."""

import copy
import json
import logging
import re
import sys
from time import sleep
from traceback import print_exc

from configargparse import ArgumentParser, DefaultsFormatter

from sqlalchemy import sql, select, delete, inspect, func, and_
from sqlalchemy.orm import ColumnProperty

from autowisp.database import defaults as database_defaults
from autowisp.database.interface import start_db_session, set_project_home
from autowisp.exceptions import ConfigurationError
from autowisp.database.data_model import provenance
from autowisp.data_reduction.data_reduction_file import DataReductionFile

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model.provenance import (
    Camera,
    CameraType,
    CameraChannel,
)

from autowisp.database.data_model import (
    Condition,
    ConditionExpression,
    Configuration,
    Image,
    ImageProcessingProgress,
    ImageType,
    LightCurveProcessingProgress,
    MasterFile,
    MasterType,
    ObservingSession,
    Parameter,
    ProcessedImages,
    ProcessingSequence,
    Step,
    StepDependencies,
    step_param_association,
)

# pylint: enable=no-name-in-module


def get_db_configuration(
    version, db_session, step_id=None, max_version_only=False
):
    """Return list of Configuration instances given version."""

    # False positives:
    # pylint: disable=no-member
    param_version_subq = (
        select(
            Configuration.parameter_id,
            # False positivie
            # pylint: disable=not-callable
            sql.func.max(Configuration.version).label("version"),
            # pylint: enable=not-callable
        )
        .filter(Configuration.version <= version)
        .group_by(Configuration.parameter_id)
        .subquery()
    )

    config_select = select(
        func.max(Configuration.version) if max_version_only else Configuration
    ).join(
        param_version_subq,
        sql.expression.and_(
            (Configuration.parameter_id == param_version_subq.c.parameter_id),
            (Configuration.version == param_version_subq.c.version),
        ),
    )
    if step_id is not None:
        config_select = config_select.join(
            step_param_association,
            Configuration.parameter_id == step_param_association.c.param_id,
        ).where(step_param_association.c.step_id == step_id)

    if max_version_only:
        return db_session.scalars(config_select).one()
    return db_session.scalars(config_select).all()
    # pylint: enable=no-member


def get_processing_sequence(db_session, no_lc_postprocessing=False):
    """
    Return the sequence of (step, image type) the pipeline can run.

    Args:
        db_session:    The database session to issue queries.

        no_lc_postprocessing(bool):    If True, any steps which are blocked by
            ``create_lightcurves`` are not included.
    """

    select_seq = (
        select(Step, ImageType)
        .select_from(ProcessingSequence)
        .join(Step, ProcessingSequence.step_id == Step.id)
        .join(ImageType, ProcessingSequence.image_type_id == ImageType.id)
    )
    if no_lc_postprocessing:
        create_lc_step_id = db_session.scalar(
            select(Step.id).filter_by(name="create_lightcurves")
        )
        select_postprocessing = (
            select(StepDependencies)
            .filter_by(blocking_step_id=create_lc_step_id)
            .subquery()
        )
        select_seq = select_seq.outerjoin(
            select_postprocessing,
            and_(
                select_postprocessing.c.blocked_step_id == Step.id,
                select_postprocessing.c.blocked_image_type_id == ImageType.id,
            ),
        ).where(
            select_postprocessing.c.blocked_step_id  # pylint: disable=singleton-comparison
            == None
        )

    return db_session.execute(select_seq.order_by(ProcessingSequence.id)).all()


def list_channels(db_session):
    """List the combine set of channels for all cameras."""

    return db_session.scalars(func.distinct(CameraChannel.name)).all()


def get_progress_images(step_id, image_type_id, config_version, db_session):
    """
    Return number of images in final state and by status for given step/imtype.

    Args:
        step:    Step instance for which to return the progress.

        image_type:    ImageType instance for which to return the progress.

        config_version:    Version of the configuration for which to report
            progress.

        db_session:    Database session to use.

    Returns:
        [str, int, int]:    Information on the images in final state. The
            entries are channel name, example status (>0 indicates success <0
            indicates falure), number of images of that channel
            that have that status sign and are flagged final.

        [str, int]:    Information about the images not in final state. The
            entries are channel name, number of non-final images of that
            channel.

        [str, int, int]:    The pending images broken by status. The format is
            the same as the final state information, except for images not
            flagged as in final state for the given step.
    """

    step_version = get_db_configuration(
        config_version, db_session, step_id, max_version_only=True
    )

    def complete_processed_select(_select):
        """Return the given select joined and filtered to given processed."""

        return _select.join(
            ImageProcessingProgress,
            ProcessedImages.progress_id == ImageProcessingProgress.id,
        ).where(
            ImageProcessingProgress.step_id == step_id,
            ImageProcessingProgress.configuration_version == step_version,
            ImageProcessingProgress.image_type_id == image_type_id,
        )

    select_image_channel = (
        select(
            CameraChannel.name,
            # False poisitive
            # pylint: disable=not-callable
            # pylint: disable=no-member
            func.count(Image.id),
            # pylint: enable=not-callable
            # pylint: enable=no-member
        )
        .join(
            ObservingSession,
        )
        .join(Camera)
        .join(CameraType)
        .join(CameraChannel)
    )

    processed_subq = (
        complete_processed_select(
            select(
                ProcessedImages.channel,
                func.max(ProcessedImages.status).label("status"),
                func.max(ProcessedImages.final).label("final"),
                # False poisitive
                # pylint: disable=not-callable
                ProcessedImages.image_id,
                # pylint: enable=not-callable
            )
            .join(Image)
            .join(ImageType)
        )
        .where(ImageType.id == image_type_id)
        .group_by(ProcessedImages.image_id, ProcessedImages.channel)
        .subquery()
    )
    processed_select = select(
        processed_subq.c.channel,
        processed_subq.c.status,
        func.count(processed_subq.c.image_id),  # pylint: disable=not-callable
    )

    final = db_session.execute(
        processed_select.where(processed_subq.c.final).group_by(
            processed_subq.c.status > 0,
            processed_subq.c.channel,
        )
    ).all()
    by_status = db_session.execute(
        processed_select.where(~processed_subq.c.final).group_by(
            processed_subq.c.status, processed_subq.c.channel
        )
    ).all()
    processed_subquery = (
        complete_processed_select(
            select(ProcessedImages.image_id, ProcessedImages.channel)
        )
        .where(ProcessedImages.final)
        .group_by(ProcessedImages.image_id, ProcessedImages.channel)
        .subquery()
    )

    pending = db_session.execute(
        select_image_channel.outerjoin(
            processed_subquery,
            # False positive
            # pylint: disable=no-member
            and_(
                Image.id == processed_subquery.c.image_id,
                CameraChannel.name == processed_subquery.c.channel,
            ),
            # pylint: enable=no-member
        )
        .where(
            # This is how NULL comparison is done in SQLAlchemy
            # pylint: disable=singleton-comparison
            # pylint: disable=no-member
            processed_subquery.c.image_id
            == None
            # pylint: enable=singleton-comparison
            # pylint: enable=no-member
        )
        .where(
            # pylint: disable=no-member
            Image.image_type_id
            == image_type_id
            # pylint: enable=no-member
        )
        .group_by(CameraChannel.name)
    ).all()
    return final, pending, by_status


def get_progress_lightcurves(
    step_id, image_type_id, config_version, db_session
):
    """Same as `get_progress_images()` but for lightcurve steps."""

    step_version = get_db_configuration(
        config_version, db_session, step_id, max_version_only=True
    )

    final = {}
    pending = {}
    for db_sphotref in db_session.scalars(
        select(MasterFile)
        .join(MasterType)
        .where(MasterType.name == "single_photref")
    ).all():
        for _ in range(10):
            try:
                with DataReductionFile(
                    db_sphotref.filename, "r"
                ) as sphotref_dr:
                    header = sphotref_dr.get_frame_header()
                    if (
                        not db_session.scalar(
                            select(ImageType.id)
                            .select_from(Image)
                            .join(ImageType)
                            .where(
                                Image.raw_fname.contains(  # pylint: disable=no-member
                                    header["RAWFNAME"] + ".fits"
                                )
                            )
                        )
                        == image_type_id
                    ):
                        continue
                    channel = header["CLRCHNL"]
                    if channel not in final:
                        final[channel] = 0
                    if channel not in pending:
                        pending[channel] = 0

                    if db_session.scalar(
                        select(
                            func.max(LightCurveProcessingProgress.final)
                        ).filter_by(
                            step_id=step_id,
                            single_photref_id=db_sphotref.id,
                            configuration_version=step_version,
                        )
                    ):
                        final[channel] += 1
                    else:
                        pending[channel] += 1
                break
            # h5py refuses to provide public interface to exceptions
            # pylint: disable=bare-except
            except:
                sleep(10)
            # pylint: enable=bare-except

    return (
        [(channel, 1, count) for channel, count in final.items()],
        list(pending.items()),
        [],
    )


def get_progress(step, *args, **kwargs):
    """Return info about completed work ona given step."""

    if step.name in [
        "epd",
        "tfa",
        "generate_epd_statistics",
        "generate_tfa_statistics",
    ]:
        return get_progress_lightcurves(step.id, *args, **kwargs)

    return get_progress_images(step.id, *args, **kwargs)


def _get_config_info(version, step="All"):
    """Return info for displaying the configuration with given version."""

    with start_db_session() as db_session:
        if step != "All":
            restrict_param_ids = set(
                param.id
                for param in db_session.scalar(
                    select(Step).filter_by(name=step)
                ).parameters
            )

        config_list = get_db_configuration(version, db_session)
        config_info = {}
        for config in config_list:
            if (
                step != "All"
                and config.parameter.id
                not in restrict_param_ids  # pylint: disable=possibly-used-before-assignment
            ):
                continue
            if config.parameter.name not in config_info:
                config_info[config.parameter.name] = {
                    "values": {},
                    "expression_counts": {},
                    "description": config.parameter.description,
                }
            param_info = config_info[config.parameter.name]
            param_info["values"][config.value] = set(
                expr.expression
                for expr in config.condition_expressions
                if expr.expression != "True"
            )
            for expression in config.condition_expressions:
                param_info["expression_counts"][expression.expression] = (
                    param_info["expression_counts"].get(
                        expression.expression, 0
                    )
                    + 1
                )
    return config_info


def get_json_config(version=0, step="All", **dump_kwargs):
    """Return the configuration as a JSON object."""

    def get_children(values, expression_order):
        """Return the sub-tree for the given expressions."""

        result = []
        child_values = {}
        sibling_values = {}
        for value, val_expressions in values.items():
            if not val_expressions:
                result.append({"name": value, "type": "value", "children": []})
            elif expression_order[0] in val_expressions:
                child_values[value] = val_expressions - set(
                    [expression_order[0]]
                )
            else:
                sibling_values[value] = val_expressions

        if child_values:
            result.append(
                {
                    "name": expression_order[0],
                    "type": "condition",
                    "children": get_children(
                        child_values, expression_order[1:]
                    ),
                }
            )
        if sibling_values:
            result.extend(get_children(sibling_values, expression_order[1:]))
        return result

    config_data = {
        "name": "All" if step == "All" else step,
        "type": "step",
        "children": [],
    }
    for param, param_info in _get_config_info(version, step).items():
        expression_order = [
            expr_count[0]
            for expr_count in sorted(
                param_info["expression_counts"].items(),
                key=lambda expr_count: expr_count[1],
                reverse=True,
            )
        ]
        config_data["children"].append(
            {
                "name": param,
                "type": "parameter",
                "description": param_info["description"],
                "children": get_children(
                    param_info["values"], expression_order
                ),
            }
        )
    return json.dumps(config_data, **dump_kwargs)


def _parse_json_config(json_config):
    """
    Organize the given JSON configuration to parameters and conditions.

    Args:
        json_config: JSON configuration to be parsed. Formatted as a decision
            tree, where the path through the tree defines the combination of
            condition expressions that must be satisfied and the leaf at the end
            specifies the value for the parameter .

    Returns:
        dict:
            parameter name: [
                {
                    'expressions': set(expression ID index in below list),

                    'value': value of parameter if all expressions are satisfied

                },

                ...

            ]

        [str]:
            list of expression strings
    """

    result = {}
    expression_list = []

    # Rejecting every malformed node shape with its own message is
    # inherently branchy; the alternative is one vague error for all of
    # them.
    # pylint: disable=too-many-branches
    def walk_json(sub_tree, parameter=None, expression_ids=None):
        """Recursively walk the JSON configuration tree adding to results."""

        if sub_tree["type"] == "parameter":
            if parameter is not None or expression_ids is not None:
                raise ConfigurationError(
                    f'Parameter {sub_tree["name"]} is nested under parameter '
                    f"{parameter} in the JSON configuration; parameters must "
                    "be at the top level!"
                )
            if sub_tree["name"] in result:
                raise ConfigurationError(
                    f'Parameter {sub_tree["name"]} is specified more than '
                    "once in the JSON configuration!"
                )
            if not sub_tree["children"]:
                raise ConfigurationError(
                    f'Parameter {sub_tree["name"]} in the JSON configuration '
                    "has no value or condition under it!"
                )
            for child in sub_tree["children"]:
                walk_json(child, sub_tree["name"], ())
        elif sub_tree["type"] == "value":
            if sub_tree["children"]:
                raise ConfigurationError(
                    f'Value {sub_tree["name"]} of parameter {parameter} has '
                    "further nodes under it in the JSON configuration; "
                    "values must be the leaves of the tree!"
                )
            if not parameter or expression_ids is None:
                raise ConfigurationError(
                    f'Value {sub_tree["name"]} in the JSON configuration is '
                    "not under any parameter!"
                )
            if parameter not in result:
                result[parameter] = []
            print(
                "Adding to parsed: "
                + repr(set(expression_ids))
                + " -> "
                + repr(sub_tree["name"])
            )
            result[parameter].append(
                {"expressions": set(expression_ids), "value": sub_tree["name"]}
            )
        elif sub_tree["type"] == "condition":
            if not sub_tree["children"]:
                raise ConfigurationError(
                    f'Condition {sub_tree["name"]} of parameter {parameter} '
                    "has no value or further condition under it in the JSON "
                    "configuration!"
                )
            if not parameter:
                raise ConfigurationError(
                    f'Condition {sub_tree["name"]} in the JSON configuration '
                    "is not under any parameter!"
                )
            try:
                condition_id = expression_list.index(sub_tree["name"])
            except ValueError:
                condition_id = len(expression_list)
                expression_list.append(sub_tree["name"])
            for child in sub_tree["children"]:
                walk_json(child, parameter, expression_ids + (condition_id,))
        else:
            raise ConfigurationError(
                f'Unexpected node type: {sub_tree["type"]} in JSON'
                " configuration"
            )

    for child in json_config["children"]:
        walk_json(child)

    return result, expression_list


def _get_db_conditions(db_session):
    """Return dict of condition IDs containing sets of expression IDs."""

    result = {}
    for condition_id, expression_id in db_session.execute(
        # False positive
        # pylint: disable=no-member
        select(Condition.id, Condition.expression_id)
        # pylint: enable=no-member
    ).all():
        if condition_id not in result:
            result[condition_id] = set()
        result[condition_id].add(expression_id)
    return result


def _save_expressions(expressions, db_session):
    """Save new expressions to database and update configuration with DB IDs."""

    expression_db_ids = [None for _ in expressions]
    for expr_ind, expression_str in enumerate(expressions):
        expression = db_session.execute(
            select(ConditionExpression).where(
                ConditionExpression.expression == (expression_str or "True")
            )
        ).scalar_one_or_none()
        if expression is None:
            expression = ConditionExpression(expression=expression_str)
            db_session.add(expression)
            db_session.flush()
        expression_db_ids[expr_ind] = expression.id
    return expression_db_ids


def _save_conditions(configuration, expression_db_ids, db_session):
    """Create new conditions encounted in configuration and add their IDs."""

    db_conditions = _get_db_conditions(db_session)
    print(
        "DB conditions:\n\t"
        + "\n\t".join(f"{k}: {v!r}" for k, v in db_conditions.items())
    )
    print("DB condition values: " + repr(db_conditions.values()))

    new_condition_id = db_session.scalar(
        # False positive
        # pylint: disable=no-member
        select(sql.functions.max(Condition.id) + 1)
        # pylint: enable=no-member
    )
    default_expression_set = set(
        [
            db_session.scalar(
                select(ConditionExpression.id).where(
                    ConditionExpression.expression == "True"
                )
            )
        ]
    )
    default_condition_id = [
        k for k, v in db_conditions.items() if v == default_expression_set
    ][0]

    for param_info in configuration.values():
        for param_condition in param_info:
            condition_expression_ids = (
                set(
                    expression_db_ids[expr_id]
                    for expr_id in param_condition["expressions"]
                )
                - default_expression_set
            )
            param_condition["expressions"] = condition_expression_ids
            if not condition_expression_ids:
                param_condition["condition_id"] = default_condition_id
            else:
                matching_condition = [
                    k
                    for k, v in db_conditions.items()
                    if v == condition_expression_ids
                ]
                if matching_condition:
                    param_condition["condition_id"] = matching_condition[0]
                else:
                    db_session.add_all(
                        # False positive
                        # pylint: disable=not-callable
                        Condition(
                            id=new_condition_id, expression_id=expression_id
                        )
                        # pylint: enable=not-callable
                        for expression_id in condition_expression_ids
                    )
                    param_condition["condition_id"] = new_condition_id
                    new_condition_id += 1


def save_json_config(json_config, version):
    """Save configuration provided in JSON format to the database."""

    configuration, expressions = _parse_json_config(
        json.loads(json_config.decode("ascii"))
    )
    with start_db_session() as db_session:
        compare_config = get_db_configuration(version, db_session)

        _save_conditions(
            configuration,
            _save_expressions(expressions, db_session),
            db_session,
        )
        params_to_save = {}
        for param_name, param_info in configuration.items():
            param_id = db_session.scalar(
                select(Parameter.id).where(Parameter.name == param_name)
            )
            for condition_info in param_info:
                found = False
                for old_config in compare_config:
                    if (
                        old_config.parameter_id == param_id
                        and (
                            old_config.condition_id
                            == condition_info["condition_id"]
                        )
                        and old_config.value == condition_info["value"]
                    ):
                        found = True
                        compare_config.remove(old_config)
                        break
                if not found:
                    params_to_save[param_name] = param_id
        for old_config in compare_config:
            if old_config.parameter.name not in configuration:
                continue
            params_to_save[old_config.parameter.name] = old_config.parameter_id
        for param_name, param_info in configuration.items():
            if param_name in params_to_save:
                parameter_id = params_to_save[param_name]
                # False positive
                # pylint: disable=no-member
                delete_statement = (
                    delete(Configuration)
                    .where(Configuration.parameter_id == parameter_id)
                    .where(Configuration.version == version)
                )
                # pylint: enable=no-member
                db_session.execute(delete_statement)
                db_session.add_all(
                    # False positive
                    # pylint: disable=not-callable
                    Configuration(
                        parameter_id=parameter_id,
                        condition_id=condition_info["condition_id"],
                        value=condition_info["value"],
                        version=version,
                    )
                    # pylint: enable=not-callable
                    for condition_info in param_info
                )


def list_steps():
    """List the pipeline steps."""

    with start_db_session() as db_session:
        return db_session.scalars(select(Step.name)).all()


def add_camera_type_channels(camera_type_id, properties, db_session):
    """
    Add channels to the given camera type and return partial channel entries.

    Args:
        camera_type_id(int):    The ID of the camera type to which to add
            channels.

        properties(dict-like):    The information being changed for the survey.
            For each channel to add there should be exactly two keywords:
            ``"channel-{channel_id}-name"`` and
            ``"channel-{channel_id}-slice"``. Where ``{channel_id}`` should be
            either an int specifying the identifier of the channel in the
            database or ``"new"`` specifying a new channel to add.
            ``{channel_id}``entries should be unique (for example only one new
            channel can be added). Channel slices have the format:
            ``"{x_offset}:{x_step};{y_offset}:{y_step}"``. Anything not related
            to channels is ignored.

        db_session:    The database session to use for updating.

    Returns:
        int or None, str or None:
            The channel ID and property (one of ``"name"`` or ``"slice"``)
            which is not fully specified or is mal-formatted. If more than one,
            the one wit the lowest ID is returned. If the new channel is
            unspecified, the channel returned is ``None``. If everything is
            fully specified ``None, None`` is returned.
    """

    def get_channel_info():
        """From the inputs extract the information to add to the database."""

        result = {}
        for key in properties:
            if key.startswith("channel-"):
                channel_id, channel_property = key.rsplit("-")[1:]
                assert channel_property in [
                    "name",
                    "slice",
                ], f"Unrecognized channel property {key}"
                if channel_id != "new":
                    channel_id = int(channel_id)
                if channel_id not in result:
                    result[channel_id] = {}
                if channel_property == "name":
                    assert "name" not in result[channel_id], (
                        "Duplicate name entry encountered for channel ID "
                        f"{channel_id}"
                    )
                    result[channel_id]["name"] = properties[key]
                else:
                    try:
                        values = sum(
                            (
                                dir_slice.split(":")
                                for dir_slice in properties[key].split(";")
                            ),
                            [],
                        )
                        values = [int(v) for v in values]
                        for attr, val in zip(
                            ["x_offset", "x_step", "y_offset", "y_step"], values
                        ):
                            assert attr not in result[channel_id], (
                                "Duplicate slice entry encountered for channel "
                                f"ID {channel_id}"
                            )
                            result[channel_id][attr] = val
                    except ValueError:
                        print_exc()
        return result

    def remove_unspecified(channel_info):
        """Leave only fully specified channels in update info, return result."""

        edit_id = None
        edit_property = None
        to_delete = set()
        required_attributes = get_editable_attributes(
            provenance.CameraChannel  # pylint: disable=no-member
        )
        required_attributes.remove("type")

        for channel_id, channel_attrs in channel_info.items():
            for attr in required_attributes:
                if attr not in channel_attrs:
                    print(
                        f"Attribute {attr} mising. "
                        f"Deleting channel {channel_id}."
                    )
                    if edit_id is None or edit_id > channel_id:
                        edit_id = channel_id
                        edit_property = "name" if attr == "name" else "slice"
                    to_delete.add(channel_id)
        for channel_id in to_delete:
            del channel_info[channel_id]
        return edit_id, edit_property

    channel_info = get_channel_info()
    result = remove_unspecified(channel_info)
    if channel_info:
        assert (
            camera_type_id >= 0
        ), "Attempting to set channels of non-existant camera type"

    for channel_id, channel_properties in channel_info.items():
        if channel_id == "new":
            db_channel = provenance.CameraChannel(  # pylint: disable=no-member
                camera_type_id=camera_type_id, **channel_properties
            )
        else:
            db_channel = db_session.scalar(
                select(
                    provenance.CameraChannel  # pylint: disable=no-member
                ).filter_by(id=channel_id, camera_type_id=camera_type_id)
            )
            for attr, value in channel_properties.items():
                setattr(db_channel, attr, value)

        if channel_id == "new":
            db_session.add(db_channel)
    return result


def get_editable_attributes(db_class):
    """List the user-editable attributes for the given component DB class."""

    def sort_key(colname):
        """Define the order in which attributes should be displayed."""

        if colname in ["name", "serial_number"]:
            return 0
        if colname == "type":
            return 1
        if colname == "notes":
            return 3
        return 2

    columns = [
        str(a).split(".", 1)[1]
        for a in inspect(db_class).attrs
        if isinstance(a, ColumnProperty)
    ]
    result = [
        "type" if col_name.endswith("_type_id") else col_name
        for col_name in columns
        if col_name not in ["id", "timestamp"]
    ]
    if "type" in result:
        result.remove("type")
        result.append("type")
    if db_class == provenance.CameraType:  # pylint: disable=no-member
        result.append("channels")
    return sorted(result, key=sort_key)


def get_human_name(column_name):
    """Return human friendly name for the given column."""

    if column_name == "serial_number":
        return "serial no"
    if column_name == "f_ratio":
        return "focal ratio"
    if column_name.endswith("_type_id"):
        return "type"
    return column_name.replace("_", " ")


def update_db_entry(
    db_session, properties, db_class, entry_id, component_type=None
):
    """
    Add/update a survey component or type, return its ID and what to autofocus.
    """

    incomplete = None
    entry_id = int(entry_id)
    if entry_id < 0:
        db_item = db_class()
    else:
        db_item = db_session.scalar(
            select(db_class).where(db_class.id == entry_id)
        )

    attribute_names = get_editable_attributes(db_class)
    for attr in attribute_names:
        if attr == "channels":
            assert (
                db_class == provenance.CameraType  # pylint: disable=no-member
            ), (
                f"Attempting to set channels for {db_class} (not a camera "
                "type)!"
            )
            channel_incomplete = add_camera_type_channels(
                entry_id, properties, db_session
            )
            if (
                channel_incomplete[0] is not None
                or channel_incomplete[1] is not None
            ):
                incomplete = {"channel": channel_incomplete}
        elif attr != "type":
            setattr(db_item, attr, properties[get_human_name(attr)])

    if "type" in attribute_names:
        type_id = int(properties.get("type-id"))
        if type_id < 0:
            raise ConfigurationError(
                f"No {component_type} type selected (got type ID "
                f"{type_id})!"
            )
        setattr(db_item, component_type + "_type_id", type_id)

    if entry_id < 0:
        db_session.add(db_item)
    db_session.flush()
    return db_item.id, incomplete


def import_json_to_survey(json_file):
    """Add to the survey configuration from given JSON encoding string."""

    def add_equipment_type(type_class, item_class, type_properties):
        """Add a single equipment type and all its devices."""

        type_id, incomplete = update_db_entry(
            db_session, type_properties, type_class, -1
        )
        if incomplete:
            return incomplete

        for component in type_properties["devices"]:
            component["type-id"] = type_id
            update_db_entry(
                db_session,
                component,
                item_class,
                -1,
                item_class.__tablename__,
            )
        if type_class == provenance.CameraType:  # pylint: disable=no-member
            for channel_name, channel_config in type_properties[
                "channels"
            ].items():
                channel_config["name"] = channel_name
                channel_config["type-id"] = type_id
                incomplete = update_db_entry(
                    db_session,
                    channel_config,
                    provenance.CameraChannel,  # pylint: disable=no-member
                    -1,
                    "camera",
                )[1]
                if incomplete:
                    return incomplete
        return None

    config = json.load(json_file)
    assert isinstance(
        config, dict
    ), "Malformatted JSON file encountered during import"

    with start_db_session() as db_session:
        for key, value in config.items():
            key = key.title()
            assert key.endswith(
                "s"
            ), f"Survey class {key} does not end with 's'."
            if key in ["Observers", "Observatories"]:
                db_class = (
                    provenance.Observer  # pylint: disable=no-member
                    if key == "Observers"
                    else provenance.Observatory  # pylint: disable=no-member
                )
                for properties in value:
                    incomplete = update_db_entry(
                        db_session, properties, db_class, -1
                    )[1]
            else:
                component_type = key[:-1]
                for type_properties in value:
                    incomplete = add_equipment_type(
                        getattr(provenance, component_type + "Type"),
                        getattr(provenance, component_type),
                        type_properties,
                    )
            assert incomplete is None, (
                "Mal-formatted or not fully specified configuration for "
                f"{key}: {value!r}"
            )


def master_config_json_to_settings(json_obj):
    """Translate a ``master_config.json`` structure into master settings.

    The JSON has the shape
    ``{master_type: {"enabled": ..., "split": [...], "match": [...]}, ...}``,
    where ``master_type`` is ``"zero"``, ``"dark"``, ``"flat"``, etc. The
    returned mapping uses ``"master-<type>-<param>"`` keys for every
    ``(type, param)`` pair iterated over by the BUI -- ``master_info`` is
    the canonical iteration order, with both ``highflat`` and ``lowflat``
    collapsed onto ``flat`` so the keys produced here match what the BUI's
    session storage uses.

    Args:
        json_obj: The result of ``json.load(...)`` on a master-config file.

    Returns:
        dict[str, str | list[str]]: Suitable as ``master_settings`` for
        :func:`apply_master_config`.
    """

    master_settings = {}
    for master_type in database_defaults.master_info:
        if master_type in ("highflat", "lowflat"):
            master_type = "flat"
        for param in ("enabled", "split", "match"):
            master_settings[f"master-{master_type}-{param}"] = json_obj[
                master_type
            ][param]
    return master_settings


def apply_master_config(master_settings):
    """Apply user-supplied master settings to the default step / master info.

    This is the pure-Python core of what the BUI uses to take the user's
    chosen master configuration and produce the ``step_dependencies`` /
    ``master_info`` arguments for
    :func:`autowisp.database.initialize_database.initialize_database`.

    Args:
        master_settings: Mapping keyed by ``"master-<type>-<param>"``,
            with values:

            * ``"<type>-enabled"``: ``"always"`` or an int-as-string
              (``"0"`` to disable, ``"1"`` to enable).
            * ``"<type>-split"``, ``"<type>-match"``: iterables of header
              expressions.

            Both plain ``dict`` and Django ``QueryDict`` are accepted; the
            list-valued keys are read via ``.getlist(...)`` when available
            and otherwise via ``mapping[key]``.

    Returns:
        tuple of (step_dependencies, master_info): deep copies of the
        ``database.defaults`` values with disabled masters pruned and the
        ``must_match`` / ``split_by`` sets replaced from ``master_settings``.
    """

    def get_list(key):
        if hasattr(master_settings, "getlist"):
            return master_settings.getlist(key)
        return master_settings[key]

    step_dependencies = copy.deepcopy(database_defaults.step_dependencies)
    master_info = copy.deepcopy(database_defaults.master_info)

    disabled_masters = [
        master_type
        for master_type in ("zero", "dark", "flat")
        if int(master_settings[f"master-{master_type}-enabled"]) == 0
    ]
    for i in range(len(step_dependencies) - 1, -1, -1):
        if step_dependencies[i][0] == "calibrate":
            if step_dependencies[i][1] in disabled_masters:
                del step_dependencies[i]
            else:
                assert step_dependencies[i][1] in (
                    "zero",
                    "dark",
                    "flat",
                    "object",
                ), (
                    "The built-in step dependencies have calibrate consuming "
                    f"an unknown master type: {step_dependencies[i][1]}!"
                )
                for master_type in disabled_masters:
                    try:
                        step_dependencies[i][2].remove(
                            (
                                "stack_to_master"
                                + ("_flat" if master_type == "flat" else ""),
                                master_type,
                            )
                        )
                    except ValueError:
                        pass
        elif (
            step_dependencies[i][0].startswith("stack_to_master")
            and step_dependencies[i][1] in disabled_masters
        ):
            del step_dependencies[i]

    for master_type in disabled_masters:
        if master_type == "flat":
            del master_info["highflat"]
            del master_info["lowflat"]
        else:
            del master_info[master_type]

    for master_type, master_config in master_info.items():
        if master_type in ("highflat", "lowflat"):
            master_type = "flat"
        enabled = master_settings[f"master-{master_type}-enabled"]
        assert enabled == "always" or int(enabled) == 1, (
            f"Master {master_type} is set to {enabled!r} yet survived the "
            "removal of the masters the user disabled!"
        )
        master_config["must_match"] = frozenset(
            filter(None, get_list(f"master-{master_type}-match"))
        )
        master_config["split_by"] = frozenset(
            filter(None, get_list(f"master-{master_type}-split"))
        )

    return step_dependencies, master_info


def parse_config_overwrites(lines):
    """Parse plain-text configuration lines into an overwrites dict.

    Each non-blank, non-comment, non-section line is parsed as ``key`` or
    ``key = value`` (``:`` is also accepted as the separator). Trailing
    ``#`` comments are stripped. ``;`` is *not* a comment marker --
    semicolons in unquoted values are preserved as data. Quoted values
    keep their inner content (the surrounding quotes are removed).
    Section headers (``[name]``) and blank lines are skipped.

    Keys that are never valid as DB parameter overrides are silently
    dropped, mirroring what the BUI's
    ``static/home/js/create.project.js`` does before populating the
    textarea: ``project-home`` (the project directory itself) and
    ``split-channels`` (generated from camera information). The list is
    a subset of what
    :class:`autowisp.database.initialize_database.StepCreator` already
    refuses to insert as a Parameter row.

    Args:
        lines: Iterable of strings, one per config line. Trailing
            newlines are tolerated.

    Returns:
        dict[str, list[tuple[None, str | None]]]:
            Suitable for the ``overwrite_default_config`` argument of
            :func:`autowisp.database.initialize_database.initialize_database`.
            Keys without a value are stored as ``None``.
    """

    line_rex = re.compile(
        r"^\s*(?P<key>[A-Za-z_][\w\-.]*)\s*"
        r"(?:[:=]\s*"
        r"(?:(?P<q>['\"])(?P<quoted>.*?)(?P=q)|(?P<value>[^#\n]*?))"
        r"\s*)?"
        r"(?:#.*)?$"
    )
    non_parameter_keys = frozenset({"project-home", "split-channels"})
    overwrites = {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "[")):
            continue
        match = line_rex.match(raw)
        if match is None:
            continue
        key = match.group("key")
        if key in non_parameter_keys:
            continue
        value = match.group("quoted")
        if value is None:
            value = match.group("value")
            if value is not None:
                value = value.strip() or None
        overwrites[key] = [(None, value)]
    return overwrites


def plural(word):
    """Mostly working pluralization."""

    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


def export_survey_to_json(destination, **limit_to):
    """Create JSON file storing selected survey information."""

    def get_export_objects(equipment_class, db_type=None):
        """Return list of DB types of equipment of given class to export."""

        is_type = (
            equipment_class not in ["Observer", "Observatory"]
            and db_type is None
        )
        export_limit = limit_to.get(
            equipment_class.lower() + ("_type" if is_type else ""),
            "all",
        )
        if export_limit == "none":
            return []

        db_class = getattr(
            provenance, equipment_class + ("Type" if is_type else "")
        )
        export_select = select(db_class)
        if db_type is not None:
            export_select = export_select.filter_by(
                **{f"{equipment_class.lower()}_type_id": db_type.id}
            )
        if export_limit != "all":
            export_select = export_select.where(db_class.id.in_(export_limit))
        return db_session.scalars(export_select).all()

    def get_config(equipment_class, db_class):
        """Return the configuration of the given class, including children."""

        result = db_class.to_dict()
        if equipment_class not in ["Observer", "Observatory"]:
            result["devices"] = [
                device.to_dict()
                for device in get_export_objects(equipment_class, db_class)
            ]
        return result

    config = {}
    with start_db_session() as db_session:
        for equipment_class in [
            "Camera",
            "Telescope",
            "Mount",
            "Observer",
            "Observatory",
        ]:
            export_list = get_export_objects(equipment_class)
            config[plural(equipment_class)] = [
                get_config(equipment_class, export) for export in export_list
            ]
    json.dump(config, destination, indent=4)


def parse_command_line():
    """Return the parsed command line configuration."""

    parser = ArgumentParser(
        description="Import or export survey configuration to/from JSON.",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "project_home",
        help="Path to the project home directory.",
    )
    parser.add_argument(
        "action",
        choices=["import", "export"],
        help="Whether to import JSON into the survey database or export "
        "survey data to JSON.",
    )
    parser.add_argument(
        "--filename",
        "-f",
        default=None,
        help="File to read from (import) or write to (export). "
        "Defaults to stdin for import and stdout for export.",
    )
    return parser.parse_args()


def main():
    """Run the import or export action specified on the command line."""

    config = parse_command_line()
    set_project_home(config.project_home)
    if config.action == "import":
        if config.filename is None:
            import_json_to_survey(sys.stdin)
        else:
            with open(config.filename, "r", encoding="utf-8") as json_file:
                import_json_to_survey(json_file)
    else:
        if config.filename is None:
            export_survey_to_json(sys.stdout)
        else:
            with open(config.filename, "w", encoding="utf-8") as json_file:
                export_survey_to_json(json_file)


if __name__ == "__main__":
    main()
