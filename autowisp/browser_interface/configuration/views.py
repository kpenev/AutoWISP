"""The views to display and edit pipeline configuration."""

from collections import namedtuple
from traceback import print_exc
import json

from sqlalchemy import select, func, inspect, delete
from sqlalchemy.orm import ColumnProperty
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from autowisp.database.user_interface import (
    get_json_config,
    save_json_config,
    list_steps,
)
from autowisp.database.interface import start_db_session, set_sqlite_database

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    ImageProcessingProgress,
    ObservingSession,
)
from autowisp.database.data_model import provenance

# pylint: enable=no-name-in-module


def config_tree(request, version=0, step="All", force_unlock=False):
    """Landing page for the configuration interface."""

    set_sqlite_database(request.session["project_db_path"])

    with start_db_session() as db_session:
        defined_versions = sorted(
            db_session.scalars(
                select(func.distinct(Configuration.version))
            ).all()
        )
        max_used_version = db_session.scalar(
            select(func.max(ImageProcessingProgress.configuration_version))
        )
        if max_used_version is None:
            max_used_version = -1

    return render(
        request,
        "configuration/config_tree.html",
        {
            "selected_step": step,
            "selected_version": version,
            "config_json": get_json_config(version, step=step, indent=4),
            "pipeline_steps": ["All"] + list_steps(),
            "config_versions": defined_versions,
            "max_locked_version": max_used_version,
            "locked": (not force_unlock) and version <= max_used_version,
        },
    )


def save_config(request, version):
    """Save a user-defined configuration to the database."""

    set_sqlite_database(request.session["project_db_path"])
    save_json_config(request.body, version)
    return HttpResponseRedirect(reverse("configuration:config_tree"))


def get_human_name(column_name):
    """Return human friendly name for the given column."""

    if column_name == "serial_number":
        return "serial no"
    if column_name == "f_ratio":
        return "focal ratio"
    if column_name.endswith("_type_id"):
        return "type"
    return column_name.replace("_", " ")


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


def format_channel_attr(camera_type):
    """Format the channel information for camera type for render context."""

    return [
        (
            channel.id,
            channel.name,
            f"{channel.x_offset}:{channel.x_step}"
            f";{channel.y_offset}:{channel.y_step}",
        )
        for channel in camera_type.channels
    ]


def add_survey_items_to_context(context, selected, db_session):
    """Add the current survey configuration to the given context."""

    def get_data(db_class):
        """Return the necessary information for the given survey component."""

        return db_session.execute(
            select(
                db_class,
                func.count(ObservingSession.id),  # pylint: disable=not-callable
            )
            .join(ObservingSession, isouter=True)
            .group_by(db_class.id)
        ).all()

    for component_class in ["camera", "mount", "telescope"]:

        attributes = get_editable_attributes(
            getattr(provenance, component_class.title())
        )

        tuple_type = namedtuple(
            component_class,
            attributes
            + [
                "id",
                "str",
                "access",
                "type_id",
                "component_class",
                "can_delete",
            ],
        )

        context[component_class + "s"] = []
        for equipment, has_data in get_data(
            getattr(provenance, component_class.title())
        ):
            equipment_type = getattr(equipment, component_class + "_type")
            context[component_class + "s"].append(
                tuple_type(
                    *(
                        getattr(
                            equipment,
                            attr,
                            getattr(
                                equipment_type,
                                attr,
                                (
                                    equipment_type.make
                                    + " "
                                    + equipment_type.model
                                    if attr == "type"
                                    else None
                                ),
                            ),
                        )
                        for attr in attributes
                    ),
                    equipment.id,
                    "S/N: " + equipment.serial_number,
                    equipment in getattr(selected, component_class + "s", []),
                    getattr(equipment, component_class + "_type_id"),
                    component_class,
                    not has_data,
                )
            )
        context[component_class + "s"].append(
            tuple_type(
                *(len(attributes) * ("",)),
                -1,
                "Add new " + component_class,
                False,
                1,
                component_class,
                True,
            )
        )

        db_type_class = getattr(provenance, component_class.title() + "Type")
        type_attributes = get_editable_attributes(db_type_class)
        context["type_attributes"][component_class] = [
            (get_human_name(col_name), col_name) for col_name in type_attributes
        ]
        type_attributes.append("id")
        type_attributes.append("can_delete")

        context["types"][component_class] = []
        for db_type in db_session.scalars(select(db_type_class)).all():
            can_delete = not getattr(db_type, component_class + "s")
            context["types"][component_class].append(
                namedtuple(component_class + "_type", type_attributes)(
                    *[
                        (
                            format_channel_attr(db_type)
                            if attr == "channels"
                            else getattr(db_type, attr, can_delete)
                        )
                        for attr in type_attributes
                    ]
                )
            )
        context["types"][component_class].append(
            namedtuple(component_class + "_type", type_attributes)(
                *[-1 if attr == "id" else "" for attr in type_attributes]
            )
        )

    tuple_type = namedtuple(
        "observer",
        [
            "id",
            "str",
            "name",
            "email",
            "phone",
            "notes",
            "access",
            "type",
            "can_delete",
        ],
    )
    context["observers"] = [
        tuple_type(
            obs.id,
            obs.name,
            obs.name,
            obs.email,
            obs.phone,
            obs.notes,
            obs in getattr(selected, "observers", []),
            "observer",
            not has_data,
        )
        for obs, has_data in get_data(
            provenance.Observer  # pylint: disable=no-member
        )
    ]
    context["observers"].append(
        tuple_type(-1, "Add new observer", *(5 * ("",)), "observer", True)
    )

    tuple_type = namedtuple(
        "observatory",
        [
            "id",
            "str",
            "name",
            "latitude",
            "longitude",
            "altitude",
            "type",
            "can_delete",
        ],
    )
    context["observatories"] = [
        tuple_type(
            obs.id,
            obs.name,
            obs.name,
            obs.latitude,
            obs.longitude,
            obs.altitude,
            "observatory",
            not has_data,
        )
        for obs, has_data in get_data(
            provenance.Observatory  # pylint: disable=no-member
        )
    ]
    context["observatories"].append(
        tuple_type(-1, "Add new observatory", *(4 * ("",)), "observatory", True)
    )


def edit_survey(
    request,
    *,
    selected_component=None,
    selected_id=None,
    selected_type_id=None,
    create_new_types="",
):
    """
    Add/delete instruments/observers to the currently configured survey.

    Args:
        request:    See django.

        selected_component(str):    What type of survey component is
            currently selected. One of ``'observer'``, ``'observatory'``,
            ``'camera'``, ``'mount'``, ``'telescope'``

        selected_id(str):    The ID of the selected component within the
            corresponding database table (should be convertable to int).

        create_new_types([str]):    Which of the equipment types (camera,
        telesceope, mount) do we want to create a new type for.
    """

    create_new_types = create_new_types.strip().split()
    if selected_id:
        selected_id = int(selected_id)
        assert selected_type_id is None
    else:
        selected_id = None

    selected = None
    set_sqlite_database(request.session["project_db_path"])
    with start_db_session() as db_session:

        if selected_component is not None and selected_type_id is None:
            assert selected_id is not None
            selected_component_type = getattr(
                provenance, selected_component.title()
            )
            selected = db_session.scalar(
                select(selected_component_type).where(
                    selected_component_type.id == selected_id
                )
            )

        context = {
            "selected_component": selected_component,
            "selected_id": selected_id,
            "selected_type_id": (
                int(selected_type_id) if selected_type_id else None
            ),
            "attributes": {
                component: [
                    (get_human_name(col_name), col_name)
                    for col_name in get_editable_attributes(
                        getattr(provenance, component.title())
                    )
                ]
                for component in [
                    "camera",
                    "telescope",
                    "mount",
                    "observatory",
                    "observer",
                ]
            },
            "types": {},
            "type_attributes": {},
            "create_new_types": create_new_types or [],
        }

        add_survey_items_to_context(context, selected, db_session)
        print(repr(context))

    return render(request, "configuration/edit_survey.html", context)


def delete_from_survey(
    request, component_type, component_id=None, component_type_id=None
):
    """Deleta a component of the survey network."""

    assert component_id or component_type_id
    assert component_id is None or component_type_id is None
    db_class = getattr(
        provenance,
        component_type.title() + ("Type" if component_id is None else ""),
    )
    set_sqlite_database(request.session["project_db_path"])
    with start_db_session() as db_session:
        if db_class == provenance.CameraType:  # pylint: disable=no-member
            db_type = db_session.scalar(
                select(db_class).filter_by(id=component_type_id)
            )
            for channel in db_type.channels:
                db_session.delete(channel)
        db_session.execute(
            delete(db_class).where(
                db_class.id == (component_id or component_type_id)
            )
        )
    return HttpResponseRedirect(reverse("configuration:survey"))


def add_camera_type_channel(camera_type_id, properties, db_session):
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
    print(80 * "*")
    print(f"Channel info: {channel_info!r}")
    result = remove_unspecified(channel_info)
    print(f"Cleaned channel info: {channel_info!r}")
    print(f"Result: {result!r}")
    if channel_info:
        assert (
            camera_type_id >= 0
        ), "Attempting to set channels of non-existant camera type"

    for channel_id, channel_properties in channel_info.items():
        print(f"Editing channel {channel_id} per: {channel_properties!r}")
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
    print(80 * "*")
    return result


def update_db_entry(
    db_session, properties, db_class, entry_id, component_type=None
):
    """
    Add/update a survey component or type, return its ID and what to autofocus.
    """

    print(80 * "*")
    print(repr(properties))
    print(80 * "*")

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
            channel_incomplete = add_camera_type_channel(
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
        assert type_id >= 0
        setattr(db_item, component_type + "_type_id", type_id)

    if entry_id < 0:
        db_session.add(db_item)
    db_session.commit()
    return db_item.id, incomplete


def update_survey_component_type(request, component_type, type_id):
    """Add or update a survey component type."""

    set_sqlite_database(request.session["project_db_path"])

    with start_db_session() as db_session:
        type_id, incomplete = update_db_entry(
            db_session,
            request.POST,
            getattr(provenance, component_type.title() + "Type"),
            type_id,
        )

    return HttpResponseRedirect(
        reverse(
            "configuration:survey",
            kwargs=(
                {}
                if incomplete is None
                else {
                    "selected_type_id": type_id,
                    "selected_component": component_type.lower(),
                }
            ),
        )
    )


def update_survey_component(request, component_type, component_id):
    """Add new or edit a component of the survey network."""

    set_sqlite_database(request.session["project_db_path"])

    with start_db_session() as db_session:
        update_db_entry(
            db_session,
            request.POST,
            getattr(provenance, component_type.title()),
            component_id,
            component_type,
        )
    return HttpResponseRedirect(reverse("configuration:survey"))


def import_json_to_survey(json_file):
    """Add to the survey configuration from given JSON encoding string."""

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
                incomplete = update_db_entry(db_session, value, db_class, -1)[1]
            else:
                component_type = key[:-1]
                db_class = getattr(provenance, component_type + "Type")
                type_id, incomplete = update_db_entry(
                    db_session, value, db_class, -1
                )
                if incomplete:
                    break

                db_class = getattr(provenance, component_type)
                for component in value["devices"]:
                    component["type-id"] = type_id
                    update_db_entry(
                        db_session,
                        component,
                        db_class,
                        -1,
                        component_type.lower(),
                    )
                if component_type == "Camera":
                    for channel_name, channel_config in value["channels"]:
                        channel_config["name"] = channel_name
                        incomplete = update_db_entry(
                            db_session,
                            channel_config,
                            provenance.CameraChannel,  # pylint: disable=no-member
                            -1,
                            "camera",
                        )[1]
                        if incomplete:
                            break
            assert incomplete is None, (
                "Mal-formatted or not fully specified configuration for "
                f"{key}: {value!r}"
            )


def change_access(  # pylint: disable=too-many-positional-arguments too-many-arguments line-too-long
    request,
    new_access,
    selected_component,
    selected_id,
    target_component,
    target_id,
):
    """Change an observer's access to something."""

    if selected_component == "observer":
        observer_id = selected_id
        equipment_id = target_id
        equipment_column = target_component
        access_class = getattr(provenance, target_component.title() + "Access")
    else:
        observer_id = target_id
        equipment_id = selected_id
        equipment_column = selected_component
        access_class = getattr(
            provenance, selected_component.title() + "Access"
        )
    equipment_column += "_id"

    set_sqlite_database(request.session["project_db_path"])
    with start_db_session() as db_session:
        if new_access:
            db_session.add(
                access_class(
                    observer_id=observer_id, **{equipment_column: equipment_id}
                )
            )
        else:
            db_session.execute(
                delete(access_class)
                .where(access_class.observer_id == observer_id)
                .where(getattr(access_class, equipment_column) == equipment_id)
            )

    return HttpResponseRedirect(
        reverse(
            "configuration:survey",
            kwargs={
                "selected_component": selected_component,
                "selected_id": selected_id,
            },
        )
    )
