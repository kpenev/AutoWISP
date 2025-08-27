"""The views to display and edit pipeline configuration."""

from collections import namedtuple
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
    if db_class == provenance.CameraType: # pylint: disable=no-member
        result.append("channels")
    return sorted(result, key=sort_key)


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
                        getattr(db_type, attr, can_delete)
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
        db_session.execute(
            delete(db_class).where(
                db_class.id == (component_id or component_type_id)
            )
        )
    return HttpResponseRedirect(reverse("configuration:survey"))


def add_camera_type_channel(camera_type_id, properties, db_session):
    """Add to the given camera type all channels found in properties."""

    channel_info = {}
    for key in properties:
        if key.startswith("channel-"):
            channel_id, channel_property = key.rsplit("-")[1:]
            assert channel_property in [
                "name",
                "slice",
            ], f"Unrecognized channel property {key}"
            if channel_id != "new":
                channel_id = int(channel_id)
            if channel_id not in channel_info:
                channel_info[channel_id] = {}
            if channel_property == "name":
                assert "name" not in channel_info[channel_id], (
                    "Duplicate name entry encountered for channel ID "
                    f"{channel_id}"
                )
                channel_info[channel_id]["name"] = properties[key]
            else:
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
                    assert attr not in channel_info[channel_id], (
                        "Duplicate slice entry encountered for channel ID "
                        f"{channel_id}"
                    )
                    channel_info[channel_id][attr] = val

    for channel_id, channel_attrs in channel_info.items():
        if channel_id == "new":
            db_channel = provenance.CameraChannel()  # pylint: disable=no-member
        else:
            db_channel = db_session.scalar(
                select(
                    provenance.CameraChannel  # pylint: disable=no-member
                ).filter_by(id=channel_id)
            )
        db_channel.camera_type_id = camera_type_id
        for attr, val in channel_attrs.items():
            setattr(db_channel, attr, val)

        if channel_id == "new":
            db_session.add(db_channel)


def update_db_entry(properties, db_class, entry_id, component_type=None):
    """Add/update a survey component/component type and return its ID."""

    print(80 * "*")
    print(repr(properties))
    print(80 * "*")

    entry_id = int(entry_id)
    with start_db_session() as db_session:
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
                    db_class
                    == provenance.CameraType  # pylint: disable=no-member
                ), (
                    f"Attempting to set channels of {db_class} (not a camera "
                    "type)!"
                )
                assert (
                    entry_id >= 0
                ), "Attempting to set channels of non-existant camera"
                add_camera_type_channel(entry_id, properties, db_session)
            elif attr != "type":
                setattr(db_item, attr, properties[get_human_name(attr)])

        if "type" in attribute_names:
            type_id = int(properties.get("type-id"))
            assert type_id >= 0
            setattr(db_item, component_type + "_type_id", type_id)

        if entry_id < 0:
            db_session.add(db_item)
        db_session.commit()
        return db_item.id


def update_survey_component_type(request, component_type, type_id):
    """Add or update a survey component type."""

    set_sqlite_database(request.session["project_db_path"])
    update_db_entry(
        request.POST,
        getattr(provenance, component_type.title() + "Type"),
        type_id,
    )

    return HttpResponseRedirect(reverse("configuration:survey"))


def update_survey_component(request, component_type, component_id):
    """Add new or edit a component of the survey network."""

    set_sqlite_database(request.session["project_db_path"])
    update_db_entry(
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
    for key, value in config.items():
        key = key.title()
        assert key.endswith("s"), f"Survey class {key} does not end with 's'."
        if key in ["Observers", "Observatories"]:
            db_class = (
                provenance.Observer  # pylint: disable=no-member
                if key == "Observers"
                else provenance.Observatory  # pylint: disable=no-member
            )
            update_db_entry(value, db_class, -1)
        else:
            component_type = key[:-1]
            db_class = getattr(provenance, component_type + "Type")
            type_id = update_db_entry(value, db_class, -1)

            db_class = getattr(provenance, component_type)
            for component in value["devices"]:
                component["type-id"] = type_id
                update_db_entry(component, db_class, -1, component_type.lower())
            if component_type == "Camera":
                for channel_name, channel_config in value["channels"]:
                    channel_config["name"] = channel_name
                    update_db_entry(
                        channel_config,
                        provenance.CameraChannel,  # pylint: disable=no-member
                        -1,
                        "camera",
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
