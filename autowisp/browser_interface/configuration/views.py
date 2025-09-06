"""The views to display and edit pipeline configuration."""

from collections import namedtuple
from io import StringIO, BytesIO
import json

from sqlalchemy import select, func, delete
from django.shortcuts import render, redirect, HttpResponse
from django.http import FileResponse

from autowisp.database.user_interface import (
    get_json_config,
    save_json_config,
    list_steps,
    get_editable_attributes,
    update_db_entry,
    get_human_name,
    import_json_to_survey,
    export_survey_to_json,
)
from autowisp.database.interface import start_db_session

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    provenance,
    Configuration,
    ImageProcessingProgress,
    ObservingSession,
)

# pylint: enable=no-name-in-module


def config_tree(request, version=0, step="All", force_unlock=False):
    """Landing page for the configuration interface."""

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

    if request.method == "POST":
        config = request.FILES["import-config"].read().decode()
        force_unlock = True
    else:
        config = get_json_config(version, step=step, indent=4)

    return render(
        request,
        "configuration/config_tree.html",
        {
            "selected_step": step,
            "selected_version": version,
            "config_json": config,
            "pipeline_steps": ["All"] + list_steps(),
            "config_versions": defined_versions,
            "max_locked_version": max_used_version,
            "locked": (not force_unlock) and version <= max_used_version,
        },
    )


def save_config(request, version):
    """Save a user-defined configuration to the database."""

    save_json_config(request.body, version)
    return redirect("configuration:config_tree")


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
    return redirect("configuration:survey")


def update_survey_component_type(request, component_type, type_id):
    """Add or update a survey component type."""

    with start_db_session() as db_session:
        type_id, incomplete = update_db_entry(
            db_session,
            request.POST,
            getattr(provenance, component_type.title() + "Type"),
            type_id,
        )

    return redirect(
        "configuration:survey",
        **(
            {}
            if incomplete is None
            else {
                "selected_type_id": type_id,
                "selected_component": component_type.lower(),
            }
        ),
    )


def update_survey_component(request, component_type, component_id):
    """Add new or edit a component of the survey network."""

    with start_db_session() as db_session:
        update_db_entry(
            db_session,
            request.POST,
            getattr(provenance, component_type.title()),
            component_id,
            component_type,
        )
    return redirect("configuration:survey")


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

    return redirect(
        "configuration:survey",
        selected_component=selected_component,
        selected_id=selected_id,
    )


def import_survey_info(request):
    """Add survey equipment from JSON file."""

    assert request.method == "POST"
    import_json_to_survey(request.FILES["survey-import"])
    return redirect("configuration:survey")


def export_survey_info(_):
    """Save (some of) the survey equipment to a JSON file."""

    with StringIO() as export_stream:
        export_survey_to_json(export_stream)
        return HttpResponse(
            export_stream.getvalue().encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Content-Disposition": ('attachment; filename="survey.json"'),
            },
        )
