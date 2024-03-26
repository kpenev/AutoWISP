"""The views to display and edit pipeline configuration."""

from collections import namedtuple

from sqlalchemy import select, func, inspect
from sqlalchemy.orm import ColumnProperty
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from superphot_pipeline.database.user_interface import\
    get_json_config,\
    save_json_config,\
    list_steps
from superphot_pipeline.database.interface import Session
#False positive
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    Configuration,\
    ImageProcessingProgress
from superphot_pipeline.database.data_model import provenance
#pylint: enable=no-name-in-module


def config_tree(request, version=0, step='All', force_unlock=False):
    """Landing page for the configuration interface."""

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        defined_versions = sorted(
            db_session.scalars(
                select(
                    func.distinct(Configuration.version)
                )
            ).all()
        )
        max_used_version = db_session.scalar(
            select(
                func.max(ImageProcessingProgress.configuration_version)
            )
        )
        if max_used_version is None:
            max_used_version = -1

    return render(
        request,
        'configuration/config_tree.html',
        {
            'selected_step': step,
            'selected_version': version,
            'config_json': get_json_config(version, step=step, indent=4),
            'pipeline_steps': ['All'] + list_steps(),
            'config_versions': defined_versions,
            'max_locked_version': max_used_version,
            'locked': (not force_unlock) and version <= max_used_version
        }
    )


def save_config(request, version):
    """Save a user-defined configuration to the database."""

    save_json_config(request.body, version)
    return HttpResponseRedirect(reverse("configuration:config_tree"))


def edit_survey(request,
                selected_component_type=None,
                selected_id=None,
                create_new_types=''):
    """
    Add/delete instruments/observers to the currently configured survey.

    Args:
        request:    See django.

        selected_component_type(str):    What type of survey component is
            currently selected. One of ``'observer'``, ``'observatory'``,
            ``'camera'``, ``'mount'``, ``'telescope'``

        selected_id(int):    The ID of the selected component within the
            corresponding database table.

        create_new_types([str]):    Which of the equipment types (camera,
        telesceope, mount) do we want to create a new type for.
    """

    create_new_types = create_new_types.strip().split()

    context = {
        'selected_component': selected_component_type,
        'selected_id': selected_id,
        'attributes': {
            'camera': ['serial', 'notes', 'type'],
            'telescope': ['serial', 'notes', 'type'],
            'mount': ['serial', 'notes', 'type'],
            'observer': ['name', 'email', 'phone', 'notes'],
            'observatory': ['name',
                            'latitude',
                            'longitude',
                            'altitude',
                            'notes']
        },
        'types': {},
        'type_attributes': {},
        'create_new_types': create_new_types or []
    }
    selected = None
    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        if selected_component_type is not None:
            assert selected_id is not None
            selected_component_type = getattr(provenance,
                                              selected_component_type.title())
            selected = db_session.scalar(
                select(
                    selected_component_type
                ).where(
                    selected_component_type.id == selected_id
                )
            )
        for component_type in ['camera', 'mount', 'telescope']:
            context[component_type + 's'] = [
                namedtuple(
                    component_type,
                    ['id', 'str', 'serial', 'make', 'model', 'access', 'type']
                )(
                    equipment.id,
                    str(equipment),
                    equipment.serial_number,
                    getattr(equipment, component_type + '_type').make,
                    getattr(equipment, component_type + '_type').model,
                    equipment in getattr(selected, component_type + 's', []),
                    component_type
                )
                for equipment in db_session.scalars(
                    select(
                        getattr(provenance, component_type.title())
                    )
                ).all()
            ]
            db_type_class = getattr(provenance, component_type.title() + 'Type')
            context['type_attributes'][component_type] = [
                str(a).split('.', 1)[1]
                for a in inspect(db_type_class).attrs
                if (
                    isinstance(a, ColumnProperty)
                    and
                    not str(a).endswith('.timestamp')
                )
            ]
            context['types'][component_type] = [
                namedtuple(
                    component_type + '_type',
                    context['type_attributes'][component_type]
                )(
                    *[
                        getattr(db_type, attr)
                        for attr in context['type_attributes'][component_type]
                    ]
                )
                for db_type in db_session.scalars(select(db_type_class)).all()
            ]
        context['observers'] = [
            namedtuple(
                'observer',
                [
                    'id',
                    'str',
                    'name',
                    'email',
                    'phone',
                    'notes',
                    'access',
                    'type',
                ]
            )(
                obs.id,
                str(obs),
                obs.name,
                obs.email,
                obs.phone,
                obs.notes,
                obs in getattr(selected, 'observers', []),
                'observer',
            )
            for obs in db_session.scalars(select(provenance.Observer)).all()
        ]
        context['observatories'] = [
            namedtuple(
                'observatory',
                [
                    'id',
                    'str',
                    'name',
                    'latitude',
                    'longitude',
                    'altitude',
                    'type'
                ]
            )(
                obs.id,
                str(obs),
                obs.name,
                obs.latitude,
                obs.longitude,
                obs.altitude,
                'observatory'
            )
            for obs in db_session.scalars(select(provenance.Observatory)).all()
        ]
        print(repr(context))

    return render(
        request,
        'configuration/edit_survey.html',
        context
    )

