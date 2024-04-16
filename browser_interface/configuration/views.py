"""The views to display and edit pipeline configuration."""

from collections import namedtuple

from sqlalchemy import select, func, inspect, delete
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
    ImageProcessingProgress,\
    ObservingSession
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


def get_human_name(column_name):
    """Return human friendly name for the given column."""

    if column_name == 'serial_number':
        return 'serial no'
    if column_name == 'f_ratio':
        return 'focal ratio'
    if column_name.endswith('_type_id'):
        return 'type'
    return column_name.replace('_', ' ')


def get_editable_attributes(db_class):
    """List the user-editable attributes for the given component DB class."""

    def sort_key(colname):
        """Define the order in which attributes should be displayed."""

        if colname in ['name', 'serial_number']:
            return 0
        if colname == 'type':
            return 1
        if colname == 'notes':
            return 3
        print('Column: ' + colname)
        return 2

    columns = [
        str(a).split('.', 1)[1]
        for a in inspect(db_class).attrs
        if isinstance(a, ColumnProperty)
    ]
    result = [
        'type' if col_name.endswith('_type_id') else col_name
        for col_name in columns if col_name not in ['id', 'timestamp']
    ]
    if 'type' in result:
        result.remove('type')
        result.append('type')
    return sorted(result, key=sort_key)


def add_survey_items_to_context(context, selected, db_session):
    """Add the current survey configuration to the given context."""

    def get_data(db_class):
        """Return the necessary information for the given survey component."""

        return db_session.execute(
            select(
                db_class,
                func.count(ObservingSession.id)
            ).join(
                ObservingSession,
                isouter=True
            ).group_by(
                db_class.id
            )
        ).all()

    for component_class in ['camera', 'mount', 'telescope']:

        attributes = get_editable_attributes(
            getattr(provenance, component_class.title())
        )

        tuple_type = namedtuple(
            component_class,
            attributes
            +
            ['id', 'str', 'access', 'type_id', 'component_class', 'can_delete']
        )

        context[component_class + 's'] = []
        for equipment, has_data in get_data(
                getattr(provenance, component_class.title())
        ):
            equipment_type = getattr(equipment, component_class + '_type')
            context[component_class + 's'].append(
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
                                    + ' ' +
                                    equipment_type.model
                                    if attr == 'type' else
                                    None
                                )
                            )
                        )
                        for attr in attributes
                    ),
                    equipment.id,
                    'S/N: ' + equipment.serial_number,
                    equipment in getattr(selected, component_class + 's', []),
                    getattr(equipment, component_class + '_type_id'),
                    component_class,
                    not has_data
                )
            )
        context[component_class + 's'].append(
            tuple_type(
                *(len(attributes) * ('',)),
                -1,
                'Add new ' + component_class,
                False,
                1,
                component_class,
                True
            )
        )

        db_type_class = getattr(provenance, component_class.title() + 'Type')
        type_attributes = get_editable_attributes(db_type_class)
        context['type_attributes'][component_class] = [
            get_human_name(col_name)
            for col_name in type_attributes
        ]
        type_attributes.append('id')
        context['types'][component_class] = [
            namedtuple(
                component_class + '_type',
                type_attributes
            )(
                *[
                    getattr(db_type, attr)
                    for attr in type_attributes
                ]
            )
            for db_type in db_session.scalars(select(db_type_class)).all()
        ]

    tuple_type = namedtuple(
        'observer',
        ['id', 'str', 'name', 'email', 'phone', 'notes', 'access', 'type',
         'can_delete']
    )
    context['observers'] = [
        tuple_type(
            obs.id,
            obs.name,
            obs.name,
            obs.email,
            obs.phone,
            obs.notes,
            obs in getattr(selected, 'observers', []),
            'observer',
            not has_data
        )
        for obs, has_data in get_data(provenance.Observer)
    ]
    context['observers'].append(
        tuple_type(-1, 'Add new observer', *(5 * ('',)), 'observer', True)
    )

    tuple_type = namedtuple(
        'observatory',
        ['id', 'str', 'name', 'latitude', 'longitude', 'altitude', 'type',
         'can_delete']
    )
    context['observatories'] = [
        tuple_type(
            obs.id,
            obs.name,
            obs.name,
            obs.latitude,
            obs.longitude,
            obs.altitude,
            'observatory',
            not has_data
        )
        for obs, has_data in get_data(provenance.Observatory)
    ]
    context['observatories'].append(
        tuple_type(-1, 'Add new observatory', *(4 * ('',)), 'observatory', True)
    )


def edit_survey(request,
                *,
                selected_component=None,
                selected_id=None,
                create_new_types=''):
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
    else:
        selected_id = None

    selected = None
    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member

        if selected_component is not None:
            assert selected_id is not None
            selected_component_type = getattr(provenance,
                                              selected_component.title())
            selected = db_session.scalar(
                select(
                    selected_component_type
                ).where(
                    selected_component_type.id == selected_id
                )
            )

        context = {
            'selected_component': selected_component,
            'selected_id': selected_id,
            'attributes': {
                component: [
                    (
                        get_human_name(col_name),
                        col_name
                    )
                    for col_name in get_editable_attributes(
                        getattr(provenance, component.title())
                    )
                ]
                for component in ['camera',
                                  'telescope',
                                  'mount',
                                  'observatory',
                                  'observer']
            },
            'types': {},
            'type_attributes': {},
            'create_new_types': create_new_types or [],
        }

        add_survey_items_to_context(context, selected, db_session)
        print(repr(context))

    return render(
        request,
        'configuration/edit_survey.html',
        context
    )

def update_survey_component(request, component_type, component_id):
    """Add a new component to the survey network."""

    component_title = component_type.title()
    component_id = int(component_id)
    print(repr(request.POST))
    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        db_class = getattr(provenance, component_title)
        if request.POST['todo'] == 'Add':
            assert component_id < 0
            db_item = db_class()
        elif request.POST['todo'] == 'Delete':
            assert component_id >= 0
            db_session.execute(
                delete(db_class).where(db_class.id == component_id)
            )
            return HttpResponseRedirect(reverse("configuration:survey"))
        else:
            assert component_id >= 0
            db_item = db_session.scalar(
                select(db_class).where(db_class.id == component_id)
            )

        attribute_names = get_editable_attributes(db_class)
        for attr in attribute_names:
            if attr != 'type':
                setattr(
                    db_item,
                    attr,
                    request.POST[component_type + '-' + get_human_name(attr)]
                )

        if 'type' in attribute_names:
            type_id = request.POST.get(component_type + '-type-id')
            if type_id is None:
                type_db_class = getattr(provenance, component_title + 'Type')
                type_attr_names = get_editable_attributes(type_db_class)
                new_type = type_db_class(**{
                    attr: request.POST[
                        component_type + '-type-' + get_human_name(attr)
                    ]
                    for attr in type_attr_names
                })
                db_session.add(new_type)
                db_session.flush()
                db_session.refresh(new_type, ['id'])
                type_id = new_type.id

            setattr(db_item, component_type + '_type_id', type_id)

        if component_id < 0:
            db_session.add(db_item)
            db_session.flush()
            db_session.refresh(db_item, ['id'])

        return HttpResponseRedirect(reverse(
            "configuration:survey",
            kwargs={'selected_component': component_type,
                    'selected_id': db_item.id}
        ))


def change_access(request,
                  new_access,
                  selected_component,
                  selected_id,
                  target_component,
                  target_id):
    """Change an observer's access to something."""

    if selected_component == 'observer':
        observer_id = selected_id
        equipment_id = target_id
        equipment_column = target_component
        access_class = getattr(provenance, target_component.title() + 'Access')
    else:
        observer_id = target_id
        equipment_id = selected_id
        equipment_column = selected_component
        access_class = getattr(provenance,
                               selected_component.title() + 'Access')
    equipment_column += '_id'

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        if new_access:
            db_session.add(
                access_class(observer_id=observer_id,
                             **{equipment_column: equipment_id})
            )
        else:
            db_session.execute(
                delete(access_class).where(
                    access_class.observer_id == observer_id
                ).where(
                    getattr(access_class, equipment_column) == equipment_id
                )
            )

    return HttpResponseRedirect(reverse(
            "configuration:survey",
            kwargs={'selected_component': selected_component,
                    'selected_id': selected_id}
    ))
