"""Define interface to the pipeline database."""

import json
import logging

from sqlalchemy import sql, select, func, and_

from superphot_pipeline.database.interface import Session
#False positive
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model.provenance import\
    Camera,\
    CameraType,\
    CameraChannel

from superphot_pipeline.database.data_model import\
    ObservingSession,\
    Configuration,\
    Step,\
    Image,\
    ImageType,\
    ProcessingSequence,\
    ProcessedImages,\
    ImageProcessingProgress,\
    step_param_association
#pylint: enable=no-name-in-module

def get_db_configuration(version,
                         db_session,
                         step_id=None,
                         max_version_only=False):
    """Return list of Configuration instances given version."""

    #False positives:
    #pylint: disable=no-member
    param_version_subq = select(
        Configuration.parameter_id,
        #False positivie
        #pylint: disable=not-callable
        sql.func.max(Configuration.version).label('version'),
        #pylint: enable=not-callable
    ).filter(
        Configuration.version <= version
    ).group_by(
        Configuration.parameter_id
    ).subquery()

    config_select = select(
        func.max(Configuration.version) if max_version_only else Configuration
    ).join(
        param_version_subq,
        sql.expression.and_(
            (
                Configuration.parameter_id
                ==
                param_version_subq.c.parameter_id
            ),
            (
                Configuration.version
                ==
                param_version_subq.c.version
            )
        )
    )
    if step_id is not None:
        config_select = config_select.join(
            step_param_association,
            Configuration.parameter_id == step_param_association.c.param_id
        ).where(
            step_param_association.c.step_id == step_id
        )

    if max_version_only:
        return db_session.scalars(config_select).one()
    return db_session.scalars(config_select).all()
    #pylint: enable=no-member


def get_processing_sequence(db_session):
    """Return the sequence of step/image type pairs to process."""

    return db_session.execute(
        select(
            Step,
            ImageType
        ).select_from(
            ProcessingSequence
        ).join(
            Step,
            ProcessingSequence.step_id == Step.id
        ).join(
            ImageType,
            ProcessingSequence.image_type_id == ImageType.id
        )
    ).all()


def list_channels(db_session):
    """List the combine set of channels for all cameras."""

    return db_session.scalars(func.distinct(CameraChannel.name)).all()


def get_progress(step_id, image_type_id, config_version, db_session):
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
            entries are channel name, status, number of images of that channel
            that have that status which is flagged final.

        [str, int]:    Information about the images not in final state. The
            entries are channel name, number of non-final images of that
            channel.

        [str, int, int]:    The pending images broken by status. The format is
            the same as the final state information, except for images not
            flagged as in final state for the given step.
    """

    step_version = get_db_configuration(config_version,
                                        db_session,
                                        step_id,
                                        max_version_only=True)

    def complete_processed_select(_select):
        """Return the given select joined and filtered to given processed."""

        return _select.join(
            ImageProcessingProgress
        ).where(
            ImageProcessingProgress.step_id == step_id
        ).where(
            ImageProcessingProgress.configuration_version == step_version
        )

    select_image_channel = select(
        CameraChannel.name,
        #False poisitive
        #pylint: disable=not-callable
        func.count(Image.id)
        #pylint: enable=not-callable
    ).join(
        ObservingSession,
    ).join(
        Camera
    ).join(
        CameraType
    ).join(
        CameraChannel
    )

    processed_select = complete_processed_select(
        select(
            ProcessedImages.channel,
            ProcessedImages.status,
            #False poisitive
            #pylint: disable=not-callable
            func.count(ProcessedImages.image_id)
            #pylint: enable=not-callable
        ).join(
            Image
        ).join(
            ImageType
        )
    ).where(
        ImageType.id == image_type_id
    )
    final = db_session.execute(
        processed_select.where(
            ProcessedImages.final
        ).group_by(
            ProcessedImages.status,
            ProcessedImages.channel
        )
    ).all()
    by_status = db_session.execute(
        processed_select.where(
            ~ProcessedImages.final
        ).group_by(
            ProcessedImages.status,
            ProcessedImages.channel
        )
    ).all()
    processed_subquery = complete_processed_select(
        select(
            ProcessedImages.image_id,
            ProcessedImages.channel
        )
    ).where(
        ProcessedImages.final
    ).subquery()

    pending = db_session.execute(
        select_image_channel.outerjoin(
            processed_subquery,
            and_(Image.id == processed_subquery.c.image_id,
                 CameraChannel.name == processed_subquery.c.channel),
        ).where(
            #This is how NULL comparison is done in SQLAlchemy
            #pylint: disable=singleton-comparison
            processed_subquery.c.image_id == None
            #pylint: enable=singleton-comparison
        ).where(
            Image.image_type_id == image_type_id
        ).group_by(
            CameraChannel.name
        )
    ).all()
    return final, pending, by_status

    result = {}
    for channel, status, count in by_status:
        if channel not in result:
            result[channel] = [
                final.get(channel, (0, None)),
                pending.get(channel, 0),
                [
                    (status, count)
                ]
            ]
        else:
            result[channel][-1].append((status, count))
    for channel, num_pending in pending.items():
        if channel not in result:
            result[channel] = [(0, None), num_pending, []]
    return [
        (channel, *result[channel])
        for channel in sorted(result)
    ]


def _get_config_info(version, step='All'):
    """Return info for displaying the configuration with given version."""

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        if step != 'All':
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
                step != 'All'
                and
                config.parameter.id not in restrict_param_ids
            ):
                continue
            if config.parameter.name not in config_info:
                config_info[config.parameter.name] = {
                    'values': {},
                    'expression_counts': {}
                }
            param_info = config_info[config.parameter.name]
            param_info['values'][config.value] = set(
                expr.expression for expr in config.condition_expressions
                if expr.expression != 'True'
            )
            for expression in config.condition_expressions:
                param_info['expression_counts'][expression.expression] = (
                    param_info['expression_counts'].get(
                        expression.expression,
                        0
                    )
                    +
                    1
                )
    return config_info


def get_json_config(version=0, step='All', **dump_kwargs):
    """Return the configuration as a JSON object."""

    def get_children(values, expression_order, parent_id, first_child_id=0):
        """Return the sub-tree for the given expressions."""

        result = []
        child_values = {}
        sibling_values = {}
        child_id = first_child_id
        for value, val_expressions in values.items():
            if not val_expressions:
                result.append(
                    {
                        'name': value,
                        'type': 'value',
                        'id': f'{parent_id}.{child_id}',
                        'children': []
                    }
                )
                child_id += 1
            elif expression_order[0] in val_expressions:
                child_values[value] = (val_expressions
                                       -
                                       set([expression_order[0]]))
            else:
                sibling_values[value] = val_expressions

        if child_values:
            id_str = f'{parent_id}.{child_id}'
            child_id += 1
            result.append(
                {
                    'name': expression_order[0],
                    'type': 'condition',
                    'id': id_str,
                    'children': get_children(child_values,
                                             expression_order[1:],
                                             id_str)
                }
            )
        if sibling_values:
            result.extend(get_children(sibling_values,
                                       expression_order[1:],
                                       parent_id,
                                       child_id))
        return result


    config_data = {
        'name': 'All' if step == 'All' else step,
        'type': 'step',
        'id': '',
        'children': []
    }
    for sub_id, (param, param_info) in enumerate(
            _get_config_info(version, step).items()
    ):
        expression_order = [
            expr_count[0]
            for expr_count in sorted(
                param_info['expression_counts'].items(),
                key=lambda expr_count: expr_count[1],
                reverse=True
            )
        ]
        id_str = f'{sub_id}'
        config_data['children'].append(
            {
                'name': param,
                'type': 'parameter',
                'id': id_str,
                'children': get_children(param_info['values'],
                                         expression_order,
                                         id_str)
            }
        )
    return json.dumps(config_data, **dump_kwargs)


def list_steps():
    """List the pipeline steps."""

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        return db_session.scalars(
            select(Step.name)
        ).all()


def main():
    """Avoid polluting the global namespace."""

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        print('Channels: ' + repr(list_channels(db_session)))
        print(get_progress(8, 4, 0, db_session))

if __name__ == '__main__':
    main()
