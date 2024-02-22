"""Define interface to the pipeline database."""

import json

from sqlalchemy import select
from sqlalchemy import sql

from superphot_pipeline.database.interface import Session
#False positive
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    Configuration,\
    Step
#pylint: enable=no-name-in-module

def get_db_configuration(version, db_session):
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

    return db_session.scalars(
        select(
            Configuration
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
    ).all()
    #pylint: enable=no-member


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

    def get_children(values, expression_order):
        """Return the sub-tree for the given expressions."""

        result = []
        child_values = {}
        sibling_values = {}
        for value, val_expressions in values.items():
            if not val_expressions:
                result.append(
                    {
                        'name': value,
                        'content': ''
                    }
                )
            elif expression_order[0] in val_expressions:
                child_values[value] = (val_expressions
                                       -
                                       set([expression_order[0]]))
            else:
                sibling_values[value] = val_expressions

        if child_values:
            result.append(
                {
                    'name': expression_order[0],
                    'content': '',
                    'children': get_children(child_values,
                                             expression_order[1:])
                }
            )
        if sibling_values:
            result.extend(get_children(sibling_values,
                                       expression_order[1:]))
        return result


    config_data = {
        'name': 'All' if step == 'All' else step,
        'content': '',
        'children': []
    }
    for param, param_info in _get_config_info(version, step).items():
        expression_order = [
            expr_count[0]
            for expr_count in sorted(
                param_info['expression_counts'].items(),
                key=lambda expr_count: expr_count[1],
                reverse=True
            )
        ]
        config_data['children'].append(
            {
                'name': param,
                'content': '',
                'children': get_children(param_info['values'], expression_order)
            }
        )
    return json.dumps(config_data, **dump_kwargs)


def list_steps(enabled_only=False):
    """List the pipeline steps (or only those from the processing sequence)."""

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        return db_session.scalars(
            select(Step.name)
        ).all()


if __name__ == '__main__':
    print(get_json_config(indent=4, sort_keys=True))
