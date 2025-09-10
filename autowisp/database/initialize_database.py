#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

import re
import logging

from configargparse import ArgumentParser, DefaultsFormatter
from sqlalchemy import sql, select, delete, MetaData, update

from autowisp.database.interface import (
    get_db_engine,
    start_db_session,
    initialize_cmdline_database,
)
from autowisp.database.data_model.base import DataModelBase
from autowisp.database import defaults

from autowisp import processing_steps

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    ImageType,
    Step,
    StepDependencies,
    Parameter,
    AlternateParameterName,
    Configuration,
    Condition,
    ConditionExpression,
    ProcessingSequence,
    MasterType,
    InputMasterTypes,
)

# pylint: enable=no-name-in-module

_logger = logging.getLogger(__name__)


def get_command_line_parser():
    """Create a parser with all required command line arguments."""

    parser = ArgumentParser(
        description="Initialize the database for first time use of the "
        "pipeline.",
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "--config-file",
        "-c",
        is_config_file=True,
        # default=config_file,
        help="Specify a configuration file in liu of using command line "
        "options. Any option can still be overriden on the command line.",
    )
    parser.add_argument(
        "--drop-all-tables",
        action="store_true",
        help="If passed all pipeline tables are deleted before new ones are "
        "created",
    )
    parser.add_argument(
        "--drop-hdf5-structure-tables",
        "--drop-structure",
        action="store_true",
        help="If passed, tables defining the structure of HDF5 files are "
        "dropped first and then re-created and filled. Otherwise, if tables "
        "exist, their contents is not modified.",
    )
    parser.add_argument(
        "--verbose",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the verbosity of the DB logger.",
    )
    return parser


# This is meant to function as callable
# pylint: disable=too-few-public-methods
class StepCreator:
    """Add steps to the database one by one."""

    def __init__(self):
        """Get ready to add steps to the database."""

        with start_db_session() as db_session:
            self._step_id = 1
            self._db_parameters = {}

            default_expression = ConditionExpression(
                id=1, expression="True", notes="Default expression"
            )
            db_session.add(default_expression)

            # False positive
            # pylint: disable=not-callable
            self._default_condition = Condition(
                id=1,
                expression_id=default_expression.id,
                notes="Default configuration",
            )
            # pylint: enable=not-callable

            db_session.add(self._default_condition)

    def __call__(self, step_name, db_session):
        """Add a step with the given name to the database."""

        self._default_condition = db_session.merge(self._default_condition)
        step_module = getattr(processing_steps, step_name)
        new_step = Step(
            id=self._step_id, name=step_name, description=step_module.__doc__
        )
        self._step_id += 1

        print(f"Initializing {step_name} parameters")
        default_step_config = step_module.parse_command_line([])
        print(
            "Default step config:\n\t"
            + "\n\t".join(
                f"{param}: {value}"
                for param, value in default_step_config[
                    "argument_defaults"
                ].items()
            )
        )
        for param in default_step_config["argument_descriptions"].keys():
            if (
                param
                not in [
                    "h",
                    "config-file",
                    "extra-config-file",
                    "split-channels",
                    "database-fname",
                ]
                and not param.endswith("-only-if")
                and not param.endswith("-version")
            ):
                description = default_step_config["argument_descriptions"][
                    param
                ]
                if isinstance(description, dict):
                    description = description["help"]
                    configuration = None
                if param not in self._db_parameters:
                    self._db_parameters[param] = Parameter(
                        name=param, description=description
                    )
                    print(
                        f"Setting {param} = "
                        f'{default_step_config["argument_defaults"][param]}'
                    )
                    # False positive
                    # pylint: disable=not-callable
                    configuration = Configuration(
                        version=0,
                        condition_id=self._default_condition.id,
                        value=default_step_config["argument_defaults"][param],
                    )
                    # pylint: enable=not-callable
                    configuration.parameter = self._db_parameters[param]

                    for alt_name in default_step_config[
                        "alternate_argument_names"
                    ][param]:
                        alt_name_entry = AlternateParameterName(
                            alt_name=alt_name
                        )
                        self._db_parameters[param].alternate_names.append(
                            alt_name_entry
                        )

                    db_session.add(configuration)

                new_step.parameters.append(self._db_parameters[param])

        db_session.add(new_step)
        return new_step


# pylint: enable=too-few-public-methods


def add_master_dependencies(db_session, master_info):
    """Fill the master_types table."""

    def get_imtype_id(imtype_name):
        """Return the ID of the image type with the given name."""

        return db_session.scalar(
            select(ImageType.id).where(ImageType.name == imtype_name)
        )

    def get_step_id(step_name):
        """Return the ID of the step with the given name."""

        return db_session.scalar(select(Step.id).where(Step.name == step_name))

    expressions = set()
    for master_config in master_info.values():
        expressions.update(
            master_config["must_match"], master_config["split_by"]
        )

    db_expressions = {
        expr: ConditionExpression(expression=expr) for expr in expressions
    }
    db_session.add_all(db_expressions.values())

    next_condition_id = db_session.scalar(
        # False positive
        # pylint: disable=no-member
        select(sql.functions.max(Condition.id) + 1)
        # pylint: enable=no-member
    )

    condition_ids = {}
    for master_type, master_config in master_info.items():
        print(f"Master {master_type} config: {master_config!r}")
        for condition in [
            master_config["must_match"],
            master_config["split_by"],
        ]:
            if condition and condition not in condition_ids:
                db_session.add_all(
                    [
                        # False positive
                        # pylint: disable=not-callable
                        Condition(
                            id=next_condition_id,
                            expression_id=db_expressions[expr].id,
                        )
                        # pylint: enable=not-callable
                        for expr in condition
                    ]
                )
                condition_ids[condition] = next_condition_id
                next_condition_id += 1
        db_master_type = MasterType(
            name=master_type,
            condition_id=condition_ids.get(master_config["must_match"], 1),
            maker_step_id=(
                None
                if master_config["created_by"] is None
                else get_step_id(master_config["created_by"][0])
            ),
            maker_image_type_id=(
                None
                if master_config["created_by"] is None
                else get_imtype_id(master_config["created_by"][1])
            ),
            maker_image_split_condition_id=(
                condition_ids.get(master_config["split_by"])
            ),
            description=master_config["description"],
        )
        db_session.add(db_master_type)
        for step, image_type, optional in master_config["used_by"]:
            db_session.add(
                InputMasterTypes(
                    step_id=get_step_id(step),
                    image_type_id=get_imtype_id(image_type),
                    master_type_id=db_master_type.id,
                    config_name=master_config["config_name"],
                    optional=optional,
                )
            )


# No good way to simplify
# pylint: disable=too-many-locals
def init_processing(step_dependencies, master_info):
    """Initialize the tables controlling how processing is to be done."""

    image_type_list = []
    for processing_order in step_dependencies:
        if (
            processing_order[1] is not None
            and processing_order[1] not in image_type_list
        ):
            image_type_list.append(processing_order[1])

    add_processing_step = StepCreator()

    with start_db_session() as db_session:
        for image_type_id, image_type in enumerate(image_type_list, 1):
            db_session.add(ImageType(id=image_type_id, name=image_type))
        db_steps = {}

        for processing_id, (step_name, image_type, dependencies) in enumerate(
            step_dependencies, 1
        ):
            if step_name not in db_steps:
                db_steps[step_name] = add_processing_step(step_name, db_session)
            if step_name not in ["add_images_to_db", "calculate_photref_merit"]:
                db_session.add(
                    ProcessingSequence(
                        id=processing_id,
                        step_id=db_steps[step_name].id,
                        image_type_id=(
                            None
                            if image_type is None
                            else image_type_list.index(image_type) + 1
                        ),
                    )
                )
            for required_step, required_imtype in dependencies:
                db_session.add(
                    StepDependencies(
                        blocked_step_id=db_steps[step_name].id,
                        blocked_image_type_id=image_type_list.index(image_type)
                        + 1,
                        blocking_step_id=db_steps[required_step].id,
                        blocking_image_type_id=image_type_list.index(
                            required_imtype
                        )
                        + 1,
                    )
                )
        add_master_dependencies(db_session, master_info)


# pylint: enable=too-many-locals


def drop_tables_matching(pattern):
    """Drop tables with names matching a pre-compiled regular expression."""

    if pattern is None:
        metadata = MetaData()
        metadata.reflect(get_db_engine())
        metadata.drop_all(get_db_engine())
    else:
        DataModelBase.metadata.drop_all(
            get_db_engine(),
            filter(
                lambda table: pattern.fullmatch(table.name),
                reversed(DataModelBase.metadata.sorted_tables),
            ),
        )


def _overwrite_default_config(new_default_config):
    """Overwrite default values from what is defined in the steps."""

    db_conditions = {}
    db_expressions = {}
    with start_db_session() as db_session:
        for param, all_values in new_default_config.items():
            alternate = db_session.execute(
                select(AlternateParameterName).filter_by(alt_name=param)
            ).scalar_one_or_none()
            if alternate is not None:
                param = alternate.parameter.name
            param_id = db_session.scalar(
                select(Parameter.id).filter_by(name=param)
            )
            assert param_id is not None, f"Parameter {param} not found in DB"
            delete_default = True
            for condition_expressions, value in all_values:
                if condition_expressions is None:
                    assert (
                        db_session.execute(
                            update(Configuration)
                            .where(
                                Configuration.parameter_id  # pylint: disable=no-member
                                == param_id
                            )
                            .values(value=value)
                        ).rowcount
                        == 1
                    ), f"Failed to update default for {param!r} to {value!r}"
                    delete_default = False
                else:
                    for expression in condition_expressions:
                        if expression not in db_expressions:
                            db_session.add(
                                ConditionExpression(expression=expression)
                            )
                            db_expressions[expression] = db_session.scalar(
                                select(ConditionExpression).filter_by(
                                    expression=expression
                                )
                            )

                    expression_set = frozenset(
                        db_expressions[expr].id
                        for expr in condition_expressions
                    )
                    _logger.debug("Expression set: %s", repr(expression_set))
                    condition_id = db_conditions.get(expression_set)
                    _logger.debug(
                        "Corresponding condition id: %s", condition_id
                    )

                    if condition_id is None:
                        condition_id = db_session.scalar(
                            select(
                                sql.functions.max(
                                    Condition.id  # pylint: disable=no-member
                                )
                                + 1
                            )
                        )
                        _logger.debug("New condition id: %d", condition_id)

                        for expression_id in expression_set:
                            db_session.add(
                                Condition(  # pylint: disable=not-callable
                                    id=condition_id, expression_id=expression_id
                                )
                            )
                            db_session.flush()
                        db_conditions[expression_set] = condition_id
                    db_session.add(
                        Configuration(  # pylint: disable=not-callable
                            parameter_id=param_id,
                            condition_id=condition_id,
                            version=0,
                            value=value,
                        )
                    )
            if delete_default:
                assert (
                    db_session.execute(
                        delete(Configuration).where(
                            Configuration.parameter_id  # pylint: disable=no-member
                            == param_id,
                            Configuration.condition_id  # pylint: disable=no-member
                            == 1,
                            Configuration.version  # pylint: disable=no-member
                            == 0,
                        )
                    ).rowcount
                    == 1
                )


def initialize_database(
    cmdline_args,
    step_dependencies=None,
    master_info=None,
    overwrite_default_config=None,
):
    """Initialize the database as specified on the command line."""

    if cmdline_args.drop_hdf5_structure_tables:
        drop_tables_matching(re.compile("hdf5_.*"))
    if cmdline_args.drop_all_tables:
        drop_tables_matching(re.compile(".*"))

    DataModelBase.metadata.create_all(get_db_engine())
    initialize_cmdline_database()
    if not cmdline_args.drop_all_tables:
        return

    init_processing(
        step_dependencies or defaults.step_dependencies,
        master_info or defaults.master_info,
    )
    if overwrite_default_config is None:
        overwrite_default_config = {
            param: [(("False",), None)]
            for param in [
                "srcfind-tool",
                "brightness-threshold",
                "filter-sources",
                "srcextract-max-sources",
            ]
        }
    _overwrite_default_config(overwrite_default_config)


if __name__ == "__main__":
    initialize_database(get_command_line_parser().parse_args())
