#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

import re

from configargparse import ArgumentParser, DefaultsFormatter

from superphot_pipeline.database.interface import db_engine, Session
from superphot_pipeline.database.data_model.base import DataModelBase

from superphot_pipeline.database.initialize_data_reduction_structure import\
    get_default_data_reduction_structure
from superphot_pipeline.database.initialize_light_curve_structure import\
    get_default_light_curve_structure
from superphot_pipeline import processing_steps
#false positive due to unusual importing
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    ImageType,\
    Step,\
    Parameter,\
    Configuration,\
    Condition,\
    ConditionExpression
#pylint: enable=no-name-in-module

def get_command_line_parser():
    """Create a parser with all required command line arguments."""

    parser = ArgumentParser(
        description='Initialize the database for first time use of the '
        'pipeline.',
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        # default=config_file,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )
    parser.add_argument(
        '--drop-all-tables',
        action='store_true',
        help='If passed all pipeline tables are deleted before new ones are '
        'created'
    )
    parser.add_argument(
        '--drop-hdf5-structure-tables', '--drop-structure',
        action='store_true',
        help='If passed, tables defining the structure of HDF5 files are '
        'dropped first and then re-created and filled. Otherwise, if tables '
        'exist, their contents is not modified.'
    )
    parser.add_argument(
        '--verbose',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the verbosity of the DB logger.'
    )
    return parser


def add_default_hdf5_structures(data_reduction=True, light_curve=True):
    """
    Add a default HDF5 structure to the database.

    Args:
        data_reduction(bool):    Should the structure of data reduction files be
            initialized?

        light_curve(bool):    Should the structure of light curve files be
            initialized?
    """

    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        if data_reduction:
            db_session.add(get_default_data_reduction_structure())
        if light_curve:
            db_session.add(get_default_light_curve_structure(db_session))


#No good way to simplify
#pylint: disable=too-many-locals
def init_processing():
    """Initialize the tables controlling how processing is to be done."""

    step_dependencies = [
        ('add_images_to_db', []),
        ('calibrate', []),
        ('find_stars', ['calibrate']),
        ('solve_astrometry', ['find_stars']),
        ('fit_star_shape', ['solve_astrometry', 'calibrate']),
        ('measure_aperture_photometry', ['fit_star_shape', 'calibrate']),
        ('fit_magnitudes', ['fit_star_shape', 'measure_aperture_photometry']),
        ('fit_source_extracted_psf_map', ['find_stars', 'solve_astrometry']),
        ('create_lightcurves', ['solve_astrometry',
                                'fit_star_shape',
                                'measure_aperture_photometry',
                                'fit_magnitudes',
                                'fit_source_extracted_psf_map']),
        ('epd', ['create_lightcurves']),
        ('tfa', ['create_lightcurves', 'epd'])
    ]

    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        for image_type in ['bias', 'dark', 'flat', 'object']:
            db_session.add(ImageType(type_name=image_type))
        db_steps = {}
        db_parameters = {}
        db_configurations = []
        default_expression = ConditionExpression(id=1,
                                                 expression='True',
                                                 notes='Default expression')
        db_session.add(default_expression)

        default_condition = Condition(id=1,
                                      expression_id=default_expression.id,
                                      notes='Default configuration')
        db_session.add(default_condition)

        for step_id, (step_name, dependencies) in enumerate(step_dependencies):
            step_module = getattr(processing_steps, step_name)
            db_steps[step_name] = Step(
                id=step_id + 1,
                name=step_name,
                description=step_module.__doc__
            )
            for required_name in dependencies:
                db_steps[step_name].requires.append(db_steps[required_name])

            print(f'Initializing {step_name} parameters')
            default_step_config = step_module.parse_command_line([])
            print(
                'Default step config:\n\t'
                +
                '\n\t'.join(
                    f'{param}: {value}'
                    for param, value in default_step_config.items()
                )
            )
            for param in default_step_config['argument_descriptions'].keys():
                if (
                        param not in ['h',
                                      'config-file',
                                      'extra-config-file',
                                      'num-parallel-processes',
                                      'epd-datasets',
                                      'tfa-datasets']
                        and
                        not param.endswith('-only-if')
                        and
                        not param.endswith('-version')
                        and
                        not param.endswith('-catalog')
                ):
                    description = (
                        default_step_config['argument_descriptions'][param]
                    )
                    if isinstance(description, dict):
                        description = description['help']
                        configuration = None
                    if param not in db_parameters:
                        db_parameters[param] = Parameter(
                            name=param,
                            description=description
                        )
                        configuration = Configuration(
                            version=0,
                            condition_id=default_condition.id,
                            value=default_step_config[
                                'argument_defaults'
                            ][
                                param
                            ]
                        )
                        configuration.parameter = db_parameters[param]
                        db_configurations.append(configuration)
                    db_steps[step_name].parameters.append(db_parameters[param])

            db_session.add(db_steps[step_name])

        db_session.add_all(db_configurations)
#pylint: enable=too-many-locals


def drop_tables_matching(pattern):
    """Drop tables with names matching a pre-compiled regular expression."""


    if pattern is None:
        DataModelBase.metadata.drop_all(db_engine)
    else:
        DataModelBase.metadata.drop_all(
            db_engine,
            filter(
                lambda table: pattern.fullmatch(table.name),
                reversed(DataModelBase.metadata.sorted_tables)
            )
        )


def initialize_database(cmdline_args):
    """Initialize the database as specified on the command line."""

    if cmdline_args.drop_hdf5_structure_tables:
        drop_tables_matching(re.compile('hdf5_.*'))
    if cmdline_args.drop_all_tables:
        drop_tables_matching(re.compile('.*'))
    DataModelBase.metadata.create_all(db_engine)
    init_processing()
    add_default_hdf5_structures()


if __name__ == '__main__':
    initialize_database(get_command_line_parser().parse_args())
