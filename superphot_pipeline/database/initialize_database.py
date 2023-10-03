#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

from argparse import ArgumentParser
import re
from sqlalchemy.exc import OperationalError

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
    Step,\
    Parameter,\
    Configuration,\
    Condition
#pylint: enable=no-name-in-module

def parse_command_line():
    """Parse the commandline optinos to attributes of an object."""

    parser = ArgumentParser(
        description='Initialize the database for first time use of the '
        'pipeline.'
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
    return parser.parse_args()


def add_default_hdf5_structures(data_reduction=True, light_curve=True):
    """
    Add a default HDF5 structure to the database.

    Args:
        data_reduction(bool):    Should the structure of data reduction files be
            initialized?

        light_curve(bool):    Should the structure of light curve files be
            initialized?
    """

    with Session.begin() as db_session:
        if data_reduction:
            db_session.add(get_default_data_reduction_structure())
        if light_curve:
            db_session.add(get_default_light_curve_structure(db_session))


def init_processing():
    """Initialize the tables controlling how processing is to be done."""

    step_dependencies = [
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
    with Session.begin() as db_session:
        db_steps = {}
        db_parameters = {}
        db_configurations = []
        default_condition = Condition(expression_id=None,
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
                f'Default step config:\n\t'
                +
                '\n\t'.join(
                    f'{param}: {value}'
                    for param, value in default_step_config.items()
                )
            )
            for param in default_step_config.keys():
                if (
                        param not in ['argument_descriptions',
                                      'argument_defaults',
                                      'num_parallel_processes',
                                      'epd_datasets',
                                      'tfa_datasets']
                        and
                        not param.endswith('_only_if')
                        and
                        not param.endswith('_version')
                        and
                        not param.endswith('_catalog')
                ):
                    description = (
                        default_step_config['argument_descriptions'][param]
                    )
                    if isinstance(description, dict):
                        param = description['rename']
                        description = description['help']
                        configuration = None
                    if param not in db_parameters:
                        db_parameters[param] = Parameter(
                            name=param,
                            description=description
                        )
                        configuration = Configuration(
                            version=0,
                            value=(
                                default_step_config['argument_defaults'][param]
                            )
                        )
                        configuration.condition = default_condition
                        configuration.parameter = db_parameters[param]
                        db_configurations.append(configuration)
                    db_steps[step_name].parameters.append(db_parameters[param])

            db_session.add(db_steps[step_name])

        db_session.add_all(db_configurations)


def drop_tables_matching(pattern):
    """Drop tables with names matching a pre-compiled regular expression."""

    for table in reversed(DataModelBase.metadata.sorted_tables):
        if pattern.fullmatch(table.name):
            try:
                table.drop(db_engine)
            except OperationalError:
                pass

if __name__ == '__main__':
    cmdline_args = parse_command_line()

    if cmdline_args.drop_hdf5_structure_tables:
        drop_tables_matching(re.compile('hdf5_.*'))
    if cmdline_args.drop_all_tables:
        drop_tables_matching(re.compile('.*'))
    DataModelBase.metadata.create_all(db_engine)
    init_processing()
    add_default_hdf5_structures()
