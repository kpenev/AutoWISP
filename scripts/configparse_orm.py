""" Program to upload camera configuration files to database.

Usage in terminal:
    > python configparse.py --dbpath <path to db> --filename <path to configuration file>
"""
import sqlite3
from configargparse import ArgumentParser
# from sqlalchemy import create_engine
from sqlalchemy.orm import Session
# from sqlalchemy.ext.automap import automap_base
from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase
from superphot_pipeline.database.data_model.configuration import Configuration
from superphot_pipeline.database.data_model.condition_expression import ConditionExpression
from superphot_pipeline.database.data_model.condition import Condition
from superphot_pipeline.database.data_model.steps_and_parameters import Parameter
from datetime import datetime
import configparser
from itertools import chain

import sys
sys.path.append("..") #from parent directory import...
from superphot_pipeline.processing_steps import __all__ as all_steps


""" Retrieves desired parameters from configuration files.
    Args:
        filename: name of camera configuration file to extract parameters from
    Returns:
        A dictionary mapping parameters and values from configuration file
"""
def parse_cmd(filename):
    config_dict = {}
    for x in all_steps:
        if (hasattr(x, 'parse_command_line')):  # check processing steps for parse_cmd function present
            name = x.__name__.split('.')
            if name[2] == "fit_star_shape":     # extra parameters for fit_star_shape
                list = x.parse_command_line(['-c', filename, '--photometry-catalogue', 'dummy'])
            else:
                list = x.parse_command_line(['-c', filename])
            config_dict = config_dict | list
    return config_dict


""" Filters and organizes dictionary of parameters and values.
    
    Splits given dictionary into configuration and condition dictionaries
    Removes files from configuration dictionary
    
    Args:
        config_dict: dictionary of retrieved parameters from camera configuration
    Returns:
        Two filtered dictionaries one of configurations another of conditions and condition expressions
"""
def filter_dict(config_dict):
    res = {}
    conditions = {}
    #values to be removed/filtered
    keys_remove = ["only_if", "master_"]
    values_remove = [".fits.fz", ".h5", ".ucac4", "hdf5.0", ".fits", ".txt"]

    for key, value in config_dict.items():
        # only check values that are strings and are not None
        if isinstance(value, str) and value is not None:
            # check if values to remove are in value
            if [ele for ele in values_remove if (ele in value)]:
                continue    # value found, don't want in dictionary
        # check if keys to remove in key
        if [ele for ele in keys_remove if(ele in key)]:
            # found a condition, put in conditions dictionary
            if "only_if" in key:
                conditions[key] = value
            continue
        # parameters we do want
        res[key] = value

    return res,conditions


""" Adds configuration and condition dictionaries to database
    Args:
        dbpath: path to database to add information
        filename: path to camera configuration file
    Returns:
        None
"""
#comments seem to be getting stuck when reading with configparse so remove them
def remove_comments(text):
    sep = '#'
    stripped = text.split(sep, 1)[0]
    return stripped

def add_to_db(dbpath, filename):
    config = configparser.ConfigParser()
    # append dummy section to appease configparse gods
    with open(filename) as f:
        file_content = '[dummy_section]\n' + f.read()

    config.read_string(file_content)
    # print({section: dict(config[section]) for section in config})

    merged_config = {}
    for section in config:
        merged_config = merged_config | dict(config[section])

    # print(merged_config)

    # for keys,values in merged_config.items():
    #     print(keys + " : " + values)


    with db_session_scope() as db_session:
        for keys,values in merged_config.items():
            # check that key matches parameter in table
            exists = db_session.query(Parameter.id).filter_by(name=keys).first()
            if (exists):
                db_session.add(Configuration(parameter_id=exists[0], version=0, condition_id=0, value=values, notes="test config",
                                             timestamp=datetime.utcnow()))
                # print(exists[0])
            # print(keys + " : " + str(exists))

        # db_session.add(ConditionExpression(id=7,expression="test expression",notes="test condition expression",timestamp=datetime.utcnow()))
        # # db_session.add(Condition(id=0,expression_id=2,notes="test condition",timestamp=datetime.utcnow()))
        # # db_session.add(Configuration(parameter_id=1,version=0,condition_id=0,value=0,notes="test config",timestamp=datetime.utcnow()))
        # # for x in configdict:
        # #     db_session.add(Condition)
        db_session.commit()

    # try:
    #     # Base = automap_base()
    #     # engine = create_engine(dbpath, echo=True)
    #     # #reflect the tables
    #     # Base.prepare(autoload_with=engine)
    #     # Configuration = DataModelBase.classes.configuration
    #     # Condition = Base.classes.condition
    #     # Condition_Expression = Base.classes.condition_expression
    #     # Parameter = Base.classes.parameter
    #
    #     local_session = Session(bind=engine)
    #     print("Database created and Successfully Connected to SQLite")
    #
    #     # get all parameters needed from config file and put into dictionary
    #     config_dict, condition_dict = filter_dict(parse_cmd(filename))
    #
    #     #get how many elements in table to keep track of id
    #     param_id = local_session.query(Configuration).count()
    #
    #     # populate configuration table
    #     for x in config_dict:
    #         param = str(x)
    #         val = str(config_dict[x])
    #         local_session.add(Parameter(id=param_id, name=param, description="", timestamp=datetime.utcnow()))
    #         local_session.add(Configuration(parameter_id=param_id, version=0, condition_id=0, value=val, notes='', timestamp=datetime.utcnow()))
    #         param_id += 1
    #     local_session.commit()
    #
    #     condition_id = local_session.query(Condition).count()
    #     expression_id = local_session.query(Condition_Expression).count()
    #
    #     # populate condition and condition_expressions tables
    #     for x in condition_dict:
    #         condition = str(x)
    #         expression = str(condition_dict[x])
    #         local_session.add(Condition(id=condition_id,expression_id=0,notes=condition,timestamp=datetime.utcnow()))
    #
    #         #check that expression does not already exist
    #         exists = local_session.query(Condition_Expression).filter(Condition_Expression.expression==expression).first() is not None
    #         if(not exists):
    #             local_session.add(Condition_Expression(id=expression_id,expression=expression,notes='',timestamp=datetime.utcnow()))
    #             expression_id +=1
    #         condition_id +=1
    #     local_session.commit()
    # except sqlite3.Error as error:
    #     print("Error while connecting to sqlite", error)


""" Retrieves database path and configuration file path from terminal
"""
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', help='name of the configuration file to add to database')
    parser.add_argument('--dbpath', help='path to db to add configurations to')
    args = parser.parse_args()
    # example dbpath = scripts/automateDb.db
    # example filename = scripts/PANOPTES_R.cfg
    add_to_db(args.dbpath, args.filename)


