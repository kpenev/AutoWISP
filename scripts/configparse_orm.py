""" Program to upload camera configuration files to database.

Usage in terminal:
    > python configparse.py --dbpath <path to db> --filename <path to configuration file>
"""
import sqlite3
from configargparse import ArgumentParser
from sqlalchemy.orm import Session
from superphot_pipeline.database.interface import Session
from superphot_pipeline.database.processing import ProcessingManager
from superphot_pipeline.database.data_model.configuration import Configuration
from superphot_pipeline.database.data_model.steps_and_parameters import Parameter
from datetime import datetime
import configparser

import sys
sys.path.append("..") #from parent directory import...

#comments seem to be getting stuck when reading with configparse so remove them
def remove_comments(text):
    sep = '#'
    stripped = text.split(sep, 1)[0]
    return stripped

def add_to_db(version, filename):
    config = configparser.ConfigParser()
    # append dummy section to appease configparse gods
    with open(filename) as f:
        file_content = '[dummy_section]\n' + f.read()

    config.read_string(file_content)

    merged_config = {}
    for section in config:
        merged_config = merged_config | dict(config[section])

    test = []
    with Session.begin() as db_session:
        # if no version was specified use n+1 version >> processing script
            # we'd want it to be a large number to capture all current versions
        process_manage = ProcessingManager(sys.maxsize) #is this an ok value to have here?
        for keys,values in merged_config.items():
            config = process_manage.configuration.get(keys)
            # check that key matches parameter in table
            if config is not None:
                if version == -1:
                    version = int(config.get('version')) + 1  # making it version n+1
                value = config.get('value')
                value = list(value.values())[0]
                test.append((remove_comments(values), value))
                #check that value is not already in table, if so skip
                if(remove_comments(values) == value):
                    print("not adding %s to table because it matches previous value"%(keys))
                    continue

            # check that configuration exists in parameter table and then add if so
            param = db_session.query(Parameter.id).filter_by(name=keys).first()
            if param is not None:
                # adding stuff to table with n+1 configuration
                print(keys,values)
                db_session.add(
                    Configuration(parameter_id=param[0], version=version, condition_id=1, value=remove_comments(values),
                                  notes="test config", timestamp=datetime.utcnow()))
            else:
                print("Key: %s is NOT a valid value in PARAMETER table"%(keys))
                #use sys.exit here??
                # sys.exit()
        db_session.commit()




""" Retrieves database path and configuration file path from terminal
"""
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', help='name of the configuration file to add to database')
    # default of version should come from processing script
    parser.add_argument('--version', nargs='?', type=int, default=-1, help='path to db to add configurations to')
    args = parser.parse_args()
    # example dbpath = scripts/automateDb.db
    # example filename = scripts/PANOPTES_R.cfg
    add_to_db(args.version, args.filename)


