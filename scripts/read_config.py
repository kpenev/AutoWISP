""" Program to upload camera configuration files to database.

Usage in terminal:
    > python3 read_config.py --filename (REQUIRED) [name/path to config file] 
                            --version [what version to assign updated parameters]
                            --option (REQUIRED) [how to deal with duplicate parameters]
"""
from configargparse import ArgumentParser
from autowisp.database.interface import Session
from autowisp.database.processing import ProcessingManager
from datetime import datetime
from autowisp.database.data_model.steps_and_parameters import Parameter
from autowisp.database.data_model.configuration import Configuration
from sqlalchemy import select, func
import configparser

import sys
sys.path.append("..") #from parent directory import...

#comments seem to be getting stuck when reading with configparse so remove them
def remove_comments(text):
    sep = '#'
    stripped = text.split(sep, 1)[0]
    return stripped


def add_to_db(version, filename, option):
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
            # largest version here
        # find max version in table and add 1 for next set of parameters to be entered
        max_version = version
        if max_version is None:
            max_version = (db_session.query(func.max(Configuration.version)).first())[0] + 1
        
        process_manage = ProcessingManager(max_version) #is this an ok value to have here?
        # print(merged_config.items())
        for keys,values in merged_config.items():
            config = process_manage.configuration.get(keys)
            # check that key matches parameter in table
            # this if statement get's all the values that match to that parameter
            if config is not None:
                value = config.get('value')
                # test.append(value)
                value = list(value.values())###FROZEN LIST HERE
                # test.append(value)
            
            # check that configuration exists in parameter table and then add if so
            param = db_session.query(Parameter.id).filter_by(name=keys).first()
            if (param is not None):
                #check if value exists in config table
                exists = False
                for val in value:
                    if (val is not None and remove_comments(val) == values):
                        exists = True
                        continue

                if option == 1 and exists: #throw error if configuration value already exists
                    raise Exception("configuration {} with value {} already exists in table".format(keys,values))
                elif option == 2 and exists:   #if exists ignore and don't add to db
                    print("not adding %s to table because it matches previous value"%(keys))
                    break
                else:   #add new configuration regardless of existence
                    # continue
                    # adding stuff to table with n+1 configuration
                    db_session.add(
                        Configuration(parameter_id=param[0], version= max_version, condition_id=1, value=remove_comments(values),
                                      notes="test config", timestamp=datetime.utcnow()))
                # print(keys,values)
            else:
                print("Key: %s is NOT a valid value in PARAMETER table"%(keys))
                #use sys.exit here??
                # sys.exit()
        db_session.commit()
        # print(test)

""" Retrieves configuration file path from terminal
"""
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', help='name of the configuration file to add to database', required=True)
    # default of version should come from processing script
    parser.add_argument('--version', type=int, default=None, help='version of configurations to add to db')
    parser.add_argument('--option', choices=['1','2','3'], help='''method to add configurations to db
                                                   (1) Throw error if configuration to enter already exists
                                                   (2) If configuration exists in db ignore and don\'t add to db
                                                   (3) Add new configuration regardless of existence in d''', required=True)
    args = parser.parse_args()

    # example dbpath = scripts/automateDb.db
    # example filename = scripts/PANOPTES_R.cfg
    add_to_db(args.version, args.filename, int(args.option))

    #for testing
    # add_to_db(None, 'PANOPTES_R.cfg', 3)