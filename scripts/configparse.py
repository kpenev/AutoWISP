""" Program to upload camera configuration files to database.

Usage in terminal:
    > python configparse.py --dbpath <path to db> --filename <path to configuration file>
"""
import sqlite3
from configargparse import ArgumentParser
import sys
sys.path.append("..") #from parent directory import...
from autowisp.processing_steps import __all__ as all_steps


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
def add_to_db(dbpath, filename):
    try:
        # need to allow any configuration file and database!!!
        sqliteConnection = sqlite3.connect(dbpath)
        cursor = sqliteConnection.cursor()
        print("Database created and Successfully Connected to SQLite")

        #get all parameters needed from config file and put into dictionary
        config_dict, condition_dict = filter_dict(parse_cmd(filename))

        #get how many elements in table to keep track of id
        id = (cursor.execute("SELECT COUNT(id) FROM configuration")).fetchall()[0][0]

        # populate configuration table
        for x in config_dict:
            sqlcmd = "INSERT INTO configuration VALUES (?,?,?,?,?,?,?)"
            param = str(x)
            val = str(config_dict[x])
            cursor.execute(sqlcmd, (id, 0, 0, param, val, '', 0))
            id +=1
        sqliteConnection.commit()

        condition_id = (cursor.execute("SELECT COUNT(id) FROM conditions")).fetchall()[0][0]
        expression_id = (cursor.execute("SELECT COUNT(id) FROM condition_expressions")).fetchall()[0][0]

        # populate condition and condition_expressions tables
        for x in condition_dict:
            conditions_sql = "INSERT INTO conditions VALUES (?,?,?,?)"
            expressions_sql = "INSERT INTO condition_expressions VALUES (?,?,?,?)"
            condition = str(x)
            expression = str(condition_dict[x])
            cursor.execute(conditions_sql, (condition_id, 0, condition, 0))

            #check that expression does not already exist
            cursor.execute("SELECT expression FROM condition_expressions WHERE expression = ?", (expression,))
            if not cursor.fetchone():
                cursor.execute(expressions_sql, (expression_id, expression, '', 0))
                expression_id +=1
            condition_id +=1
        sqliteConnection.commit()

        cursor.close()
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The SQLite connection is closed")


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
