import sqlite3
import unittest.mock as mock
# import configargparse
import sys
sys.path.append("..") #from parent directory import...
from superphot_pipeline.processing_steps import __all__ as all_steps


def parse_cmd(filename):
    config_dict = {}
    for x in all_steps:
        if (hasattr(x, 'parse_command_line')):
            print(x.__name__)
            try:
                list = x.parse_command_line(['-c', filename])
            except SystemExit:
                list = x.parse_command_line(['-c', filename, '--photometry-catalogue', 'dummy'])
            config_dict = config_dict | list
            # try:
            #     list = x.parse_command_line(['-c', filename])
            #     config_dict = config_dict | list
            # except BaseException:  # ok to use BaseException? too broad, Configargparse.ArgumentException was not it
            #     list = x.parse_command_line(['-c', filename, '--photometry-catalogue', 'dummy'])
            #     config_dict = config_dict | list
    print("COMBINED ***************************")
    # print(config_dict)
    # for x in config_dict:
    #     print(x, config_dict[x])
    return config_dict

def add_to_db(dbpath):
    try:
        sqliteConnection = sqlite3.connect('automateDb.db')
        cursor = sqliteConnection.cursor()
        print("Database created and Successfully Connected to SQLite")

        #get all parameters needed from config file and put into dictionary
        config_dict = parse_cmd('PANOPTES_R.cfg')

        #get how many elements in table to keep track of id
        id = (cursor.execute("SELECT COUNT(id) FROM configuration")).fetchall()[0][0]

        # cursor.execute("INSERT INTO configuration VALUES (3, 0, 0, 0, 0, 0, 0)")
        # sqliteConnection.commit()

        for x in config_dict:
            sqlcmd = "INSERT INTO configuration VALUES (?,?,?,?,?,?,?)"
            param = str(x)
            val = str(config_dict[x])
            cursor.execute(sqlcmd, (id, 0, 0, param, val, 0, 0))
            id +=1
        sqliteConnection.commit()

        cursor.close()
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The SQLite connection is closed")


if __name__ == '__main__':
    # parse_cmd('PANOPTES_R.cfg')
    add_to_db('path dummy')
# parser = argparse.ArgumentParser()
# parser.add_argument('--filename', help='name of the configuration file to add to database')
# filename = parser.parse_args()