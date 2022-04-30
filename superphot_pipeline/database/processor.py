#!/usr/bin/env python3
#TODO rename filename for its purpose (depends on what is has such as processing_progress etc.)
#pylint: disable=invalid-name

"""General processing of image per step configuration"""

from ctypes import c_char

import numpy
from astropy.io import fits
from configargparse import ArgumentParser, DefaultsFormatter
import re
from sqlalchemy import exc
import os.path
import os
import logging
import traceback
from functools import reduce
from configargparse import ArgumentParser, DefaultsFormatter

from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

from datetime import datetime
from data_model import Image, \
    StepConfiguration, \
    StepType, \
    ProcessingProgress, \
    ProcessingThread, \
    FilenameConvention

###Import for proper step###
from superphot_pipeline.image_calibration.fits_util import create_result
from superphot_pipeline.image_utilities import get_fits_fnames

DataModelBase.metadata.bind = db_engine

def processor(step_type, parameters, values):
    with db_session_scope() as db_session:

        #Find the step_type_id from given step_type can be either a description or the id itself
        if db_session.query.filter(Steptype.id.match(step_type)):
            type_id = db_session.query.get(Steptype.id.match(step_type))
        elif db_session.query.filter(Steptype.description.match(step_type)):
            type_id = db_session.query.get(Steptype.description.match(step_type))
        else:
            raise ValueError('The corresponding step_type id or description does not exist')

        for parameter, value in zip(parameters,values):
            step_configuration = StepConfiguration(step_type_id=type_id,
                                                   parameter=parameter,
                                                   value= value)
