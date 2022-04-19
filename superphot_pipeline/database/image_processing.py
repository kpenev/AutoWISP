#!/usr/bin/env python3
#TODO rename filename for its purpose (depends on what is has such as add_9mage)
#pylint: disable=invalid-name

"""Add images/frames from list to database."""

from ctypes import c_char

import numpy
from astropy.io import fits
from configargparse import ArgumentParser, DefaultsFormatter
import re
from sqlalchemy import exc

from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase
from datetime import datetime
from data_model import Image, ImageType, ObservingSession

DataModelBase.metadata.bind = db_engine

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description='Apply image processing to frame list'
    )
    parser.add_argument(
        'images',
        nargs='+',
        help='The images to process into the database'
    )
    parser.add_argument(
        '--image_type',
        choices=('', '', '', ''), #TODO fill choices from dict
        default='', #TODO set default value for image_type to process
        help='Set the verbosity of logging output.'
    )
    parser.add_argument(
        '--observing_session_notes',
        help='The observing session notes used for corresponding database entry of observing session, '
             'which is used to match to said observing session.'
    )
    parser.add_argument(
        '--notes',
        help='Pass this option to add notes for the image(s) database entry'
    )
    parser.add_argument(
        '--description',
        help='Pass this option to add the description of the image(s) for database entry'
    )
    parser.add_argument(
        '--panoptes',
        default=False,
        help='Pass this option if PANOPTES images are used'
    )
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
        help='Set the verbosity of logging output.'
    )
    image_ftype = dict() #TODO ask Kalo what dict values for each image type
    loglevel = dict(critical=logging.CRITICAL,
                    debug=logging.DEBUG,
                    error=logging.ERROR,
                    info=logging.INFO,
                    warning=logging.WARNING
                    )
    arguments = parser.parse_args()
    arguments.image_type = image_ftype[arguments.image_type] #TODO ask Kalo if this is the correct way of implementing it
    arguments.verbose = loglevel[arguments.verbose]
    return arguments

def get_header(image_fname):
    """
    Return a the header of the given frame supplemented with filename info.

    Args:
        image_fname:    The filename of the frame to get the header of.

    Returns:
        header:    The header of the frame (can add more keywords if needed)
    """

    header = read_image_components(image_fname,
                                   read_image=False,
                                   read_error=False,
                                   read_mask=False,
                                   read_header=True)[0]

    return header

def add_image_database(image_fname, image_ftype, observing_session_notes=None, notes=None, description=None, panoptes=False):
    """
        Adds the image to the database.

        Args:
            image_fname:    The filename of the frame to add to the database.

            image_ftype:    The image type of the frame

            notes:  The notes section to add for the following database entry for the image

            description:    The description of the image type used

        """
    #TODO add/fix arguments so anyone can use, make a special panoptes argument
    #TODO specify imagetype argument and then read from the header, have it intially as none and if its none then it will pull from the imagtype keyword from the header
    # TODO add imaGE_TYPE TO DATABASE INTILIAZATION
    # TODO the image type needs to be read and itll make some dict of all of the image types that become registered

    header = get_header(image_fname)
    if panoptes:

        with db_session_scope() as db_session:

            if db_session.query.filter(ImageType.type_name.match(image_ftype)):
                image_type = db_session.query.filter(ImageType.type_name.match(image_ftype))

            elif header['IMAGETYP'] == image_ftype:
                image_type = ImageType(type_name=header['IMAGETYP'],
                                       description=description,
                                       timestamp=datetime.now())

            else:
                image_type = ImageType(type_name=image_ftype,
                                       description=description,
                                       timestamp=datetime.now())

            image = Image(notes=notes,
                          timestamp=datetime.now())

            image.image_type = image_type
            image_type.image = image
            image.observing_session = db_session.query.filter(ObservingSession.notes.match(header['SEQID']))

            db_session.add_all([image_type,
                                image])
    else:

        with db_session_scope() as db_session:

            if db_session.query.filter(ImageType.type_name.match(image_ftype)):
                image_type = db_session.query.filter(ImageType.type_name.match(image_ftype))

            elif header['IMAGETYP']==image_ftype:
                image_type = ImageType(type_name=header['IMAGETYP'],
                                       description=description,
                                       timestamp=datetime.now())

            else:
                image_type = ImageType(type_name=image_ftype,
                                       description=description,
                                       timestamp=datetime.now())

            image = Image(notes=notes,
                          timestamp=datetime.now())

            image.image_type = image_type
            image_type.image = image
            if observing_session_notes:
                image.observing_session = db_session.query.filter(ObservingSession.notes.match(observing_session_note))
            else:
                raise ValueError('Missing the proper observing session note to match to corresponding observing session')

            db_session.add_all([image_type,
                                image])


if __name__ == '__main__':
    cmdline_args = parse_command_line()
    logging.basicConfig(level=cmdline_args.verbose)

    for image in cmdline_args.images:
        add_image_database(image,
                           cmdline_args.image_type,
                           cmdline_args.observing_session_notes,
                           cmdline_args.notes,
                           cmdline_args.description,
                           cmdline_args.panoptes)
