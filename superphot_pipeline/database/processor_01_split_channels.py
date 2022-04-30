#!/usr/bin/env python3
#TODO rename filename for its purpose (depends on what is has such as processing_progress etc.)
#pylint: disable=invalid-name

"""Perform splitting of channels and update corresponding database"""

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

from superphot_pipeline.image_calibration.fits_util import create_result
from superphot_pipeline.image_utilities import get_fits_fnames
from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

from datetime import datetime
from data_model import Image, \
    StepConfiguration, \
    StepType, \
    ProcessingProgress, \
    ProcessingThread, \
    StepInput, \
    FilenameConvention

DataModelBase.metadata.bind = db_engine

def parse_configuration():
    """Return the configuration to use for splitting by channel."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['TESS_sector07_camera01.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        '--config', '-c',
        is_config_file=True,
        help='Config file to use instead of default.'
    )

    parser.add_argument(
        'images',
        nargs='+',
        help='The images to split by channel. Should include either fits images'
        ' and/or directories. In the latter case, all `fits.fz` files in the'
        ' specified directory are split (sub-directories are not searched).'
    )
    parser.add_argument(
        '--outfname-pattern',
        default=('/mnt/md0/PANOPTES/%(FIELD)s/SPLIT_CHANNELS/'
                 '%(IMAGEID)s_%(CHANNEL)s.fits.fz'),
        help='A %%-substitution pattern involving FITS header keywords that '
        'expands to the filename for storing the by channel images. Note that '
        'the FILTERS keyword is replaced by the particular channel being saved.'
    )
    parser.add_argument(
        '--allow-overwrite', '--overwrite', '-f',
        action='store_true',
        help='If images exist and this argument is not passed, an excetpion is '
        'thrown.'
    )
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
        help='Set the verbosity of logging output.'
    )
    parser.add_argument(
        '--ignore-errors',
        action='store_true',
        help='Pass this option to ensure the script continues even if it '
        'encounters invalid FITS files or other runtime errors.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Pass this option to skip creating by channel files that already '
        'exist. If only some of the channel files for a given input image '
        'exist, those are overwritten if allowed, or an error is raised if not.'
    )
    return parser.parse_args()

def split_image(image_fname, outfname_pattern, allow_overwrite, skip_done):
    """Split the given image to its four channels."""

    channel_list = ['R0', 'G0', 'G1', 'B0']

    with fits.open(image_fname) as input_fits:
        #false positive
        #pylint: disable=no-member
        header = fits.Header(input_fits[1].header)
        outfnames = [
            outfname_pattern % dict(header, CHANNEL=channel)
            for channel in channel_list
        ]
        assert header['FILTER'].strip() == 'RGGB'
        if skip_done:
            if reduce(
                    lambda found, outfname: os.path.exists(outfname) and found,
                    outfnames,
                    True
            ):
                logging.info('All channel files for %s already exist.',
                             image_fname)
                return

        #pylint: enable=no-member
        for channel, x_offset, y_offset in [
                ('R0', 0, 1),
                ('G0', 1, 1),
                ('G1', 0, 0),
                ('B0', 1, 0)
        ]:
            header['FILTER'] = channel[0]
            header['CHANNEL'] = channel
            outfname = outfname_pattern % header
            #false positive
            #pylint: disable=no-member
            pixels = input_fits[
                1
            ].data[
                y_offset::2,
                x_offset::2
            ].astype(
                numpy.float32
            )
            #pylint: enable=no-member

            output_dir = os.path.dirname(outfname)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            create_result(
                [
                    pixels,
                    pixels**0.5,
                    numpy.zeros(pixels.shape, dtype=numpy.uint8)
                ],
                header,
                outfname,
                -0.5,
                allow_overwrite
            )

def split(configuration):
    """Actually perform the splitting specified by the given configuration."""

    logging.basicConfig(level=getattr(logging, configuration.log_level))
    for image_fname in get_fits_fnames(configuration.images):
        with db_session_scope() as db_session:
            if db_session.query.filter(Image.notes.match(image_fname)):
                logging.debug('Splitting %s', repr(image_fname))
                try:
                    split_image(image_fname,
                                configuration.outfname_pattern,
                                configuration.allow_overwrite,
                                configuration.resume)
                except RuntimeError:
                    if configuration.ignore_errors:
                        logging.error('Failed to split %s:\n%s',
                                      image_fname,
                                      traceback.format_exc())
                    else:
                        raise


if __name__ == '__main__':
    split(parse_configuration())
