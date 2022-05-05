#!/usr/bin/env python3

"""Perform aperture photometry on a set of frames in parallel."""

from ctypes import c_char, c_double
from functools import partial
from multiprocessing import Pool

import numpy

from superphot import SubPixPhot, SuperPhotIOTree

from superphot_pipeline.fits_utilities import\
    get_primary_header,\
    read_image_components
from superphot_pipeline.image_utilities import find_fits_fnames
from superphot_pipeline.processing_steps.manual_util import\
    get_cmdline_parser,\
    read_subpixmap
from superphot_pipeline.processing_steps.fit_star_shape import add_image_options

from superphot_pipeline import DataReductionFile

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = get_cmdline_parser(
        __doc__,
        input_type='calibrated + dr',
        help_extra='The corresponding DR files must alread contain a PSF fit.',
        add_component_versions=('srcproj', 'background', 'shapefit'),
        allow_parallel_processing=True
    )
    parser.add_argument(
        '--apphot-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )

    add_image_options(parser)
    parser.add_argument(
        '--shapefit-group',
        type=int,
        default=0,
        help='If grouping was used during shape fitting, use this option to '
        'specify which of PSF map to use.'
    )
    parser.add_argument(
        '--apertures',
        nargs='+',
        type=float,
        help='The apretures to use for photometry.'
    )
    parser.add_argument(
        '--error-offset',
        type=float,
        help='A constant error to add to the formal error estimate from the '
        'measurement.'
    )
    return parser.parse_args()


def get_photometer(configuration):
    """
    Create an instance of SubPixPhot ready to be applied.

    Args:
        configuration(dict):    The configuration specifying how to run
            photometry.

    Returns:
        SubPixPhot:
            A fully configured instance ready to carry out photometry as
            specified by the given configuration.
    """

    return SubPixPhot(
        subpixmap=read_subpixmap(configuration['subpixmap']),
        apertures=numpy.array(configuration['apertures']),
        gain=configuration['gain'],
        magnitude_1adu=configuration['magnitude_1adu'],
        const_error=configuration['error_offset']
    )


def photometer_frame(frame_fname, configuration):
    """Perform aperture photometry on a single frame."""

    photometer = get_photometer(configuration)
    header = get_primary_header(frame_fname, True)
    header['FITGROUP'] = configuration['shapefit_group']

    with DataReductionFile(
            configuration['data_reduction_fname'].format_map(header),
            'a'
    ) as dr_file:
        io_tree = SuperPhotIOTree(photometer)
        num_sources = dr_file.fill_aperture_photometry_input_tree(
            io_tree,
            background_version=0,
            srcproj_version=0,
            shapefit_version=0
        )
        #False positive
        #pylint: disable=unbalanced-tuple-unpacking
        pixel_values, pixel_errors, pixel_mask = read_image_components(
            frame_fname,
            read_header=False
        )
        #pylint: enable=unbalanced-tuple-unpacking
        photometer(
            (
                pixel_values.astype(c_double, copy=False),
                pixel_errors.astype(c_double, copy=False),
                #False positive
                #pylint: disable=no-member
                pixel_mask.astype(c_char, order='C')
                #pylint: enable=no-member
            ),
            io_tree
        )
        dr_file.add_aperture_photometry(io_tree,
                                        num_sources,
                                        len(configuration['apertures']))


def photometer_image_collection(image_collection, configuration):
    """Extract aperture photometry from the given images."""

    photometer_one = partial(photometer_frame, configuration=configuration)
    if configuration['num_parallel_processes'] == 1:
        for frame_fname in image_collection:
            photometer_one(frame_fname)
    else:
        pool = Pool(processes=configuration['num_parallel_processes'])
        pool.imap_unordered(
            photometer_one,
            image_collection
        )
        pool.close()
        pool.join()

if __name__ == '__main__':
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    photometer_image_collection(
        find_fits_fnames(cmdline_config.pop('calibrated_images'),
                         cmdline_config.pop('apphot_only_if')),
        cmdline_config
    )
