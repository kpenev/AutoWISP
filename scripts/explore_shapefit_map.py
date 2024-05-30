#!/usr/bin/env python3

"""A script for visually exploring the results of a PSF/PRF fit."""

import functools
import os.path
import sys

from matplotlib import pyplot
import numpy
from astropy.io import fits

from astrowisp.utils import explore_prf

from autowisp import PiecewiseBicubicPSFMap
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import\
    get_primary_header,\
    read_image_components
from autowisp.processing_steps.manual_util import\
    ManualStepArgumentParser

def parse_command_line():
    """Parse command line to attributes of an object."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='calibrated + dr',
#        processing_step='explore_shapefit_map',
        add_component_versions=('srcproj', 'background', 'shapefit'),
        convert_to_dict=False
    )

    parser.add_argument(
        '--num-simultaneous',
        type=int,
        default=1,
        help='The number of frames that were fit simultaneously, with a unified'
        ' PSF/PRF model. Each simultaneous group consists of consecutive '
        'entries in the input list of frames, so unless this argument is `1`, '
        'the input must be exactly the as the one used for fitting the shape.'
    )
    parser.add_argument(
        '--subpixmap',
        default=None,
        help='If passed, should point to a FITS image to be used as the '
        'sub-pixel sensitivity map. Otherwise, uniform pixels are assumed.'
    )
    parser.add_argument(
        '--assume-psf',
        action='store_true',
        default=False,
        help='If passed, the map contained in the given file is integrated, '
        'possibly combined with the sub-pixel map, to predict the response of '
        'pixels assuming the map is a PSF (as opposed to PRF) map.'
    )
    parser.add_argument(
        '--gain',
        default=None,
        type=float,
        help='The gain to assume for the input image (electrons/ADU). If not '
        'specified, it must be defined in the header as GAIN keyword.'
    )

    return explore_prf.parse_command_line(parser,
                                          assume_sources=True,
                                          add_config_file=False,
                                          add_frame_arg=False)


def main(cmdline_args, last=True):
    """Avoid polluting global namespace."""

    if cmdline_args.skip_existing_plots:
        all_plots_exist = True
        for plot_fname in explore_prf.list_plot_filenames(cmdline_args):
            all_plots_exist = all_plots_exist and os.path.exists(plot_fname)
        if all_plots_exist:
            return

    header = read_image_components(cmdline_args.frame_fname,
                                   read_image=False,
                                   read_error=False,
                                   read_mask=False)[0]
    #False positive
    #pylint: disable=unsubscriptable-object
    image_resolution = (header['NAXIS2'], header['NAXIS1'])
    #pylint: enable=unsubscriptable-object
    prf_map = PiecewiseBicubicPSFMap()
    sources = prf_map.load(
        cmdline_args.data_reduction_fname.format_map(
            get_primary_header(cmdline_args.frame_fname, True)
        ),
        return_sources=True,
        **{component + '_version': getattr(cmdline_args, component + '_version')
           for component in ('srcproj', 'background', 'shapefit')}
    ).to_records()
    sources['flux'] *= cmdline_args.gain

    # image_center_x = image_resolution[1] / 2
    # image_center_y = image_resolution[0] / 2

    eval_coords = [
        numpy.linspace(grid.min(), grid.max(), cmdline_args.spline_spacing)
        for grid in prf_map.configuration['grid']
    ]

    eval_coords = numpy.meshgrid(*eval_coords)

#    prf = prf_map(x=numpy.array([image_center_x]),
#                  y=numpy.array([image_center_y]))
#    eval_prf = numpy.array(
#        [prf(x=grid_x[i], y=grid_y[i]) for i in range(grid_x[0].size)]
#    )

    image_slices = explore_prf.get_image_slices(
        cmdline_args.split_image,
        cmdline_args.discard_image_boundary
    )

    slice_prf_data = explore_prf.extract_pixel_data(cmdline_args,
                                                    image_slices,
                                                    sources=sources)

    slice_splines = [
        prf_map(
            numpy.array(
                (
                    (
                        x_image_slice.start
                        +
                        (x_image_slice.stop or image_resolution[1])
                    ) / 2.0,
                    (
                        y_image_slice.start
                        +
                        (y_image_slice.stop or image_resolution[0])
                    ) / 2.0,
                ),
                dtype=[('x', float), ('y', float)]
            )
        )
        for x_image_slice, y_image_slice, x_index, y_index in image_slices
    ]


    # eval_prf = numpy.array([slice(*eval_coords) for slice in slice_splines])

    if cmdline_args.assume_psf:
        if cmdline_args.subpixmap is None:
            slice_splines = [
                psf.predict_pixel for psf in slice_splines
            ]
        else:
            with fits.open(cmdline_args.subpixmap, 'readonly') as subpix_file:
                slice_splines = [
                    functools.partial(
                        psf.predict_pixel,
                        #False positive
                        #pylint: disable=no-member
                        subpix_map=(subpix_file[0].data
                                    if subpix_file[0].header['NAXIS'] > 0 else
                                    subpix_file[1].data)
                        #pylint: enable=no-member
                    )
                    for psf in slice_splines
                ]
    #use .flatten on arrays
    eval_prf = [slice(*eval_coords) for slice in slice_splines]

    explore_prf.show_plots(slice_prf_data,
                           slice_splines,
                           cmdline_args,
                           append=(not last))

    if cmdline_args.plot_3d_spline:
        explore_prf.plot_3d_prf(cmdline_args, *eval_coords, eval_prf)

    if cmdline_args.plot_entire_prf:
        explore_prf.plot_entire_prf(cmdline_args,
                                    image_slices,
                                    *eval_coords,
                                    sources=sources)


if __name__ == '__main__':
    numpy.set_printoptions(threshold=sys.maxsize)
    config = parse_command_line()
    frame_list = sorted(
        find_fits_fnames(config.calibrated_images)
    )
    for frame_index, frame_fname in enumerate(frame_list):
        print('Plotting ' + repr(frame_fname))
        config.frame_fname = frame_fname
        last_in_group = (frame_index + 1) % config.num_simultaneous == 0
        main(config, last_in_group)
        if last_in_group:
            pyplot.clf()
            pyplot.cla()
