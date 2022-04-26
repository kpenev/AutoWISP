#!/usr/bin/env python3

"""A script for visually exploring the results of a PSF/PRF fit."""

import functools
import os.path
import sys

from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from astropy.io import fits

from superphot.utils import explore_prf
from superphot.utils.file_utilities import get_fname_pattern_substitutions

from superphot_pipeline import PiecewiseBicubicPSFMap
from superphot_pipeline.image_utilities import read_image_components


def parse_command_line():
    """Parse command line to attributes of an object."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['explore_shapefit_map.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--dr-pattern',
        default=os.path.join('%(FITS_DIR)s',
                             '..',
                             'DR',
                             '%(FITS_ROOT)s.trans'),
        help="A pattern with substitutions involving any FITS header "
        "keywords, `'%%(FITS_DIR)s'` (directory containing the frame), and/or "
        "`'%%(FITS_ROOT)s'` (base filename of the frame without the `fits` or "
        "`fits.fz` extension) that expands to the filename of the data "
        "reduction file containing the PSF/PRF map to explore."
    )
    parser.add_argument(
        '--subpix-map',
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

    return explore_prf.parse_command_line(parser)


#TODO figure out this piecewise thing

def main(cmdline_args):
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
        cmdline_args.dr_pattern
        %
        get_fname_pattern_substitutions(cmdline_args.frame_fname, header),
        return_sources=True
    )

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
        if cmdline_args.subpix_map is None:
            slice_splines = [
                psf.predict_pixel for psf in slice_splines
            ]
        else:
            with fits.open(cmdline_args.subpix_map, 'readonly') as subpix_file:
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
                           cmdline_args)

    if cmdline_args.plot_3d_spline:
        explore_prf.plot_3d_prf(cmdline_args, *eval_coords, eval_prf)

    if cmdline_args.plot_entire_prf:
        explore_prf.plot_entire_prf(cmdline_args,
                                    image_slices,
                                    *eval_coords,
                                    sources=sources)


if __name__ == '__main__':
    numpy.set_printoptions(threshold=sys.maxsize)
    main(parse_command_line())
