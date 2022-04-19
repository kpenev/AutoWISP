#!/usr/bin/env python3

"""A script for visually exploring the results of a PSF/PRF fit."""

import functools
import os.path
import sys

import inspect
from configargparse import ArgumentParser, DefaultsFormatter
import scipy
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from astropy.io import fits
import pprint
from matplotlib import pyplot

from superphot import PiecewiseBicubicPSFMap, SuperPhotIOTree, SubPixPhot
from superphot.utils import explore_prf
from superphot.utils.file_utilities import get_fname_pattern_substitutions

from superphot_pipeline import DataReductionFile
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


def get_shape_map_sources(dr_fname):
    """Return the PSF map contained in the given DR file."""

    dummy_tool = SubPixPhot()
    io_tree = SuperPhotIOTree(dummy_tool)

    with DataReductionFile(dr_fname, 'r') as dr_file:
        dr_file.fill_aperture_photometry_input_tree(io_tree)

        sources = dr_file.get_source_data(magfit_iterations=[0],
                                          shapefit=True,
                                          apphot=False,
                                          shapefit_version=0,
                                          srcproj_version=0,
                                          background_version=0)
        inputs = dr_file.get_aperture_photometry_inputs(shapefit_version=0,
                                                        srcproj_version=0,
                                                        background_version=0)
    star_shape_grid = inputs['star_shape_grid']
    grid_x = star_shape_grid[0]
    grid_y = star_shape_grid[1]
    magnitudes = scipy.copy(sources['mag'])
    sources.dtype.names = tuple('flux' if field == 'mag' else field
                                for field in sources.dtype.names)
    sources['flux'] = explore_prf.flux_from_magnitude(
        magnitudes,
        io_tree.get('psffit.magnitude_1adu')
    )

    return PiecewiseBicubicPSFMap(io_tree), sources, grid_x, grid_y

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
    image_resolution = (header['NAXIS2'], header['NAXIS1'])
    prf_map, sources, grid_x, grid_y = get_shape_map_sources(
        cmdline_args.dr_pattern
        %
        get_fname_pattern_substitutions(
            cmdline_args.frame_fname,
            header
        )
    )
    # image_center_x = image_resolution[1] / 2
    # image_center_y = image_resolution[0] / 2

    spline_x_psf = scipy.array(scipy.linspace(grid_x.min(), grid_x.max(), cmdline_args.spline_spacing))
    spline_y_psf = scipy.array(scipy.linspace(grid_y.min(), grid_y.max(), cmdline_args.spline_spacing))

    meshgrid_x, meshgrid_y = numpy.meshgrid(spline_x_psf, spline_y_psf)

    # prf = prf_map(x=scipy.array([image_center_x]), y=scipy.array([image_center_y]))
    # eval_prf = scipy.array([prf(x=grid_x[i], y=grid_y[i]) for i in range(grid_x[0].size)])

    image_slices = explore_prf.get_image_slices(
        cmdline_args.split_image,
        cmdline_args.discard_image_boundary
    )

    slice_prf_data = explore_prf.extract_pixel_data(cmdline_args,
                                                    image_slices,
                                                    sources=sources)

    slice_splines = [
        prf_map(
            x=scipy.array([
                (
                    x_image_slice.start
                    +
                    (x_image_slice.stop or image_resolution[1])
                ) / 2.0
            ]),
            y=scipy.array([
                (
                    y_image_slice.start
                    +
                    (y_image_slice.stop or image_resolution[0])
                ) / 2.0
            ])
        )
        for x_image_slice, y_image_slice, x_index, y_index in image_slices
    ]


    # eval_prf = numpy.array([slice(meshgrid_x, meshgrid_y) for slice in slice_splines])

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
    eval_prf = [slice(meshgrid_x, meshgrid_y) for slice in slice_splines]

    explore_prf.show_plots(slice_prf_data,
                           slice_splines,
                           cmdline_args)

    if cmdline_args.plot_3d_spline:
        explore_prf.plot_3d_prf(cmdline_args, meshgrid_x, meshgrid_y, eval_prf)

    if cmdline_args.plot_entire_prf:
        explore_prf.plot_entire_prf(cmdline_args,
                                    image_slices,
                                    grid_x,
                                    grid_y,
                                    sources=sources)


if __name__ == '__main__':
    main(parse_command_line())
