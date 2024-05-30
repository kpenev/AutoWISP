#!/usr/bin/env python3

"""Create regions file showing which stars are photometered in an image."""

from configargparse import ArgumentParser, DefaultsFormatter
import numpy

from autowisp import DataReductionFile

from plot_detrending_stat import detect_stat_columns

def parse_command_line():
    """Return command line configuration."""

    parser = ArgumentParser(description=__doc__,
                            formatter_class=DefaultsFormatter,
                            default_config_files=[],
                            ignore_unknown_config_file_keys=True)
    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )


    parser.add_argument(
        'dr_fname',
        help='Path to the data reduction file to create regions for.'
    )
    parser.add_argument(
        'stat_fname',
        help='The statistics file to plot (e.g. from magnitude fitting).'
    )


    parser.add_argument(
        '--min-unrejected-fraction',
        type=float,
        default=0.5,
        help='The fraction of the star with the most measurements a star must '
        'have to be included in the plot.'
    )
    parser.add_argument(
        '--apertures',
        nargs='+',
        type=float,
        help='The apretures to use for photometry.'
    )
    parser.add_argument(
        '--psf-grid-size',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='The width and height of the grid to use for PSF fitting.'
    )
    parser.add_argument(
        '--output', '-o',
        default='photometered_stars.reg',
        help='The regions file to create.'
    )
    parser.add_argument(
        '--hide-rejected',
        action='store_true',
        help='Do not mark sources with too few non rejected points.'
    )

    return parser.parse_args()


def find_best_apertures(stat_fname, min_unrejected_fraction, apertures):
    """Return dict of best aperture for each star (zero if PSF fitting)."""

    apertures.insert(0, 0.0)
    num_unrejected_cols, scatter_cols = detect_stat_columns(stat_fname)[:2]
    stat_data = numpy.genfromtxt(stat_fname)
    min_unrejected = numpy.min(stat_data[:, num_unrejected_cols], 1)
    many_unrejected = (min_unrejected
                       >
                       min_unrejected_fraction * numpy.max(min_unrejected))
    stat_data = stat_data[many_unrejected]
    best_ap = numpy.asarray(apertures)[
        numpy.nanargmin(stat_data[:, scatter_cols], 1)
    ]
    star_ids = numpy.genfromtxt(stat_fname,
                                usecols=[0],
                                dtype=None)[many_unrejected]
    return dict(zip(star_ids, best_ap))


def main(config):
    """Avoid polluting global namespace."""

    best_apertures = find_best_apertures(config.stat_fname,
                                         config.min_unrejected_fraction,
                                         config.apertures)
    region_format_str = (
        '{shape:s}({x!r}, {y!r}, {width!r} {height!r}) # color={color:s}\n'
    )
    with DataReductionFile(config.dr_fname, 'r') as dr_file:
        source_data = dr_file.get_source_data(magfit_iterations=[-1],
                                              shape_fit=False,
                                              apphot=False,
                                              background=False,
                                              srcproj_version=0,
                                              shapefit_version=0,
                                              apphot_version=0,
                                              magfit_version=0)
        with open(config.output, 'w') as outf:
            outf.write('# Region file format: DS9 version 4.1\n'
                       'global width=3\n')
            for source_id, source in source_data.iterrows():
                region_cfg = dict(x=source['x'] + 0.5,
                                     y=source['y'] + 0.5)
                if source_id in best_apertures:
                    aperture = best_apertures[source_id]
                    if aperture == 0:
                        (
                            region_cfg['width'],
                            region_cfg['height']
                        ) = config.psf_grid_size
                        region_cfg['shape'] = 'box'
                        region_cfg['color'] = ('#ff7f00' if source['enabled']
                                                  else '#984ea3')
                    else:
                        region_cfg['width'] = region_cfg['height'] = aperture
                        region_cfg['shape'] = 'ellipse'
                        region_cfg['color'] = ('#4daf4a' if source['enabled']
                                                  else '#377eb8')
                    outf.write(region_format_str.format_map(region_cfg))
                elif not config.hide_rejected:
                    region_cfg['color'] = '#e41a1c'
                    outf.write(
                        (
                            'point({x!r}, {y!r}) # point=X 10 color={color:s}\n'
                        ).format_map(
                            region_cfg
                        )
                    )


if __name__ == '__main__':
    main(parse_command_line())
