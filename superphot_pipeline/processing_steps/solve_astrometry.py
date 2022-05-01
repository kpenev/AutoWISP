#!/usr/bin/env python3

"""Fit for a tronsfarmation between sky and image coordinates."""

import subprocess
from tempfile import mkstemp
import os

import numpy
import pandas

from superphot_pipeline.processing_steps.manual_util import get_cmdline_parser
from superphot_pipeline.image_utilities import find_dr_fnames
from superphot_pipeline import DataReductionFile

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = get_cmdline_parser(__doc__)
    parser.add_argument(
        'dr_files',
        nargs='+',
        help='A combination of individual data reduction files and directories '
        'to process (must already contain extracted sources). Directories are '
        'not searched recursively.'
    )
    parser.add_argument(
        '--astrometry-catalogue', '--astrometry-catalog', '--cat',
        required=True,
        help='A file containing (approximately) the same stars as those that '
        'were extracted from the frame for the area of the sky covered by the '
        'image. It is perferctly fine to include a larger area of sky, but it '
        'helps to have a brightness limit of the catalogue that matches the '
        'brightness limit of the extracted sources as closely as possible.'
    )
    parser.add_argument(
        '--frame-center-estimate',
        required=True,
        nargs=2,
        type=float,
        help='The approximate right ascention and declination of the center of '
        'the frame in degrees.'
    )
    parser.add_argument(
        '--frame-fov-estimate',
        required=True,
        type=float,
        help='Approximate field of view of the frame in degrees.'
    )
    parser.add_argument(
        '--max-srcmatch-distance',
        type=float,
        default=1.0,
        help='The maximum distance between a projected and extracted source '
        'center before we declare the two could not possibly correspond to the '
        'same star.'
    )
    parser.add_argument(
        '--astrometry-order',
        type=int,
        default=5,
        help='The order of the transformation to fit (i.e. the maximum combined'
        ' power of the cartographically projected coordinates each of the '
        'frame coordinates is allowed to depend on.'
    )
    parser.add_argument(
        '--anet-tweak',
        type=int,
        default=3,
        help='The tweak argument to anmatch anet.'
    )
    parser.add_argument(
        '--srcextract-version',
        default=0,
        help='The version of the extracted sources to use.'
    )
    parser.add_argument(
        '--anet-index-path',
        default='/data/CAT/ANET_INDEX/ucac4_2014',
        help='The path of the anet index to use.'
    )
    return parser.parse_args()


def get_sky_coord_columns(catalogue_fname):
    """Return the column numbers of RA and Dec within the catalogue file."""

    with open(catalogue_fname, 'r') as catfile:
        first_line = ''
        while not first_line:
            first_line = catfile.readline()
    assert first_line[0] == '#'

    result = dict(ra=None, dec=None)
    for column_number, column_name in enumerate(first_line[1:].split()):
        for coord in 'ra', 'dec':
            if column_name.lower().startswith(coord):
                assert result[coord] is None
                result[coord] = column_number

    return result['ra'], result['dec']


class TempAstrometryFiles:
    """Context manager for the temporary files needed for astrometry."""

    def __init__(self):
        """Create all required temporary files."""

        self._file_types = ['sources', 'match', 'trans']
        for file_type in self._file_types:
            handle, fname = mkstemp()
            setattr(self, '_' + file_type, handle)
            setattr(self, file_type + '_fname', fname)

    def __enter__(self):
        """Return the filenames of the temporary files."""

        return tuple(getattr(self, file_type + '_fname')
                     for file_type in self._file_types)

    def __exit__(self, *ignored_args, **ignored_kwargs):
        """Close and delete the temporary files."""

        for file_type in self._file_types:
            os.close(getattr(self, '_' + file_type))
            os.remove(getattr(self, file_type + '_fname'))

def solve_image(dr_fname, **configuration):
    """
    Find the astrometric transformation for a single image.

    Args:
        dr_fname(str):    The name of the data reduction file containing the
            extracted sources from the frame and that will be updated with the
            newly solved astrometry.

        configuration:    Parameters defining how astrometry is to be fit.

    Returns:
        None, but updates the input data reduction file with the newly solved
        astrometry.
    """

    def print_file_contents(fname, label):
        """Print the entire contenst of the given file."""

        print(80*'*')
        print(label.title() + ':')
        print(80*'-')
        with open(fname, 'r') as open_file:
            print(open_file.read())
        print(80*'-')


    print('Solving: ' + repr(dr_fname))
    cat_ra_col, cat_dec_col = get_sky_coord_columns(
        configuration['astrometry_catalogue']
    )
    print('RA, Dec columns: ' + repr((cat_ra_col, cat_dec_col)))
    with DataReductionFile(dr_fname, 'r+') as dr_file:
        header = dr_file.get_frame_header()
        with TempAstrometryFiles() as (sources_fname, match_fname, trans_fname):
            sources = pandas.DataFrame(
                dr_file.get_sources(
                    'srcextract.sources',
                    'srcextract_column_name',
                    srcextract_version=configuration['srcextract_version']
                )
            )
            x_col = int(numpy.argwhere(sources.columns == 'x'))
            y_col = int(numpy.argwhere(sources.columns == 'y'))
            sources.to_csv(sources_fname,
                           sep=' ',
                           na_rep='-',
                           float_format='%25.16e')
            numpy.savetxt(sources_fname, sources)
            print_file_contents(sources_fname, 'Sources')
            command = [
                'anmatch',
                '--comment',
                '--col-inp', '{0:d},{1:d}'.format(x_col + 1, y_col + 1),
                '--input', sources_fname,
                '--max-distance', repr(configuration['max_srcmatch_distance']),
                '--output-transformation', trans_fname,
                '--input-reference', configuration['astrometry_catalogue'],
                '--col-ref', '{0:d},{1:d}'.format(cat_ra_col + 1,
                                                  cat_dec_col + 1),
                '--output', match_fname,
                '--order', repr(configuration['astrometry_order']),
                '--ra', repr(configuration['frame_center_estimate'][0]),
                '--dec', repr(configuration['frame_center_estimate'][1]),
                '--anet',
                ','.join([
                    'indexpath={anet_index_path}',
                    'xsize={NAXIS1}',
                    'ysize={NAXIS2}',
                    'xcol={x_col}',
                    'ycol={y_col}',
                    'width={frame_fov_estimate}',
                    'tweak={anet_tweak}',
                    'log=1',
                    'verify=1'
                ]).format(**configuration,
                          **header,
                          x_col=x_col+1,
                          y_col=y_col+1)
            ]
            print('Executing:\n\t' + '\\\n\t'.join(command))
            subprocess.run(command, check=True)
            print_file_contents(match_fname, 'match')
            print_file_contents(trans_fname, 'trans')


def solve_astrometry(dr_collection, configuration):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    for dr_fname in dr_collection:
        solve_image(dr_fname, **configuration)


if __name__ == '__main__':
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    solve_astrometry(find_dr_fnames(cmdline_config.pop('dr_files')),
                     cmdline_config)
