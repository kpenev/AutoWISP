#!/usr/bin/env python3

"""Fit for a tronsfarmation between sky and image coordinates."""

import subprocess
from tempfile import mkstemp
import os
import csv
import logging

import numpy
import pandas

from superphot_pipeline.hat.file_parsers import parse_anmatch_transformation
from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser,\
    read_catalogue
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline import DataReductionFile

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='dr',
        inputs_help_extra='The DR files must already contain extracted sources',
        add_component_versions=('srcextract', 'catalogue', 'skytoframe')
    )
    parser.add_argument(
        '--astrometry-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--astrometry-catalogue', '--astrometry-catalog', '--cat',
        default='astrometry_catalogue.ucac4',
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
        '--anet-tweak-range',
        type=int,
        nargs=2,
        default=(2, 5),
        help='Range of tweak arguments to anmatch anet to try.'
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


def print_file_contents(fname, label):
    """Print the entire contenst of the given file."""

    print(80*'*')
    print(label.title() + ': ')
    print(80*'-')
    with open(fname, 'r') as open_file:
        print(open_file.read())
    print(80*'-')


def create_sources_file(dr_file, sources_fname, srcextract_version):
    """Create a file with the given name contaning the extracted sources."""

    sources = dr_file.get_sources(
        'srcextract.sources',
        'srcextract_column_name',
        srcextract_version=srcextract_version
    )
    x_col = int(numpy.argwhere(sources.columns == 'x')) + 1
    y_col = int(numpy.argwhere(sources.columns == 'y')) + 1
    sources.to_csv(sources_fname,
                   sep=' ',
                   na_rep='-',
                   float_format='%.16e',
                   quoting=csv.QUOTE_NONE,
                   index=True,
                   header=False)
    print_file_contents(sources_fname, 'Sources file')

    return x_col, y_col


def save_match_to_dr(catalogue_sources,
                     extracted_sources,
                     match_fname,
                     dr_file,
                     **path_substitutions):
    """Save the match to the DR file."""

    matched_ids = pandas.read_csv(
        match_fname,
        sep=r'\s+',
        header=None,
        usecols=[0, len(catalogue_sources.columns) + 1],
        names=['catalogue_id', 'extracted_id'],
        comment='#'
    )

    catalogue_sources = catalogue_sources.index.ravel()
    extracted_sources = extracted_sources.index.ravel()

    extracted_sorter = numpy.argsort(extracted_sources)
    catalogue_sorter = numpy.argsort(catalogue_sources)
    match = numpy.empty([matched_ids.index.size, 2], dtype=int)
    match[:, 0] = catalogue_sorter[
        numpy.searchsorted(catalogue_sources,
                           matched_ids['catalogue_id'].ravel(),
                           sorter=catalogue_sorter)
    ]
    match[:, 1] = extracted_sorter[
        numpy.searchsorted(extracted_sources,
                           matched_ids['extracted_id'].ravel(),
                           sorter=extracted_sorter)
    ]
    dr_file.add_dataset(
        dataset_key='skytoframe.matched',
        data=match,
        **path_substitutions
    )


def save_trans_to_dr(trans_fname,
                     configuration,
                     header,
                     dr_file,
                     **path_substitutions):
    """Save the transformation to the DR file."""

    transformation, info = parse_anmatch_transformation(trans_fname)
    terms_expression = (r'O{order:d}{{'
                        r'(xi-{offset[0]!r})/{scale!r}'
                        r','
                        r'(eta-{offset[1]!r})/{scale!r}'
                        r'}}').format(**transformation)

    dr_file.add_dataset(
        dataset_key='skytoframe.coefficients',
        data=numpy.stack((transformation['dxfit'],
                          transformation['dyfit'])),
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.type',
        attribute_value=transformation['type'],
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.terms',
        attribute_value=terms_expression,
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.sky_center',
        attribute_value=numpy.array([info['2mass']['RA'],
                                     info['2mass']['DEC']]),
        **path_substitutions
    )
    for entry in ['residual', 'unitarity']:
        dr_file.add_attribute(
            attribute_key='skytoframe.' + entry,
            attribute_value=info[entry],
            **path_substitutions
        )
    for component, config_attribute in [
            ('srcextract', 'binning'),
            ('skytoframe', 'srcextract_filter'),
            ('skytoframe', 'sky_preprojection'),
            ('skytoframe', 'max_match_distance'),
            ('skytoframe', 'frame_center'),
            ('skytoframe', 'weights_expression')
    ]:
        if config_attribute == 'max_match_distance':
            value = configuration['max_srcmatch_distance']
        elif config_attribute == 'frame_center':
            value = (
                header['NAXIS1'] / 2.0,
                header['NAXIS2'] / 2.0
            )
        else:
            value = configuration[config_attribute]
        dr_file.add_attribute(
            component + '.cfg.' + config_attribute,
            value,
            **path_substitutions
        )


#TODO: Add catalogue query configuration to DR
def save_to_dr(match_fname,
               trans_fname,
               configuration,
               header,
               dr_file):
    """Save the solved astrometry to the given DR file."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in ['srcextract_version',
                             'catalogue_version',
                             'skytoframe_version']
    }
    catalogue_sources = read_catalogue(configuration['astrometry_catalogue'])
    extracted_sources = dr_file.get_sources(
        'srcextract.sources',
        'srcextract_column_name',
        srcextract_version=configuration['srcextract_version']
    )
    dr_file.add_sources(catalogue_sources,
                        'catalogue.columns',
                        'catalogue_column_name',
                        parse_ids=True,
                        ascii_columns=['ID', 'phqual', 'magsrcflag'],
                        **path_substitutions)

    save_match_to_dr(catalogue_sources,
                     extracted_sources,
                     match_fname,
                     dr_file,
                     **path_substitutions)
    save_trans_to_dr(trans_fname,
                     configuration,
                     header,
                     dr_file,
                     **path_substitutions)


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

    print('Solving: ' + repr(dr_fname))
    cat_ra_col, cat_dec_col = get_sky_coord_columns(
        configuration['astrometry_catalogue']
    )
    with DataReductionFile(dr_fname, 'r+') as dr_file:
        header = dr_file.get_frame_header()
        with TempAstrometryFiles() as (sources_fname, match_fname, trans_fname):
            x_col, y_col = create_sources_file(
                dr_file,
                sources_fname,
                configuration['srcextract_version']
            )
            for tweak in range(*configuration['anet_tweak_range']):
                try:
                    configuration['anet_tweak'] = tweak
                    command = [
                        'anmatch',
                        '--comment',
                        '--col-inp', '{0:d},{1:d}'.format(x_col + 1, y_col + 1),
                        '--input', sources_fname,
                        '--max-distance',
                        repr(configuration['max_srcmatch_distance']),
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
                                  **dict(header),
                                  x_col=x_col+1,
                                  y_col=y_col+1)
                    ]
                    subprocess.run(command, check=True)
                    print_file_contents(trans_fname, 'trans')
                    save_to_dr(match_fname,
                               trans_fname,
                               configuration,
                               header,
                               dr_file)
                    logging.debug('Found astrometric solution for %s',
                                  repr(dr_fname))
                    return
                except subprocess.CalledProcessError:
                    pass
    logging.error('Failed to find astrometric solution for %s',
                  repr(dr_fname))


def solve_astrometry(dr_collection, configuration):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    for dr_fname in dr_collection:
        solve_image(dr_fname, **configuration)


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    cmdline_config.update(
        binning=1,
        srcextract_filter='True',
        sky_preprojection='tan',
        weights_expression='1.0'
    )
    solve_astrometry(find_dr_fnames(cmdline_config.pop('dr_files'),
                                    cmdline_config.pop('astrometry_only_if')),
                     cmdline_config)
