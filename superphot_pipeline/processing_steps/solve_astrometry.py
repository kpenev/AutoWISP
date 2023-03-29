#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import subprocess
from tempfile import mkstemp
import os

import numpy
from astropy.io import fits

from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser,\
    read_catalogue
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline import astrometry
from superphot_pipeline import DataReductionFile
from superphot_pipeline import Evaluator

#pylint:disable=R0913
#pylint:disable=R0914
#pylint:disable=R0915
#pylint:disable=C0103

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
        type=str,
        help='The approximate right ascention and declination of the center of '
        'the frame in degrees. Can be an expression involving header keywords.'
    )
    parser.add_argument(
        '--frame-fov-estimate',
        required=True,
        type=str,
        help='Approximate field of view of the frame in degrees. Can be an '
        'expression involving header keywords.'
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
        '--trans-threshold',
        type=float,
        default=1e-3,
        help='The threshold for the difference of two consecutive '
             'transformations'
    )
    parser.add_argument(
        '--image-scale-factor',
        type=float,
        default=1.3,
        help='The image scale factor to add to the given frames'
    )
    return parser.parse_args()

class TempAstrometryFiles:
    """Context manager for the temporary files needed for astrometry."""

    def __init__(self):
        """Create all required temporary files."""

        self._file_types = ['sources', 'corr', 'axy']
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

def print_file_contents(fname,
                        label):
    """Print the entire contenst of the given file."""

    print(80*'*')
    print(label.title() + ': ')
    print(80*'-')
    with open(fname, 'r') as open_file:
        print(open_file.read())
    print(80*'-')

def create_sources_file(dr_file, sources_fname, srcextract_version):
    """Create a FITS BinTable file with the given name containing
    the extracted sources.

    Returns: an array containing x-y extracted sources
    """

    sources = dr_file.get_sources(
        'srcextract.sources',
        'srcextract_column_name',
        srcextract_version=srcextract_version
    )
    x_extracted = fits.Column(name='x', format='D', array=sources['x'].values)
    y_extracted = fits.Column(name='y', format='D', array=sources['y'].values)
    xyls = fits.BinTableHDU.from_columns([x_extracted, y_extracted])
    xyls.writeto(sources_fname)

    xy_extracted = numpy.zeros(
        (len(sources['x'].values)),
        dtype=[('x','>f8'),('y','>f8')]
    )
    xy_extracted['x']=sources['x'].values
    xy_extracted['y'] = sources['y'].values

    return xy_extracted

def save_trans_to_dr(trans_x,
                     trans_y,
                     ra_cent,
                     dec_cent,
                     res_rms,
                     configuration,
                     header,
                     dr_file,
                     **path_substitutions):
    """Save the transformation to the DR file."""

    terms_expression = 'O{order:d}{{xi, eta}}'\
        .format(order=configuration['astrometry_order'])

    dr_file.add_dataset(
        dataset_key='skytoframe.coefficients',
        data=numpy.stack((trans_x.flatten(), trans_y.flatten())),
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.type',
        attribute_value='polynomial',
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.terms',
        attribute_value=terms_expression,
        **path_substitutions
    )
    dr_file.add_attribute(
        attribute_key='skytoframe.sky_center',
        attribute_value=numpy.array([ra_cent, dec_cent]),
        **path_substitutions
    )
    #TODO: need to add and figure out unitarity
    # for entry in ['residual', 'unitarity']:
    #     dr_file.add_attribute(
    #         attribute_key='skytoframe.' + entry,
    #         attribute_value=res_rms,
    #         **path_substitutions
    #     )
    dr_file.add_attribute(
        attribute_key='skytoframe.residual',
        attribute_value=res_rms,
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
def save_to_dr(cat_extracted_corr,
               trans_x,
               trans_y,
               ra_cent,
               dec_cent,
               res_rms,
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

    dr_file.add_sources(catalogue_sources,
                        'catalogue.columns',
                        'catalogue_column_name',
                        parse_ids=True,
                        ascii_columns=['ID', 'phqual', 'magsrcflag'],
                        **path_substitutions)
    dr_file.add_dataset(
        dataset_key='skytoframe.matched',
        data=cat_extracted_corr,
        **path_substitutions
    )
    save_trans_to_dr(trans_x,
                     trans_y,
                     ra_cent,
                     dec_cent,
                     res_rms,
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
    catalogue=read_catalogue(
        configuration['astrometry_catalogue']
    )
    with DataReductionFile(dr_fname, 'r+') as dr_file:
        header = dr_file.get_frame_header()
        center_ra_dec = tuple(
            float(Evaluator(header)(expression))
            for expression in configuration['frame_center_estimate']
        )
        fov_estimate = float(
            Evaluator(header)(configuration['frame_fov_estimate'])
        )
        with TempAstrometryFiles() as (sources_fname,
                                       corr_fname,
                                       axy_fname):
            xy_extracted = create_sources_file(
                dr_file,
                sources_fname,
                configuration['srcextract_version']
            )
            #pylint:disable=line-too-long
            solve_field_command = [
                'solve-field',
                sources_fname,
                '--corr', corr_fname,
                '--width', str(header['NAXIS1']),
                '--height', str(header['NAXIS2']),
                '--match', 'none',
                '--wcs', 'none',
                '--index-xyls', 'none',
                '--rdls', 'none',
                '--solved', 'none',
                '--axy', axy_fname,
                '--no-plots',
                '--scale-low',repr(fov_estimate/configuration['image_scale_factor']),
                '--scale-high',repr(fov_estimate*configuration['image_scale_factor']),
                '--overwrite'
            ]
            #pylint:enable=line-too-long
            subprocess.run(solve_field_command, check=True)

            assert os.path.isfile(corr_fname)

            with fits.open(corr_fname, mode='readonly') as corr:
                field_corr=corr[1].data[:]

            initial_corr=numpy.zeros(
                (field_corr['field_x'].shape),
                dtype=[('x','>f8'),
                       ('y','>f8'),
                       ('RA','>f8'),
                       ('Dec','>f8')]
            )

            initial_corr['x']=field_corr['field_x']
            initial_corr['y'] = field_corr['field_y']
            initial_corr['RA'] = field_corr['index_ra']
            initial_corr['Dec'] = field_corr['index_dec']

            trans_x, \
                trans_y, \
                cat_extracted_corr, \
                res_rms, \
                ratio, \
                ra_cent, \
                dec_cent = astrometry.solve(
                initial_corr=initial_corr,
                xy_extracted=xy_extracted,
                catalogue=catalogue,
                astrometry_order=configuration['astrometry_order'],
                max_srcmatch_distance=configuration['max_srcmatch_distance'],
                trans_threshold=configuration['trans_threshold'],
                ra_cent=center_ra_dec[0],
                dec_cent=center_ra_dec[1],
                x_frame=header['NAXIS1'],
                y_frame=header['NAXIS2'],
            )

            # print('trans_x:'+repr(trans_x))
            # print('trans_y:'+repr(trans_y))
            # print('matched_sources:'+repr(cat_extracted_corr))
            print('res_rms:'+repr(res_rms))
            print('ratio:'+repr(ratio))

            save_to_dr(cat_extracted_corr=cat_extracted_corr,
                       trans_x=trans_x,
                       trans_y=trans_y,
                       ra_cent=ra_cent,
                       dec_cent=dec_cent,
                       res_rms=res_rms,
                       configuration=configuration,
                       header=header,
                       dr_file=dr_file)


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
