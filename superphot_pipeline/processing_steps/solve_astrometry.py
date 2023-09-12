#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import logging
import subprocess
from multiprocessing import Queue, Process
from tempfile import mkstemp
import os
from traceback import format_exc

import numpy
from astropy.io import fits
from general_purpose_python_modules.multiprocessing_util import setup_process

from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline.astrometry import \
    estimate_transformation,\
    refine_transformation
from superphot_pipeline.catalog import read_catalog_file
from superphot_pipeline import DataReductionFile
from superphot_pipeline import Evaluator

_logger = logging.getLogger(__name__)

def parse_command_line(*args):
    """Return the parsed command line arguments."""

    if args:
        inputtype = ''
    else:
        inputtype = 'dr'

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=inputtype,
        processing_step='astrometry',
        inputs_help_extra='The DR files must already contain extracted sources',
        add_component_versions=('srcextract', 'catalogue', 'skytoframe'),
        allow_parallel_processing=True
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
        help='A file containing (approximately) all the same stars that '
        'were extracted from the frame for the area of the sky covered by the '
        'image. It is perferctly fine to include a larger area of sky and '
        'fainter brightness limit. Different brightness limits are then imposed'
        'for each color channel using the ``--catalogue-filter`` argument.'
    )
    parser.add_argument(
        '--catalogue-filter', '--catalog-filter', '--cat-filter',
        metavar=('CHANNEL:EXPRESSION'),
        type=lambda e: e.split(':'),
        action='append',
        default=[],
        help='An expression to evaluate for each catalog source to determine '
        'if the source should be used for astrometry of a given channel. If '
        'filter for a given channel is not specified, the full catalog is used '
        'for that channel.'
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
        '--tweak-order',
        type=int,
        nargs=2,
        default=(2, 5),
        help='Range of tweak arguments to solve-field to try.'
    )
    parser.add_argument(
        '--trans-threshold',
        type=float,
        default=1e-3,
        help='The threshold for the difference of two consecutive '
             'transformations'
    )
    parser.add_argument(
        '--max-astrom-iter',
        type=int,
        default=20,
        help='The maximum number of iterations the astrometry solution can '
        'pass.'
    )
    parser.add_argument(
        '--image-scale-factor',
        type=float,
        default=1.3,
        help='The image scale factor to add to the given frames'
    )
    parser.add_argument(
        '--min-match-fraction',
        type=float,
        default=0.8,
        help='The minimum fraction of extracted sources that must be matched to'
        ' a catalogue soure for the solution to be considered valid.'
    )
    parser.add_argument(
        '--max-rms-distance',
        type=float,
        default=0.5,
        help='The maximum RMS distance between projected and extracted '
        'positions for the astrometry solution to be considered valid.'
    )

    result = parser.parse_args(*args)
    result['catalogue_filter'] = dict(result['catalogue_filter'])
    return result

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
            fname = getattr(self, file_type + '_fname')
            if os.path.exists(fname):
                os.remove(fname)

def print_file_contents(fname,
                        label):
    """Print the entire contenst of the given file."""

    print(80*'*')
    print(label.title() + ': ')
    print(80*'-')
    with open(fname, 'r', encoding="utf-8") as open_file:
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
        dtype=[('x', '>f8'), ('y', '>f8')]
    )
    xy_extracted['x'] = sources['x'].values
    xy_extracted['y'] = sources['y'].values

    return xy_extracted

def save_trans_to_dr(*,
                     trans_x,
                     trans_y,
                     ra_cent,
                     dec_cent,
                     res_rms,
                     configuration,
                     header,
                     dr_file,
                     **path_substitutions):
    """Save the transformation to the DR file."""

    terms_expression = f'O{configuration["astrometry_order"]:d}{{xi, eta}}'

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
def save_to_dr(*,
               cat_extracted_corr,
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
    catalogue_sources = read_catalog_file(configuration['astrometry_catalogue'])

    dr_file.add_sources(catalogue_sources,
                        'catalogue.columns',
                        'catalogue_column_name',
                        parse_ids=False,
                        **path_substitutions)
    dr_file.add_dataset(
        dataset_key='skytoframe.matched',
        data=cat_extracted_corr,
        **path_substitutions
    )
    save_trans_to_dr(trans_x=trans_x,
                     trans_y=trans_y,
                     ra_cent=ra_cent,
                     dec_cent=dec_cent,
                     res_rms=res_rms,
                     configuration=configuration,
                     header=header,
                     dr_file=dr_file,
                     **path_substitutions)


def transformation_to_raw(trans_x, trans_y, header, in_place=False):
    """Convert the transformation coefficients to pre-channel split coords."""

    if not in_place:
        trans_x = numpy.copy(trans_x)
        trans_y = numpy.copy(trans_y)
    trans_x[0] += header['CHNLXOFF']
    trans_x *= header['CHNLXSTP']
    trans_y[0] += header['CHNLYOFF']
    trans_y *= header['CHNLYSTP']
    return trans_x, trans_y

def transformation_from_raw(trans_x, trans_y, header, in_place=False):
    """Convert the transformation coefficients to pre-channel split coords."""

    if not in_place:
        trans_x = numpy.copy(trans_x)
        trans_y = numpy.copy(trans_y)

    trans_x /= header['CHNLXSTP']
    trans_x[0] -= header['CHNLXOFF']

    trans_y /= header['CHNLYSTP']
    trans_y[0] -= header['CHNLYOFF']

    return trans_x, trans_y



#pylint: disable=too-many-locals
def solve_image(dr_fname,
                transformation_estimate=None,
                **configuration):
    """
    Find the astrometric transformation for a single image and save to DR file.

    Args:
        dr_fname(str):    The name of the data reduction file containing the
            extracted sources from the frame and that will be updated with the
            newly solved astrometry.

        transformation_estimate(None or (matrix, matrix)):    Estimate of the
            transformations x(xi, eta) and y(xi, eta) for the raw frame (i.e.
            before channel splitting) that will be refined. If ``None``,
            ``solve_field`` from astrometry.net is used to find iniitial
            estimates.

        configuration:    Parameters defining how astrometry is to be fit.

    Returns:
        trans_x(2D numpy array):
            the coefficients of the x(xi, eta) transformation converted to RAW
            image coordinates (i.e. before channel splitting)

        trans_y(2D numpy array):
            the coefficients of the y(xi, eta) transformation converted to RAW
            image coordinates (i.e. before channel splitting)

        ra_cent(float): the RA center around which the above transformation
            applies

        dec_cent(float): the Dec center around which the above transformation
            applies
    """

    _logger.debug('Solving: %s %s transformation estimate.',
                  repr(dr_fname),
                  ('with' if transformation_estimate else 'without'))
    with DataReductionFile(dr_fname, 'r+') as dr_file:

        header = dr_file.get_frame_header()

        result = {'dr_fname': dr_fname, 'fnum': header['FNUM'], 'saved': False}

        catalogue = read_catalog_file(
            configuration['astrometry_catalogue'],
            filter_expr=configuration['catalogue_filter'].get(header['CLRCHNL'])
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
            for tweak in range(*configuration['tweak_order']):
                solve_field_command = [
                    'solve-field',
                    sources_fname,
                    '--corr', corr_fname,
                    '--width', str(header['NAXIS1']),
                    '--height', str(header['NAXIS2']),
                    '--tweak-order', str(tweak),
                    '--match', 'none',
                    '--wcs', 'none',
                    '--index-xyls', 'none',
                    '--rdls', 'none',
                    '--solved', 'none',
                    '--axy', axy_fname,
                    '--no-plots',
                    '--scale-low', repr(fov_estimate
                                        /
                                        configuration['image_scale_factor']),
                    '--scale-high', repr(fov_estimate
                                         *
                                         configuration['image_scale_factor']),
                    '--overwrite'
                ]
                try:
                    subprocess.run(solve_field_command, check=True)
                except subprocess.SubprocessError:
                    _logger.critical("solve-field failed with error:\n%s",
                                     format_exc())
                    continue

                if not os.path.isfile(corr_fname):
                    _logger.critical("Correspondence file %s not created.",
                                     repr(corr_fname))
                    return result

                with fits.open(corr_fname, mode='readonly') as corr:
                    field_corr = corr[1].data[:]

                if field_corr.size > ((tweak+1)*(tweak+2))//2:

                    initial_corr = numpy.zeros(
                        (field_corr['field_x'].shape),
                        dtype=[('x', '>f8'),
                               ('y', '>f8'),
                               ('RA', '>f8'),
                               ('Dec', '>f8')]
                    )

                    initial_corr['x'] = field_corr['field_x']
                    initial_corr['y'] = field_corr['field_y']
                    initial_corr['RA'] = field_corr['index_ra']
                    initial_corr['Dec'] = field_corr['index_dec']

                    if transformation_estimate is None:
                        transformation_estimate = {
                            key: float(Evaluator(header)(expression))
                            for key, expression in zip(
                                ['ra_cent', 'dec_cent'],
                                configuration['frame_center_estimate']
                            )
                        }

                        (
                            transformation_estimate['trans_x'],
                            transformation_estimate['trans_y']
                        ) = estimate_transformation(
                            initial_corr=initial_corr,
                            tweak_order=tweak,
                            astrometry_order=configuration['astrometry_order'],
                            ra_cent=transformation_estimate['ra_cent'],
                            dec_cent=transformation_estimate['dec_cent'],
                        )
                    else:
                        (
                            transformation_estimate['trans_x'],
                            transformation_estimate['trans_y']
                        ) = transformation_from_raw(
                            transformation_estimate['trans_x'],
                            transformation_estimate['trans_y'],
                            header,
                            True
                        )


                    _logger.debug('Using transformation estimate: %s',
                                  repr(transformation_estimate))

                    try:
                        (
                            trans_x,
                            trans_y,
                            cat_extracted_corr,
                            res_rms,
                            ratio,
                            ra_cent,
                            dec_cent
                        ) = refine_transformation(
                            xy_extracted=xy_extracted,
                            catalogue=catalogue,
                            x_frame=header['NAXIS1'],
                            y_frame=header['NAXIS2'],
                            astrometry_order=configuration['astrometry_order'],
                            max_srcmatch_distance=configuration[
                                'max_srcmatch_distance'
                            ],
                            max_iterations=configuration['max_astrom_iter'],
                            trans_threshold=configuration['trans_threshold'],
                            **transformation_estimate,
                        )
                    #pylint: disable=bare-except
                    except:
                        _logger.critical(
                            'Failed to find solution to DR file %s:\n%s',
                            dr_fname,
                            format_exc()
                        )
                        return result
                    #pylint: enable=bare-except

                    try:
                        # print('trans_x:'+repr(trans_x))
                        # print('trans_y:'+repr(trans_y))
                        # print('matched_sources:'+repr(cat_extracted_corr))
                        _logger.debug('RMS residual: %s', repr(res_rms))
                        _logger.debug('Ratio: %s', repr(ratio))

                        if (
                                ratio > configuration['min_match_fraction']
                                and
                                res_rms < configuration['max_rms_distance']
                        ):
                            _logger.info(
                                'Succesful astrometry solution found for %s:',
                                dr_fname
                            )
                            save_to_dr(cat_extracted_corr=cat_extracted_corr,
                                       trans_x=trans_x,
                                       trans_y=trans_y,
                                       ra_cent=ra_cent,
                                       dec_cent=dec_cent,
                                       res_rms=res_rms,
                                       configuration=configuration,
                                       header=header,
                                       dr_file=dr_file)
                            result['saved'] = True

                            transformation_to_raw(trans_x,
                                                  trans_y,
                                                  header,
                                                  True)
                            result['raw_transformation'] = {
                                'ra_cent':  ra_cent,
                                'dec_cent': dec_cent,
                                'trans_x':  trans_x,
                                'trans_y': trans_y
                            }
                        return result

                    #pylint: disable=bare-except
                    except:
                        _logger.critical(
                            'Failed to save found astrometry solution to '
                            'DR file %s:\n%s',
                            dr_fname,
                            format_exc()
                        )
                        return result
                    #pylint: enable=bare-except

            _logger.error(
                'No Astrometry.net solution found in tweak range [%d, %d]',
                *configuration['tweak_order']
            )
            return result
#pylint: enable=too-many-locals


def astrometry_process(task_queue, result_queue, configuration):
    """Run pending astrometry tasks from the queue in process."""

    setup_process(task='solve', **configuration)
    _logger.info('Starting astrometry solving process.')
    for dr_fname, transformation_estimate in iter(task_queue.get,
                                                  'STOP'):
        result_queue.put(
            solve_image(
                dr_fname,
                transformation_estimate,
                **configuration
            )
        )
    _logger.debug('Astrometry solving process finished.')


#Could not think of good way to split
#pylint: disable=too-many-branches
def solve_astrometry(dr_collection, configuration):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    pending = {}
    failed = {}
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r+') as dr_file:
            header = dr_file.get_frame_header()
            if header['FNUM'] not in pending:
                pending[header['FNUM']] = [dr_fname]
            else:
                pending[header['FNUM']].append(dr_fname)

    task_queue = Queue()
    result_queue = Queue()

    num_queued = 0
    for fnum_pending in pending.values():
        task_queue.put((fnum_pending.pop(), None))
        num_queued += 1

    workers = [
        Process(target=astrometry_process,
                args=(task_queue, result_queue, configuration))
        for _ in range(configuration['num_parallel_processes'])
    ]

    for process in workers:
        process.start()

    while pending or num_queued:
        _logger.debug('Pending: %s', repr(pending))
        _logger.debug('Number scheduled: %d', num_queued)
        result = result_queue.get()
        num_queued -= 1

        if 'raw_transformation' in result:
            if not result['saved']:
                _logger.critical(
                    'Failed to save astrometry solution to DR file %s.',
                    result['dr_fname']
                )
                break
            if result['fnum'] in failed:
                if result['fnum'] not in pending:
                    pending[result['fnum']] = []
                pending[result['fnum']].extend(failed[result['fnum']])
                del failed[result['fnum']]

            for dr_fname in pending.get(result['fnum'], []):
                task_queue.put((dr_fname, result['raw_transformation']))
                num_queued += 1

            if result['fnum'] in pending:
                del pending[result['fnum']]
        else:
            if result['fnum'] not in failed:
                failed[result['fnum']] = []
            failed[result['fnum']].append(result['dr_fname'])
            if pending.get(result['fnum'], False):
                task_queue.put((pending[result['fnum']].pop(), None))
                num_queued += 1

            if not pending.get(result['fnum'], True):
                del pending[result['fnum']]

    _logger.debug('Stopping astrometry solving processes.')
    for process in workers:
        task_queue.put('STOP')

    for process in workers:
        process.join()
#pylint: enable=too-many-branches


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    cmdline_config.update(
        binning=1,
        srcextract_filter='True',
        sky_preprojection='tan',
        weights_expression='1.0'
    )
    setup_process(task='manage', **cmdline_config)
    solve_astrometry(find_dr_fnames(cmdline_config.pop('dr_files'),
                                    cmdline_config.pop('astrometry_only_if')),
                     cmdline_config)
