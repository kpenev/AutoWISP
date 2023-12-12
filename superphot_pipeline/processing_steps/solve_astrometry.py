#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import logging
from multiprocessing import Queue, Process
import os
from traceback import format_exc

import numpy
from astropy import units
from astropy.io import fits
from general_purpose_python_modules.multiprocessing_util import setup_process

from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser,\
    ignore_progress
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline.astrometry import \
    estimate_transformation,\
    refine_transformation,\
    find_ra_dec_center
from superphot_pipeline.catalog import read_catalog_file, create_catalog_file
from superphot_pipeline import DataReductionFile
from superphot_pipeline import Evaluator

_logger = logging.getLogger(__name__)

input_type = 'dr'
fail_reasons = {
    'failed to converge': 1,
    'few matched': 2,
    'high rms': 3,
    'solve-field failed': 4,
    'other': 5
}


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=('' if args else input_type),
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
        default='MASTERS/astrometry_catalogue.ucac4',
        help='A file containing (approximately) all the same stars that '
        'were extracted from the frame for the area of the sky covered by the '
        'image. It is perferctly fine to include a larger area of sky and '
        'fainter brightness limit. Different brightness limits can then be '
        'imposed for each color channel using the ``--catalogue-filter`` '
        'argument. If the file does not exist one is automatically generated to'
        ' cover an area larger than the field of view by '
        '``--image-scale-factor``, centered on the (RA * cos(Dec), Dec) of the '
        'frame rounded to ``--catalog-pointing-precision`` fraction of the '
        'field of view, and to have the same density as the '
        '``--catalog-density-quantile`` of the extracted sources within the '
        'frames being processed.'
    )
    parser.add_argument(
        '--catalog-magnitude-expression',
        default='phot_g_mean_mag',
        help='An expression involving the catalogue columns that correlates as '
        'closely as possible with the brightness of the star in the images in '
        'units of magnitude. Only relevant if the catalog does not exist.'
    )
    parser.add_argument(
        '--catalog-density-quantile',
        type=float,
        default=0.9,
        help='The quantile within the number extracted sources of the separate '
        'frames to use when determining the faint limit of automatically '
        'generated catalogs. Only relevant if the catalog does not exist.'
    )
    parser.add_argument(
        '--catalog-density-scaling',
        type=float,
        default=1.2,
        help='The density of stars estimated from source extraction is '
        'multiplied by this factor to determine how many stars to include in '
        'the catalog. Only relevant if the catalog does not exist.'
    )
    parser.add_argument(
        '--catalog-pointing-precision',
        type=float,
        default=0.1,
        help='The precision with which to round the center of the frame to '
        'determine the center of the catalog to use.'
    )

    parser.add_argument(
        '--catalogue-filter', '--catalog-filter', '--cat-filter',
        metavar=('CHANNEL:EXPRESSION'),
        type=lambda e: e.split(':'),
        action='append',
        default=None,
        help='An expression to evaluate for each catalog source to determine '
        'if the source should be used for astrometry of a given channel. If '
        'filter for a given channel is not specified, the full catalog is used '
        'for that channel.'
    )
    parser.add_argument(
        '--catalogue-epoch', '--catalog-epoch', '--cat-epoch',
        type=str,
        default='(float(DATE[:4])+0.5) * units.yr',
        help='An expression to evaluate for each catalog source to determine '
        'the epoch to which to propagate star positions.'
    )

    parser.add_argument(
        '--frame-center-estimate',
        nargs=2,
        type=str,
        default=None,
        help='The approximate right ascention and declination of the center of '
        'the frame in degrees. Can be an expression involving header keywords. '
        'If not specified, the center of the catalog is used.'
    )
    parser.add_argument(
        '--frame-fov-estimate',
        nargs=2,
        type=str,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='Approximate field of view of the frame in degrees. Can be an '
        'expression involving header keywords. If not specified, the field of '
        'view of the catalog divided by ``--image-scale-factor`` is used.'
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
    if result['catalogue_filter'] is not None:
        result['catalogue_filter'] = dict(result['catalogue_filter'])
    return result


def print_file_contents(fname,
                        label):
    """Print the entire contenst of the given file."""

    print(80*'*')
    print(label.title() + ': ')
    print(80*'-')
    with open(fname, 'r', encoding="utf-8") as open_file:
        print(open_file.read())
    print(80*'-')


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
    catalogue_sources = read_catalog_file(
        configuration['astrometry_catalogue'].format_map(header)
    )

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



#TODO: Think of a way to split
#pylint: disable=too-many-locals
#pylint: disable=too-many-statements
#pylint: disable=too-many-branches
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
        configuration = prepare_configuration(configuration, header)

        result = {'dr_fname': dr_fname, 'fnum': header['FNUM'], 'saved': False}

        fov_estimate = max(*configuration['frame_fov_estimate'])

        sources = dr_file.get_sources(
            'srcextract.sources',
            'srcextract_column_name',
            srcextract_version=configuration['srcextract_version']
        )
        xy_extracted = numpy.zeros(
            (len(sources['x'].values)),
            dtype=[('x', '>f8'), ('y', '>f8')]
        )
        xy_extracted['x'] = sources['x'].values
        xy_extracted['y'] = sources['y'].values

        if transformation_estimate is None:
            dr_eval = Evaluator(header)
            transformation_estimate = {
                key: dr_eval(expression).to_value('deg')
                for key, expression in zip(
                    ['ra_cent', 'dec_cent'],
                    configuration['frame_center_estimate']
                )
            }
            (
                transformation_estimate['trans_x'],
                transformation_estimate['trans_y'],
                status
            ) = estimate_transformation(
                dr_file=dr_file,
                xy_extracted=xy_extracted,
                astrometry_order=configuration['astrometry_order'],
                tweak_order_range=configuration['tweak_order'],
                fov_range=(
                    fov_estimate / configuration['image_scale_factor'],
                    fov_estimate * configuration['image_scale_factor']
                ),
                **transformation_estimate,
                header=header
            )
            if status != 'success':
                result['fail_reason'] = status
                return result

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

        filter_expr = configuration['catalogue_filter']
        if filter_expr is not None:
            filter_expr = filter_expr.get(header['CLRCHNL'])
        catalog = read_catalog_file(
            configuration['astrometry_catalogue'].format_map(header),
            filter_expr=filter_expr
        )

        try:
            (
                trans_x,
                trans_y,
                cat_extracted_corr,
                res_rms,
                ratio,
                ra_cent,
                dec_cent,
                success
            ) = refine_transformation(
                xy_extracted=xy_extracted,
                catalog=catalog,
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
            result['fail_reason'] = fail_reasons['other']
            return result
        #pylint: enable=bare-except

        try:
            _logger.debug('RMS residual: %s', repr(res_rms))
            _logger.debug('Ratio: %s', repr(ratio))

            if ratio < configuration['min_match_fraction']:
                result['fail_reason'] = fail_reasons['few matched']
            elif res_rms > configuration['max_rms_distance']:
                result['fail_reason'] = fail_reasons['high rms']
            elif not success:
                result['fail_reason'] = fail_reasons[
                    'failed to converge'
                ]
            else:
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

    result['fail_reason'] = 'solve-field failed'
    _logger.error(
        'No Astrometry.net solution found in tweak range [%d, %d]',
        *configuration['tweak_order']
    )
    return result
#pylint: enable=too-many-locals
#pylint: enable=too-many-statements
#pylint: enable=too-many-branches


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


def prepare_configuration(configuration, dr_header):
    """Apply fallbacks to the configuration."""

    _logger.debug('Preparing configuration from: %s',
                  repr(configuration))
    result = configuration.copy()
    with fits.open(
            configuration['astrometry_catalogue'].format_map(dr_header)
    ) as cat_fits:
        catalogue_header = cat_fits[1].header

    if configuration['frame_center_estimate'] is None:
        result['frame_center_estimate'] = (catalogue_header['RA'],
                                           catalogue_header['DEC'])
    if configuration['frame_fov_estimate'] is None:
        result['frame_fov_estimate'] = (
            catalogue_header['WIDTH'] / configuration['image_scale_factor'],
            catalogue_header['HEIGHT'] / configuration['image_scale_factor']
        )
    else:
        dr_eval = Evaluator(dr_header)
        result['frame_fov_estimate'] = (
            dr_eval(
                configuration['frame_fov_estimate'][0]
            ).to_value('deg'),
            dr_eval(
                configuration['frame_fov_estimate'][1]
            ).to_value('deg')
        )

    result.update(
        binning=1,
        srcextract_filter='True',
        sky_preprojection='tan',
        weights_expression='1.0'
    )
    return result


def get_catalog_info(transformation_estimate, header, configuration):
    """Get the configuration of the catalog needed for this frame."""

    frame_center = find_ra_dec_center(
        (0.0, 0.0),
        transformation_estimate['trans_x'],
        transformation_estimate['trans_y,'],
        {
            coord: transformation_estimate[coord]
            for coord in ['ra_cent', 'dec_cent']
        },
        header['NAXIS1'] / 2.0,
        header['NAXIS2'] / 2.0,
        configuration['astrometry_order']
    )

    <++>

    to_generate = {}
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r') as dr_file:
            header = dr_file.get_frame_header()

            catalog_fname = configuration['astrometry_catalogue'].format_map(
                header
            )
            eval_header_expr = Evaluator(header)
            if os.path.exists(catalog_fname):
                continue

            catalog_fov = tuple(
                (
                    eval_header_expr(fov_expression)
                    *
                    configuration['image_scale_factor']
                )
                for fov_expression in configuration['frame_fov_estimate']
            )
            catalog_epoch = eval_header_expr(configuration['catalogue_epoch'])

            if catalog_fname not in to_generate:
                to_generate[catalog_fname] = {
                    'width': catalog_fov[0],
                    'height': catalog_fov[1],
                    'ra': [],
                    'dec': [],
                    'epoch': catalog_epoch,
                    'num_stars': []
                }
            assert (
                (
                    to_generate[catalog_fname]['width'],
                    to_generate[catalog_fname]['height']
                )
                ==
                catalog_fov
            )
            assert (
                to_generate[catalog_fname]['epoch']
                ==
                catalog_epoch
            )
            to_generate[catalog_fname]['ra'].append(
                eval_header_expr(
                    configuration['frame_center_estimate'][0]
                ).to_value(
                    units.deg
                )
            )
            to_generate[catalog_fname]['dec'].append(
                eval_header_expr(
                    configuration['frame_center_estimate'][1]
                ).to_value(
                    units.deg
                )
            )

            to_generate[catalog_fname]['num_stars'].append(
                dr_file.get_dataset_shape(
                    'srcextract.sources',
                    srcextract_version=configuration['srcextract_version'],
                    srcextract_column_name='x'
                )[0]
                *
                configuration['image_scale_factor']**2
                *
                configuration['catalog_density_scaling']
            )
    for catalog_properties in to_generate.values():
        for quantity in ['ra', 'dec', 'width', 'height']:
            catalog_properties[quantity] = numpy.array(
                catalog_properties[quantity]
            ) * units.deg
    return to_generate


def ensure_catalog(transformation_estimate, header, configuration):
    """Re-use or create astrometry catalog suitable for the given frame."""


def create_catalogs(configuration, dr_collection):
    """Create al catalogs required to find astrometry of given DR files."""

    to_generate = get_new_catalog_info(configuration, dr_collection)
    for catalog_fname, catalog_properties in to_generate.items():
        _logger.info('Creating catalog %s (%s)',
                     repr(catalog_fname),
                     repr(catalog_properties))

        create_catalog_file(
            catalog_fname=catalog_fname,
            ra=numpy.median(catalog_properties['ra']),
            dec=numpy.median(catalog_properties['dec']),
            width=catalog_properties['width'],
            height=catalog_properties['height'],
            max_objects=int(
                numpy.quantile(
                    catalog_properties['num_stars'],
                    configuration['catalog_density_quantile']
                )
            ),
            magnitude_expression=configuration['catalog_magnitude_expression'],
            epoch=catalog_properties['epoch'],
            magnitude_limit=None,
            verbose=True,
            columns=['source_id',
                     'ra',
                     'dec',
                     'pmra',
                     'pmdec',
                     'phot_g_n_obs',
                     'phot_g_mean_mag',
                     'phot_g_mean_flux',
                     'phot_g_mean_flux_error',
                     'phot_bp_n_obs',
                     'phot_bp_mean_mag',
                     'phot_bp_mean_flux',
                     'phot_bp_mean_flux_error',
                     'phot_rp_n_obs',
                     'phot_rp_mean_mag',
                     'phot_rp_mean_flux',
                     'phot_rp_mean_flux_error',
                     'phot_proc_mode',
                     'phot_bp_rp_excess_factor']
        )


#Could not think of good way to split
#pylint: disable=too-many-branches
#pylint: disable=too-many-locals
def solve_astrometry(dr_collection, configuration, mark_progress):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    #create_catalogs(configuration, dr_collection)
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
    _logger.debug('Starting astrometry on %d pending frames', len(pending))

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
            mark_progress(result['dr_fname'])
            if result['fnum'] in failed:
                if result['fnum'] not in pending:
                    pending[result['fnum']] = []
                pending[result['fnum']].extend(
                    [f[0] for f in failed[result['fnum']]]
                )
                del failed[result['fnum']]

            for dr_fname in pending.get(result['fnum'], []):
                task_queue.put((dr_fname, result['raw_transformation']))
                num_queued += 1

            if result['fnum'] in pending:
                del pending[result['fnum']]
        else:
            if result['fnum'] not in failed:
                failed[result['fnum']] = []
            failed[result['fnum']].append(
                (result['dr_fname'], result['fail_reason'])
            )
            if pending.get(result['fnum'], False):
                task_queue.put((pending[result['fnum']].pop(), None))
                num_queued += 1

            if not pending.get(result['fnum'], True):
                del pending[result['fnum']]

    for fnum_failed in failed.values():
        for dr_fname, reason in fnum_failed:
            mark_progress(dr_fname, reason)
            _logger.critical(
                'Failed astrometry for DR file %s: %s',
                dr_fname,
                [fail_key for fail_key, fail_reason in fail_reasons.items()
                 if fail_reason == reason][0]
            )
    _logger.debug('Stopping astrometry solving processes.')
    for process in workers:
        task_queue.put('STOP')

    for process in workers:
        process.join()
#pylint: enable=too-many-locals
#pylint: enable=too-many-branches


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='manage', **cmdline_config)

    solve_astrometry(find_dr_fnames(cmdline_config.pop('dr_files'),
                                    cmdline_config.pop('astrometry_only_if')),
                     cmdline_config,
                     ignore_progress)
    _logger.debug('Solving astrometry for %d DR files', len(dr_collection))
