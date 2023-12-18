#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import logging
from multiprocessing import Queue, Process, Lock
import os
from traceback import format_exc
from hashlib import md5

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
        '--astrometry-catalog', '--astrometry-catalogue', '--cat',
        default='Gaia/{checksum:s}.fits',
        help='A file containing (approximately) all the same stars that '
        'were extracted from the frame for the area of the sky covered by the '
        'image. It is perferctly fine to include a larger area of sky and '
        'fainter brightness limit. Different brightness limits can then be '
        'imposed for each color channel using the ``--catalog-filter`` '
        'argument. If the file does not exist one is automatically generated to'
        ' cover an area larger than the field of view by '
        '``--image-scale-factor``, centered on the (RA * cos(Dec), Dec) of the '
        'frame rounded to ``--catalog-pointing-precision``, and to have '
        'magnitude range set by. The filename can be a format string which will'
        ' be substituted with the any header keywords or configuration for the '
        'query. It may also include ``{checksum}`` which will be replaced with '
        'the MD5 checksum of the parameters defining the query.'
    )
    parser.add_argument(
        '--catalog-magnitude-expression',
        default='phot_g_mean_mag',
        help='An expression involving the catalog columns that correlates as '
        'closely as possible with the brightness of the star in the images in '
        'units of magnitude. Only relevant if the catalog does not exist.'
    )
    parser.add_argument(
        '--catalog-max-magnitude',
        type=float,
        default=12.0,
        help='The faintest magnitude to include in the catalog.'
    )
    parser.add_argument(
        '--catalog-min-magnitude',
        type=float,
        default=None,
        help='The brightest magnitude to include in the catalog.'
    )
    parser.add_argument(
        '--catalog-pointing-precision',
        type=float,
        default=0.1,
        help='The precision with which to round the center of the frame to '
        'determine the center of the catalog to use in degrees.'
    )

    parser.add_argument(
        '--catalog-filter', '--catalogue-filter', '--cat-filter',
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
        '--catalog-epoch', '--catalogue-epoch', '--cat-epoch',
        type=str,
        default='(float(DATE[:4])+0.5) * units.yr',
        help='An expression to evaluate for each catalog source to determine '
        'the epoch to which to propagate star positions.'
    )
    parser.add_argument(
        '--catalog-columns', '--catalogue-columns', '--cat-columns',
        type=str,
        nargs='+',
        default=['source_id',
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
                 'phot_bp_rp_excess_factor'],
        help='The columns to include in the catalog file. Use \'*\' to include '
        'everything.'
    )

    parser.add_argument(
        '--frame-center-estimate',
        nargs=2,
        type=str,
        default=('RA * units.deg', 'DEC_MNT * units.deg'),
        help='The approximate right ascention and declination of the center of '
        'the frame in degrees. Can be an expression involving header keywords. '
        'If not specified, the center of the catalog is used.'
    )
    parser.add_argument(
        '--frame-fov-estimate',
        nargs=2,
        type=str,
        default=('10.0 * units.deg', '15.0 * units.deg'),
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
        ' a catalog soure for the solution to be considered valid.'
    )
    parser.add_argument(
        '--max-rms-distance',
        type=float,
        default=0.5,
        help='The maximum RMS distance between projected and extracted '
        'positions for the astrometry solution to be considered valid.'
    )

    result = parser.parse_args(*args)
    if result['catalog_filter'] is not None:
        result['catalog_filter'] = dict(result['catalog_filter'])
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

#TODO: Add catalog query configuration to DR
def save_to_dr(*,
               cat_extracted_corr,
               trans_x,
               trans_y,
               ra_cent,
               dec_cent,
               res_rms,
               configuration,
               header,
               dr_file,
               catalog):
    """Save the solved astrometry to the given DR file."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in ['srcextract_version',
                             'catalogue_version',
                             'skytoframe_version']
    }

    dr_file.add_sources(catalog,
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
                *,
                catalog_lock,
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

        filter_expr = configuration['catalog_filter']
        if filter_expr is not None:
            filter_expr = filter_expr.get(header['CLRCHNL'])
        catalog = ensure_catalog(transformation_estimate,
                                 header,
                                 configuration,
                                 catalog_lock)

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
                           dr_file=dr_file,
                           catalog=catalog)
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


def astrometry_process(task_queue, result_queue, configuration, catalog_lock):
    """Run pending astrometry tasks from the queue in process."""

    setup_process(task='solve', **configuration)
    _logger.info('Starting astrometry solving process.')
    for dr_fname, transformation_estimate in iter(task_queue.get,
                                                  'STOP'):
        result_queue.put(
            solve_image(
                dr_fname,
                transformation_estimate,
                catalog_lock=catalog_lock,
                **configuration
            )
        )
    _logger.debug('Astrometry solving process finished.')


def prepare_configuration(configuration, dr_header):
    """Apply fallbacks to the configuration."""

    _logger.debug('Preparing configuration from: %s',
                  repr(configuration))
    result = configuration.copy()

    dr_eval = Evaluator(dr_header)
    result['frame_fov_estimate'] = tuple(
        dr_eval(expr) for expr in configuration['frame_fov_estimate']
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
        transformation_estimate['trans_y'],
        {
            coord: transformation_estimate[coord.lower() + '_cent']
            for coord in ['RA', 'Dec']
        },
        header['NAXIS1'] / 2.0,
        header['NAXIS2'] / 2.0,
        configuration['astrometry_order']
    )
    pointing_precision = configuration['catalog_pointing_precision'] * units.deg

    catalog_info = {
        'dec_ind': int(
            numpy.round(frame_center['Dec'] * units.deg / pointing_precision)
        )
    }
    catalog_info['dec'] = pointing_precision * catalog_info['dec_ind']

    catalog_info['ra_ind'] = int(
        numpy.round(
            (
                frame_center['RA'] * units.deg
                *
                numpy.cos(catalog_info['dec'])
            )
            /
            pointing_precision
        )
    )
    catalog_info['ra'] = (
        pointing_precision * catalog_info['ra_ind']
        /
        numpy.cos(catalog_info['dec'])
    )
    catalog_info['magnitude_expression'] = (
        configuration['catalog_magnitude_expression']
    )

    if configuration['catalog_max_magnitude'] is None:
        assert configuration['catalog_min_magnitude'] is None
        catalog_info['magnitude_limit'] = None
    else:
        catalog_info['magnitude_limit'] = (
            (configuration['catalog_max_magnitude'],)
            if configuration['catalog_min_magnitude'] is None else
            (
                configuration['catalog_min_magnitude'],
                configuration['catalog_max_magnitude']
            )
        )

    eval_expression = Evaluator(header)
    eval_expression.symtable.update(catalog_info)

    _logger.debug('Determining catalog query size from: '
                  'frame_fov_estimate=%s, '
                  'image_scale_factor=%s',
                  repr(configuration['frame_fov_estimate']),
                  repr(configuration['image_scale_factor']))
    catalog_info['width'], catalog_info['height'] = (
        fov_expression * configuration['image_scale_factor']
        for fov_expression in configuration['frame_fov_estimate']
    )
    catalog_info['epoch'] = eval_expression(configuration['catalog_epoch'])

    catalog_info['columns'] = configuration['catalog_columns']

    get_checksum = md5()
    for cfg in sorted(catalog_info.items()):
        get_checksum.update(repr(cfg).encode('ascii'))

    catalog_info['catalog_fname'] = (
        configuration['astrometry_catalog'].format(
            **dict(header),
            **catalog_info,
            checksum=get_checksum.hexdigest()
        )
    )
    return catalog_info, frame_center

def ensure_catalog(transformation_estimate,
                   header,
                   configuration,
                   lock):
    """Re-use or create astrometry catalog suitable for the given frame."""

    catalog_info, frame_center = get_catalog_info(transformation_estimate,
                                                  header,
                                                  configuration)
    lock.acquire()
    if os.path.exists(catalog_info['catalog_fname']):
        with fits.open(catalog_info['catalog_fname']) as cat_fits:
            catalog_header = cat_fits[1].header
            #pylint: disable=too-many-boolean-expressions
            if (
                    numpy.abs(
                        catalog_header['EPOCH']
                        -
                        catalog_info['epoch'].to_value('yr')
                    ) > 0.25
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} '
                    f'has epoch {catalog_header["EPOCH"]!r}, '
                    f'but {catalog_info["epoch"]!r} is needed'
                )

            if (
                catalog_header['MAGEXPR']
                !=
                catalog_info['magnitude_expression']
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} '
                    f'has magnitude expression {catalog_header["MAGEXPR"]!r} '
                    f'instead of {catalog_info["magnitude_expression"]!r}'
                )

            if (
                catalog_header.get('MAGMIN') is not None
                and
                len(catalog_info['magnitude_limit']) == 2
                and
                (
                    catalog_header['MAGMIN']
                    >
                    catalog_info['magnitude_limit'][0]
                )
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} excludes sources '
                    f'brighter than {catalog_header["MAGMIN"]!r} but '
                    f'{catalog_info["magnitude_limit"][0]!r} are required.'
                )

            if (
                catalog_header.get('MAGMAX') is not None
                and
                (
                    catalog_header['MAGMAX']
                    <
                    catalog_info['magnitude_limit'][-1]
                )
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} excludes sources '
                    f'fainter than {catalog_header["MAGMAX"]!r} but '
                    f'{catalog_info["magnitude_limit"][-1]!r} are required.'
                )

            if (
                catalog_header['WIDTH']
                <
                catalog_info['width'].to_value(units.deg)
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} width '
                    f'{catalog_header["WIDTH"]!r} is less than the required '
                    f'{catalog_info["width"]!r}'
                )
            if (
                catalog_header['HEIGHT']
                <
                catalog_info['height'].to_value(units.deg)
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} height '
                    f'{catalog_header["HEIGHT"]!r} is less than the required '
                    f'{catalog_info["height"]!r}'
                )

            if (
                (catalog_header['RA'] - frame_center['RA']) * units.deg
                *
                numpy.cos(catalog_header['DEC'] * units.deg)
                >
                configuration['catalog_pointing_precision'] * units.deg
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} center RA '
                    f'{catalog_header["RA"]!r} is too far from the '
                    f'required RA={frame_center["RA"]!r}'
                )

            if (
                (catalog_header['DEC'] - frame_center['Dec']) * units.deg
                >
                configuration['catalog_pointing_precision'] * units.deg
            ):
                raise RuntimeError(
                    f'Catalog {catalog_info["catalog_fname"]} center Dec '
                    f'{catalog_header["DEC"]!r} is too far from the '
                    f'required Dec={frame_center["Dec"]!r}'
                )

            #pylint: enable=too-many-boolean-expressions
            filter_expr = (
                ['(' + configuration['catalog_filter'] + ')']
                if configuration['catalog_filter'] else
                []
            )
            if (
                len(catalog_info['magnitude_limit']) == 2
                and
                (
                    catalog_header.get('MAGMIN') is None
                    or
                    catalog_header['MAGMIN']
                    <
                    catalog_info['magnitude_limit'][0]
                )
            ):
                filter_expr = [
                    f'(magnitude > {catalog_info["magnitude_limit"][0]!r})'
                ]
            if(
                catalog_header.get('MAGMAX') is None
                or
                (
                    catalog_header['MAGMAX']
                    >
                    catalog_info['magnitude_limit'][-1]
                )
            ):
                filter_expr.append(
                    '(magnitude < {catalog_info["magnitude_limit"][-1]!r})'
                )

            lock.release()
            return read_catalog_file(
                cat_fits,
                filter_expr=(' and '.join(filter_expr) if filter_expr
                             else None)
            )

    del catalog_info['ra_ind']
    del catalog_info['dec_ind']

    create_catalog_file(**catalog_info, verbose=True)
    lock.release()
    return read_catalog_file(catalog_info['catalog_fname'])


#Could not think of good way to split
#pylint: disable=too-many-branches
def manage_astrometry(pending, task_queue, result_queue, mark_progress):
    """Manege solving all frames until they solve or fail hopelessly."""

    num_queued = 0
    for fnum_pending in pending.values():
        task_queue.put((fnum_pending.pop(), None))
        num_queued += 1

    failed = {}

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
#pylint: enable=too-many-branches

def solve_astrometry(dr_collection, configuration, mark_progress):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    _logger.debug('Solving astrometry for %d DR files', len(dr_collection))
    #create_catalogs(configuration, dr_collection)
    pending = {}
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r+') as dr_file:
            header = dr_file.get_frame_header()
            if header['FNUM'] not in pending:
                pending[header['FNUM']] = [dr_fname]
            else:
                pending[header['FNUM']].append(dr_fname)

    task_queue = Queue()
    result_queue = Queue()

    catalog_lock=Lock()
    workers = [
        Process(target=astrometry_process,
                args=(task_queue, result_queue, configuration, catalog_lock))
        for _ in range(configuration['num_parallel_processes'])
    ]

    for process in workers:
        process.start()
    _logger.debug('Starting astrometry on %d pending frames', len(pending))

    manage_astrometry(pending, task_queue, result_queue, mark_progress)

    _logger.debug('Stopping astrometry solving processes.')
    for process in workers:
        task_queue.put('STOP')

    for process in workers:
        process.join()


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='manage', **cmdline_config)

    solve_astrometry(
        list(find_dr_fnames(cmdline_config.pop('dr_files'),
                            cmdline_config.pop('astrometry_only_if'))),
        cmdline_config,
        ignore_progress
    )
