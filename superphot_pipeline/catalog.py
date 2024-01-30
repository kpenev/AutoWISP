#!/usr/bin/env python3
"""Utilities for querying catalogs for astrometry."""

from os import path, makedirs
import logging
from hashlib import md5

from configargparse import ArgumentParser, DefaultsFormatter
import numpy
import pandas
from astropy import units
from astropy.io import fits
from astroquery.gaia import GaiaClass, conf

from superphot_pipeline import Evaluator
from superphot_pipeline.astrometry.map_projections import \
    gnomonic_projection,\
    inverse_gnomonic_projection
from superphot_pipeline.astrometry import find_ra_dec
if __name__ == '__main__':
    from matplotlib import pyplot

_logger = logging.getLogger(__name__)

class SuperPhotGaia(GaiaClass):
    """Extend queries with condition and sorting."""

    def _get_result(self,
                    query,
                    add_propagated,
                    verbose=False):
        """Get and format the result as specified by user."""

        job = self.launch_job_async(query, verbose=verbose)
        result = job.get_results()
        _logger.debug('Gaia query result: %s', repr(result))
        _logger.debug('Gaia query result columns: %s', repr(result.colnames))
        if result.colnames == ['num_obj']:
            return result['num_obj'][0]
        result.rename_column('ra', 'RA_orig')
        result.rename_column('dec', 'Dec_orig')

        if add_propagated:
            propagated = {coord: numpy.empty(len(result))
                          for coord in ['RA', 'Dec']}
            for i, pos in enumerate(result['propagated']):
                for coord, value_str in zip(
                        ['RA', 'Dec'],
                        pos.strip().strip('()').split(',')
                ):
                    propagated[coord][i] = (float(value_str)
                                            *
                                            180.0 / numpy.pi)

            result.remove_column('propagated')
            for coord in add_propagated:
                result.add_column(propagated[coord], name=coord)

        return result


    #pylint: disable=too-many-locals
    def query_object_filtered(self,
                              *,
                              ra,
                              dec,
                              width,
                              height,
                              order_by,
                              condition=None,
                              epoch=None,
                              columns=None,
                              order_dir='ASC',
                              max_objects=None,
                              verbose=False,
                              count_only=False):
        """
        Get GAIA sources within a box satisfying given condition (ADQL).

        Args:
            ra:   The RA of the center of the box to query, with units.

            dec:    The declination of the center of the box to query,
                with units.

            width:    The width of the box (half goes on each side of ``ra``),
                with units.

            height:    The height of the box (half goes on each side of
                ``dec``), with units.

            order_by(str):    How should the stars be ordered.

            condition(str):    Condition the returned sources must satisfy
                (typically imposes a brightness limit)

            epoch:    The epoch to propagate the positions to. If unspecified,
                no propagation will be done.

            columns(iterable):    List of columns to select from the catalog. If
                unspecified, all columns will be returned.

            order_dir(str):    Should the order be ascending (``'ASC'``) or
                descending (``'DESC'``).

            max_objects(int):    Maximum number of objects to return.

            verbose(bool):    Use verbose mode when submitting querries to GAIA.

            count_only(bool):    If ``True``, only the number of objects is
                returned without actually fetching the data.

        Returns:
            astropy Table:
                The result of the query.
        """

        if count_only:
            columns = 'COUNT(*) AS num_obj'
        elif columns is None:
            columns = "*"
        else:
            add_propagated = []
            for coord in ['RA', 'Dec']:
                try:
                    add_propagated.append(coord)
                except ValueError:
                    pass

            columns = ', '.join(map(str, columns))

        if '*' in columns and not count_only:
            add_propagated = ['RA', 'Dec']

        if epoch is not None:
            epoch = epoch.to_value(units.yr)
            columns = (
                'EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, '
                f'radial_velocity, ref_epoch, {epoch}) AS propagated, '
            ) + columns

        corners = numpy.empty(shape=(4,),
                                     dtype=[('RA', float), ('Dec', float)])
        corners_xi_eta = numpy.empty(shape=(4,),
                                     dtype=[('xi', float), ('eta', float)])
        width = width.to_value(units.deg)
        height = height.to_value(units.deg)
        corners_xi_eta[0] = (-width/2, -height/2)
        corners_xi_eta[1] = (-width/2, height/2)
        corners_xi_eta[2] = (width/2, height/2)
        corners_xi_eta[3] = (width/2, -height/2)

        inverse_gnomonic_projection(corners,
                                    corners_xi_eta,
                                    RA=ra.to_value(units.deg),
                                    Dec=dec.to_value(units.deg))

        table_name = self.MAIN_GAIA_TABLE or conf.MAIN_GAIA_TABLE

        select = 'SELECT'
        if max_objects is not None:
            select += f' TOP {max_objects}'
        query_str = f"""
            {select}
            {columns}
            FROM {table_name}
            WHERE
                1 = CONTAINS(
                    POINT(
                        {self.MAIN_GAIA_TABLE_RA},
                        {self.MAIN_GAIA_TABLE_DEC}
                    ),
                    POLYGON(
                        {corners[0]['RA']},
                        {corners[0]['Dec']},
                        {corners[1]['RA']},
                        {corners[1]['Dec']},
                        {corners[2]['RA']},
                        {corners[2]['Dec']},
                        {corners[3]['RA']},
                        {corners[3]['Dec']}
                    )
                )
        """
        if condition is not None:
            query_str += f"""
                AND
                ({condition})
            """

        if not count_only:
            query_str += f"""
                ORDER BY
                    {order_by}
                    {order_dir}
            """

        return self._get_result(
            query_str,
            epoch is not None and not count_only and add_propagated,
            verbose
        )
    #pylint: enable=too-many-locals


    def query_brightness_limited(self,
                                 *,
                                 magnitude_expression,
                                 magnitude_limit,
                                 **query_kwargs):
        """
        Get sources within a box and a range of magnitudes.

        Args:
            magnitude_expression:    Expression for the relevant magnitude
                involving Gaia columns.

            magnitude_limit:    Either upper limit or lower and upper limit on
                the magnitude defined by ``magnitude_expression``.

            **query_kwargs:    Arguments passed directly to
                `query_object_filtered()`.

        Returns:
            astropy Table:
                The result of the query.
        """
        _logger.debug('Querying Gaia for sources with magnitude: %s, '
                      'limits: %s, and kwargs: %s',
                      repr(magnitude_expression),
                      repr(magnitude_limit),
                      repr(query_kwargs))

        if query_kwargs.get('columns', False):
            query_kwargs['columns'] = (
                query_kwargs['columns']
                +
                [
                    f'({magnitude_expression}) AS magnitude'
                ]
            )
        else:
            query_kwargs['columns'] = [f'({magnitude_expression}) AS magnitude',
                                       '*']

        if 'order_by' not in query_kwargs:
            query_kwargs['order_by'] = 'magnitude'
            query_kwargs['order_dir'] = 'ASC'

        if magnitude_limit is not None:
            try:
                min_mag, max_mag = magnitude_limit
                condition = (f'({magnitude_expression}) > {min_mag} AND '
                             f'({magnitude_expression}) < {max_mag}')
            except ValueError:
                condition = f'{magnitude_expression} < {magnitude_limit[0]}'
            except TypeError:
                condition = f'{magnitude_expression} < {magnitude_limit}'

            if 'condition' in query_kwargs:
                query_kwargs['condition'] = (
                    f'({query_kwargs["condition"]}) AND ({condition})'
                )
            else:
                query_kwargs['condition'] = condition

        return self.query_object_filtered(**query_kwargs)


gaia = SuperPhotGaia()
#This comes from astroquery (not in our control)
#pylint: disable=invalid-name
gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
#pylint: enable=invalid-name


def create_catalog_file(fname, overwrite=False, **query_kwargs):
    """
    Create a catalog FITS file from a Gaia query.

    Args:
        fname(str):    Name of the catalog file to create.

        **query_kwargs:    Arguments passed directly to
            `gaia.query_brightness_limited()`.
    """

    query = gaia.query_brightness_limited(**query_kwargs)
    if query_kwargs.get('count_only', False):
        print('Number of sources: ', repr(query))
        return
    for colname in ['DESIGNATION',
                    'phot_variable_flag',
                    'datalink_url',
                    'epoch_photometry_url',
                    'libname_gspphot']:
        try:
            query[colname] = query[colname].astype(str)
        except KeyError:
            pass

    query.meta['CATALOG'] = 'Gaia'
    query.meta['CATVER'] = gaia.MAIN_GAIA_TABLE
    for k in ['ra', 'dec', 'width', 'height']:
        query.meta[k.upper()] = query_kwargs[k].to_value(units.deg)

    query.meta['EPOCH'] = (
        query_kwargs['epoch'].to_value(units.yr)
        if query_kwargs['epoch'] is not None else
        None
    )
    query.meta['MAGEXPR'] = query_kwargs['magnitude_expression']
    if query_kwargs.get('magnitude_limit') is not None:
        try:
            (
                query.meta['MAGMIN'],
                query.meta['MAGMAX']
            ) = query_kwargs['magnitude_limit']
        except ValueError:
            query.meta['MAGMAX'] = query_kwargs['magnitude_limit'][0]
        except TypeError:
            query.meta['MAGMAX'] = query_kwargs['magnitude_limit']

    if (
            path.dirname(fname)
            and
            not path.exists(path.dirname(fname))
    ):
        makedirs(path.dirname(fname))
    query.write(
        fname,
        format='fits',
        overwrite=overwrite
    )


def read_catalog_file(cat_fits,
                      filter_expr=None,
                      sort_expr=None,
                      return_metadata=False,
                      add_gnomonic_projection=False):
    """
    Read a catalog FITS file.

    Args:
        cat_fits(str, or opened FITS file):    The file to read.

    Returns:
        pandas.DataFrame:
            The catalog information as columns.
    """

    if isinstance(cat_fits, str):
        with fits.open(cat_fits) as opened_cat_fits:
            return read_catalog_file(opened_cat_fits,
                                     filter_expr,
                                     sort_expr,
                                     return_metadata,
                                     add_gnomonic_projection)

    fixed_dtype = cat_fits[1].data.dtype.newbyteorder('=')
    result = pandas.DataFrame.from_records(
        cat_fits[1].data.astype(fixed_dtype),
        index='source_id'
    )
    if return_metadata or add_gnomonic_projection:
        metadata = cat_fits[1].header

    cat_eval = Evaluator(result)
    if sort_expr is not None:
        sort_val = cat_eval(sort_expr)

    if filter_expr is not None:
        print('Filter expression: ' + repr(filter_expr))
        filter_val = cat_eval(filter_expr)
        print('Filter val: ' + repr(filter_val))
        filter_val = filter_val.astype(bool)
        result = result.loc[filter_val]

        if sort_expr is not None:
            sort_val = sort_val[filter_val]

    if sort_expr is not None:
        result = result.iloc[numpy.argsort(sort_val)]

    if add_gnomonic_projection:
        if 'xi' not in result.columns:
            assert 'eta' not in result.columns
            for colname in ['xi', 'eta']:
                result.insert(len(result.columns), colname, numpy.nan)
            gnomonic_projection(result,
                                result,
                                RA=metadata['RA'],
                                Dec=metadata['Dec'])

    if return_metadata:
        return result, metadata
    return result


def parse_command_line():
    """Return configuration of catalog to create."""

    parser = ArgumentParser(
        description='Create a catalog file from a Gaia query.',
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--ra',
        type=float,
        default=118.0,
        help='The right ascention (deg) of the center of the field to query.'
    )
    parser.add_argument(
        '--dec',
        type=float,
        default=2.6,
        help='The declination (deg) of the center of the field to query.'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=17.0,
        help='The width (deg) of the field to query (along RA direction).'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=17.0,
        help='The height (deg) of the field to query (along dec direction).'
    )
    parser.add_argument(
        '--epoch', '-t',
        type=float,
        default=None,
        help='The epoch for proper motion corrections in years. If not '
        'specified, positions are not propagated.'
    )
    parser.add_argument(
        '--magnitude-expression',
        default='phot_g_mean_mag',
        help='The expression to use as the relevant magnitude estimate.'
    )
    parser.add_argument(
        '--magnitude-limit',
        nargs='+',
        type=float,
        default=12.0,
        help='Either maximum magnitude or minimum and maximum magnitude limits '
        'to impose.'
    )
    parser.add_argument(
        '--extra-condition',
        default=None,
        help='An extra condition to impose on the selected sources.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print out information about the query being executed.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing catalog file.'
    )
    parser.add_argument(
        '--catalog-fname',
        default='gaia.fits',
        help='The name of the catalog file to create.'
    )
    parser.add_argument(
        '--count-only',
        action='store_true',
        help='Only count the number of sources in the query.'
    )
    parser.add_argument(
        '--columns',
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
        '--show-stars',
        action='store_true',
        help='Show the stars in the catalog on a 3-D plot of the sky.'
    )
    return parser.parse_args()


def get_max_abs_corner_xi_eta(header,
                              transformation_estimate,
                              center=None):
    """Return the max absolute values of the frame corners xi and eta."""

    max_xi = max_eta = 0
    if center is None:
        center = {'RA': transformation_estimate['ra_cent'],
                  'Dec': transformation_estimate['dec_cent']}
    for x in [0.0, float(header['NAXIS1'])]:
        for y in [0.0, float(header['NAXIS2'])]:
            rhs = numpy.array([
                x - float(transformation_estimate['trans_x'][0]),
                y - float(transformation_estimate['trans_y'][0])
            ])
            matrix = numpy.array([
                transformation_estimate['trans_x'].ravel()[1:3],
                transformation_estimate['trans_y'].ravel()[1:3]
            ])
            _logger.debug('Solving %s * [xi, eta] = %s',
                          repr(matrix),
                          repr(rhs))

            sky_coords = find_ra_dec(
                xieta_guess = numpy.linalg.solve(matrix, rhs),
                trans_x=transformation_estimate['trans_x'],
                trans_y=transformation_estimate['trans_y'],
                radec_cent={'RA': transformation_estimate['ra_cent'],
                            'Dec': transformation_estimate['dec_cent']},
                frame_x=x,
                frame_y=y
            )
            projected_coords = {}
            gnomonic_projection(
                sky_coords,
                projected_coords,
                **center
            )
            max_xi = max(max_xi, abs(projected_coords['xi']))
            max_eta = max(max_eta, abs(projected_coords['eta']))

    return max_xi, max_eta


def get_catalog_info(transformation_estimate, header, configuration):
    """Get the configuration of the catalog needed for this frame."""

    _logger.debug('Creating catalog info from: %s', repr(configuration))
    frame_center = find_ra_dec(
        (0.0, 0.0),
        transformation_estimate['trans_x'],
        transformation_estimate['trans_y'],
        {
            coord: transformation_estimate[coord.lower() + '_cent']
            for coord in ['RA', 'Dec']
        },
        header['NAXIS1'] / 2.0,
        header['NAXIS2'] / 2.0
    )
    pointing_precision = configuration['pointing_precision'] * units.deg

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
        configuration['magnitude_expression']
    )

    if configuration['max_magnitude'] is None:
        assert configuration['min_magnitude'] is None
        catalog_info['magnitude_limit'] = None
    else:
        catalog_info['magnitude_limit'] = (
            (configuration['max_magnitude'],)
            if configuration['min_magnitude'] is None else
            (
                configuration['min_magnitude'],
                configuration['max_magnitude']
            )
        )

    eval_expression = Evaluator(header)
    eval_expression.symtable.update(catalog_info)

    trans_fov = get_max_abs_corner_xi_eta(header, transformation_estimate)
    _logger.debug('From transformation estimate half FOV is: %s',
                  repr(trans_fov))
    frame_fov_estimate = tuple(
        numpy.round(
            (
                max(
                    2.0 * trans_fov[i] * units.deg,
                    configuration['frame_fov_estimate'][i]
                ) + pointing_precision
            )
            /
            (configuration['fov_precision'] * units.deg)
        ) * configuration['fov_precision'] * units.deg
        for i in range(2)
    )

    _logger.debug('Determining catalog query size from: '
                  'frame_fov_estimate=%s, '
                  'image_scale_factor=%s',
                  repr(frame_fov_estimate),
                  repr(1.0 + configuration['fov_safety_margin']))
    catalog_info['width'], catalog_info['height'] = (
        fov_size * (1.0 + configuration['fov_safety_margin'])
        for fov_size in frame_fov_estimate
    )
    catalog_info['epoch'] = eval_expression(configuration['epoch'])

    catalog_info['columns'] = configuration['columns']

    get_checksum = md5()
    for cfg in sorted(catalog_info.items()):
        get_checksum.update(repr(cfg).encode('ascii'))

    catalog_info['fname'] = (
        configuration['fname'].format(
            **dict(header),
            **catalog_info,
            checksum=get_checksum.hexdigest()
        )
    )
    _logger.debug('Created catalog info: %s', repr(catalog_info))

    return catalog_info, frame_center


#No god way to simplify
#pylint: disable=too-many-branches
def ensure_catalog(transformation_estimate,
                   header,
                   configuration,
                   lock):
    """Re-use or create astrometry catalog suitable for the given frame."""

    catalog_info, frame_center = get_catalog_info(transformation_estimate,
                                                  header,
                                                  configuration)
    with lock:
        if path.exists(catalog_info['fname']):
            with fits.open(catalog_info['fname']) as cat_fits:
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
                        f'Catalog {catalog_info["fname"]} '
                        f'has epoch {catalog_header["EPOCH"]!r}, '
                        f'but {catalog_info["epoch"]!r} is needed'
                    )

                if (
                    catalog_header['MAGEXPR']
                    !=
                    catalog_info['magnitude_expression']
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} has '
                        f'magnitude expression {catalog_header["MAGEXPR"]!r} '
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
                        f'Catalog {catalog_info["fname"]} excludes '
                        f'sources brighter than {catalog_header["MAGMIN"]!r} '
                        f'but {catalog_info["magnitude_limit"][0]!r} are '
                        'required.'
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
                        f'Catalog {catalog_info["fname"]} excludes '
                        f'sources fainter than {catalog_header["MAGMAX"]!r} but'
                        f' {catalog_info["magnitude_limit"][-1]!r} are '
                        'required.'
                    )

                if (
                    catalog_header['WIDTH']
                    <
                    catalog_info['width'].to_value(units.deg)
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} width '
                        f'{catalog_header["WIDTH"]!r} is less than the required'
                        f' {catalog_info["width"]!r}'
                    )
                if (
                    catalog_header['HEIGHT']
                    <
                    catalog_info['height'].to_value(units.deg)
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} height '
                        f'{catalog_header["HEIGHT"]!r} is less than the '
                        f'required {catalog_info["height"]!r}'
                    )

                if (
                    (catalog_header['RA'] - frame_center['RA']) * units.deg
                    *
                    numpy.cos(catalog_header['DEC'] * units.deg)
                    >
                    configuration['pointing_precision'] * units.deg
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} center RA '
                        f'{catalog_header["RA"]!r} is too far from the '
                        f'required RA={frame_center["RA"]!r}'
                    )

                if (
                    (catalog_header['DEC'] - frame_center['Dec']) * units.deg
                    >
                    configuration['pointing_precision'] * units.deg
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} center Dec '
                        f'{catalog_header["DEC"]!r} is too far from the '
                        f'required Dec={frame_center["Dec"]!r}'
                    )

                filter_expr = configuration['filter']
                if filter_expr is None or header['CLRCHNL'] not in filter_expr:
                    filter_expr = []
                else:
                    filter_expr = ['(' + filter_expr[header['CLRCHNL']] + ')']


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
                    filter_expr.append(
                        f'(magnitude > {catalog_info["magnitude_limit"][0]!r})'
                    )
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
                #pylint: enable=too-many-boolean-expressions

                return read_catalog_file(
                    cat_fits,
                    filter_expr=(' and '.join(filter_expr) if filter_expr
                                 else None),
                    return_metadata=True
                )

        del catalog_info['ra_ind']
        del catalog_info['dec_ind']

        create_catalog_file(**catalog_info, verbose=True)
        return read_catalog_file(catalog_info['fname'],
                                 return_metadata=True)
#pylint: enable=too-many-branches


def check_catalog_coverage(header,
                           transformation_estimate,
                           catalog_header,
                           safety_margin):
    """
    Return True iff te catalog covers the frame fully including a safety margin.

    Args:
        header(dict-like):    The header of the frame being astrometried.

        transformation_estimate(dict):    The current best fit transformation.
            Should contain entries for ``'trans_x'``, ``'trans_y'``,
            ``'ra_cent'`` and ``'dec_cent'``.

        catalog_header(dict-like):    The header of the catalog being used for
            astrometry.

        safety_margin(float):    The absolute values of xi and eta
            (relative to the catalog center) corresponding to the corners of the
            frame increased by this fraction must be smaller than the field of
            view of the catalogfor the frame to be considered covered.

    Returns:
        bool:
            Whether the catalog covers the frame fully including the safety
            margin.
    """

    width, height = get_max_abs_corner_xi_eta(
        header,
        transformation_estimate,
        {'RA': catalog_header['RA'], 'Dec': catalog_header['DEC']}
    )
    factor = 2.0 * (1.0 + safety_margin)
    width *= factor
    height *= factor
    return width < catalog_header['WIDTH'] and height < catalog_header['HEIGHT']


def show_stars(catalog_fname):
    """Show the stars in the catalog on a 3-D plot of the sky."""

    phi, theta = numpy.mgrid[0.0 : numpy.pi / 2.0 : 10j,
                             0.0 : 2.0*numpy.pi: 10j]
    sphere_x = numpy.sin(phi) * numpy.cos(theta)
    sphere_y = numpy.sin(phi) * numpy.sin(theta)
    sphere_z = numpy.cos(phi)


    fig = pyplot.figure()
    axes = fig.add_subplot(111, projection='3d')
    pyplot.gca().plot_surface(sphere_x,
                              sphere_y,
                              sphere_z,
                              rstride=1,
                              cstride=1,
                              color='c',
                              alpha=0.6,
                              linewidth=1)

    stars = read_catalog_file(catalog_fname)

    stars_x = (numpy.cos(numpy.radians(stars['Dec']))
               *
               numpy.cos(numpy.radians(stars['RA'])))
    stars_y = (numpy.cos(numpy.radians(stars['Dec']))
               *
               numpy.sin(numpy.radians(stars['RA'])))
    stars_z = numpy.sin(numpy.radians(stars['Dec']))

    axes.scatter(stars_x, stars_y, stars_z, color="k", s=20)


    pyplot.show()


def main(config):
    """Avoid polluting global namespace."""

    kwargs = {
        'ra': config.ra * units.deg,
        'dec': config.dec * units.deg,
        'width': config.width * units.deg,
        'height': config.height * units.deg,
        'epoch': config.epoch * units.yr if config.epoch is not None else None,
        'magnitude_expression': config.magnitude_expression,
        'magnitude_limit': config.magnitude_limit,
        'columns': config.columns,
        'verbose': config.verbose,
        'overwrite': config.overwrite,
        'count_only': config.count_only
    }
    if config.extra_condition is not None:
        kwargs['condition'] = config.extra_condition
    create_catalog_file(config.catalog_fname, **kwargs)
    if config.show_stars and not config.count_only:
        show_stars(config.catalog_fname)


if __name__ == '__main__':
    main(parse_command_line())
