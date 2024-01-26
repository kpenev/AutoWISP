#!/usr/bin/env python3
"""Utilities for querying catalogs for astrometry."""

from os import path, makedirs
import logging

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


def create_catalog_file(catalog_fname, overwrite=False, **query_kwargs):
    """
    Create a catalog FITS file from a Gaia query.

    Args:
        catalog_fname(str):    Name of the catalog file to create.

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
            path.dirname(catalog_fname)
            and
            not path.exists(path.dirname(catalog_fname))
    ):
        makedirs(path.dirname(catalog_fname))
    query.write(
        catalog_fname,
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
