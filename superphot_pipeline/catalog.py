"""Utilities for querying catalogs for astrometry."""

import numpy
import pandas
from astropy import units
from astropy.io import fits
from astroquery.gaia import GaiaClass, conf

from superphot_pipeline import Evaluator

class SuperPhotGaia(GaiaClass):
    """Extend queries with condition and sorting."""

    def query_object_filtered(self,
                              *,
                              ra,
                              dec,
                              width,
                              height,
                              condition,
                              order_by,
                              epoch=None,
                              columns=None,
                              order_dir='ASC',
                              verbose=False):
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

            condition(str):    Condition the returned sources must satisfy
                (typically imposes a brightness limit)

            columns(iterable):    List of columns to select from the catalog. If
                unspecified, all columns will be returned.

            order_by(str):    How should the stars be ordered.

            order_dir(str):    Should the order be ascending (``'ASC'``) or
                descending (``'DESC'``).

        Returns:
            astropy Table:
                The result of the query.
        """

        def get_result(query, add_propagated):
            """Get and format the result as specified by user."""

            job = self.launch_job_async(query, verbose=verbose)
            result = job.get_results()
            result.rename_column('ra', 'ra_orig')
            result.rename_column('dec', 'dec_orig')

            if epoch is not None and add_propagated:
                propagated = {coord: numpy.empty(len(result))
                              for coord in ['ra', 'dec']}
                for i, pos in enumerate(result['propagated']):
                    for coord, value_str in zip(
                            ['ra', 'dec'],
                            pos.strip().strip('()').split(',')
                    ):
                        propagated[coord][i] = (float(value_str)
                                                *
                                                180.0 / numpy.pi)

                result.remove_column('propagated')
                for coord in add_propagated:
                    result.add_column(propagated[coord], name=coord)

            return result


        if columns is None:
            columns = "*"
        else:
            add_propagated = []
            for coord in ['ra', 'dec']:
                try:
                    columns.remove('prop_' + coord)
                    add_propagated.append(coord)
                except ValueError:
                    pass

            columns = ', '.join(map(str, columns))

        if '*' in columns:
            add_propagated = ['ra', 'dec']

        if epoch is not None:
            epoch = epoch.to_value(units.yr)
            columns = (
                'EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, '
                f'radial_velocity, ref_epoch, {epoch}) AS propagated, '
            ) + columns

        ra = ra.to_value(units.deg)
        dec = dec.to_value(units.deg)
        width = width.to_value(units.deg)
        height = height.to_value(units.deg)
        table_name = self.MAIN_GAIA_TABLE or conf.MAIN_GAIA_TABLE

        return get_result(
            f"""
             SELECT
             {columns}
             FROM {table_name}
             WHERE
                 1 = CONTAINS(
                     POINT(
                         'ICRS',
                         {self.MAIN_GAIA_TABLE_RA},
                         {self.MAIN_GAIA_TABLE_DEC}
                     ),
                     BOX(
                         'ICRS',
                         {ra},
                         {dec},
                         {width},
                         {height}
                     )
                 )
                 AND
                 ({condition})
             ORDER BY
                 {order_by}
                 {order_dir}
            """,
            add_propagated
        )


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

        if query_kwargs.get('columns', False):
            query_kwargs['columns'].append(
                f'({magnitude_expression}) AS magnitude'
            )
        else:
            query_kwargs['columns'] = [f'({magnitude_expression}) AS magnitude',
                                       '*']

        if 'order_by' not in query_kwargs:
            query_kwargs['order_by'] = 'magnitude'
            query_kwargs['order_dir'] = 'ASC'

        try:
            min_mag, max_mag = magnitude_limit
            condition = (f'({magnitude_expression}) > {min_mag} AND '
                         f'({magnitude_expression}) < {max_mag}')
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
    for colname in ['DESIGNATION',
                    'phot_variable_flag',
                    'datalink_url',
                    'epoch_photometry_url',
                    'libname_gspphot']:
        try:
            query[colname] = query[colname].astype(str)
        except KeyError:
            pass
    print('Column names: ' + '\n\t'.join(query.colnames))
    print(80*'*')
    print('Dtype: '
          +
          '\n\t'.join([repr(e) for e in query.dtype.fields.items()]))

    query.write(
        catalog_fname,
        format='fits',
        overwrite=overwrite
    )


def read_catalog_file(catalog_fname,
                      filter_expr=None,
                      sort_expr=None):
    """
    Read a catalog FITS file.

    Args:
        catalog_fname(str):    Name of the catalog file to read.

    Returns:
        pandas.DataFrame:
            The catalog information as columns.
    """

    with fits.open(catalog_fname) as cat_fits:
        result = pandas.DataFrame(cat_fits[1].data)
    result.set_index('source_id', inplace=True)

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

    if sort_expr is None:
        return result

    return result.iloc[numpy.argsort(sort_val)]


if __name__ == '__main__':
    create_catalog_file(
        'test_gaia.fits',
        ra=118.0 * units.deg,
        dec=2.6 * units.deg,
        width=1.8 * units.deg,
        height=1.3 * units.deg,
        epoch=2023.5 * units.yr,
        magnitude_expression='phot_g_mean_mag - 5',
        magnitude_limit=6,
        verbose=True,
        overwrite=True
    )
    print(
        repr(
            read_catalog_file(
                'test_gaia.fits',
                filter_expr='libname_gspphot == "PHOENIX"'
            )
        )
    )
