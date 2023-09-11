"""Utilities for querying catalogs for astrometry."""

import numpy
from astropy import units
from astroquery.gaia import GaiaClass, conf

class SuperPhotGaia(GaiaClass):
    """Extend query_object with condition and sorting."""

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
                              order_dir='DESC',
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
        """

        def get_result(query, add_propagated):
            """Get and format the result as specified by user."""

            job = self.launch_job_async(query, verbose=verbose)
            result = job.get_results()
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
                    result.add_column(propagated[coord], name='prop_' + coord)

            return result


        if columns is None:
            columns = "*"
            add_propagated = ['ra', 'dec']
        else:
            add_propagated = []
            for coord in ['ra', 'dec']:
                try:
                    columns.remove('prop_' + coord)
                    add_propagated.append(coord)
                except ValueError:
                    pass

            columns = ','.join(map(str, columns))

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

gaia = SuperPhotGaia()

if __name__ == '__main__':
    gaia.query_object_filtered(
        ra=118.0 * units.deg,
        dec=2.6 * units.deg,
        width=1.8 * units.deg,
        height=1.3 * units.deg,
        epoch=2023.5 * units.yr,
        condition='phot_g_mean_mag < 11',
        order_by='phot_g_mean_mag',
        verbose=True
    ).write(
        'test_gaia.txt',
        format='ascii.fixed_width',
    )
