"""Restore the fill value of lightcurve frame numbers lost to a FLOAT column.

A dataset's ``replace_nonfinite`` is also its HDF5 fill value: what an entry
nothing was written to reads as. For the frame numbers of lightcurves, an
integer dataset, it is the largest 32-bit unsigned integer, 4294967295. The
column holding it is a float one, since it holds the fill values of every
dataset, and until revision 0009 it was a 32-bit FLOAT on MySQL and MariaDB.
That cannot hold 4294967295 and stored 4294967296 instead, which a 32-bit
``numpy.uint`` (as on Windows) cannot hold either. 0009 widened the column
to DOUBLE, which holds it exactly, but could not recover what was lost; this
restores it. SQLite stored it exactly all along.

This revision changes rows, not the schema. The names and values below are a
copy of what project creation writes, made so that this revision keeps doing
what it did if that changes.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0019_restore_lc_fnum_fill"
down_revision = "0018_projhome_paths"
branch_labels = None
depends_on = None

PRODUCT = "light_curve"
DATASET = "fitsheader.fnum"

#: The fill value as a 32-bit FLOAT stored it, and as it should be.
FILL_VALUES = (4294967296, 4294967295)


def upgrade():
    """Restore the fill value where a FLOAT column rounded it."""

    connection = alembic.op.get_bind()
    metadata = sqlalchemy.MetaData()
    datasets, versions, products = (
        sqlalchemy.Table(name, metadata, autoload_with=connection)
        for name in (
            "hdf5_datasets",
            "hdf5_structure_versions",
            "hdf5_products",
        )
    )
    connection.execute(
        datasets.update()
        .where(
            datasets.c.pipeline_key == DATASET,
            datasets.c.replace_nonfinite == FILL_VALUES[0],
            datasets.c.hdf5_structure_version_id.in_(
                sqlalchemy.select(versions.c.id)
                .join(products, versions.c.hdf5_product_id == products.c.id)
                .where(products.c.pipeline_key == PRODUCT)
            ),
        )
        .values(replace_nonfinite=FILL_VALUES[1])
    )


def downgrade():
    """
    Deliberately not undone.

    The rounded value was never intended, and every version works with the
    restored one.
    """
