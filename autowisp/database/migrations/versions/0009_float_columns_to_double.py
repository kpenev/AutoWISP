"""Widen every 32-bit FLOAT column to DOUBLE.

SQLAlchemy's ``Float`` becomes a 32-bit ``FLOAT`` on MySQL and MariaDB but a
64-bit ``REAL`` on SQLite, so the same column held different precision
depending on the backend. Every value AutoWISP puts in these columns is a
Python float, i.e. a double, so ``Double`` is what they should always have
been.

Storage was only half of it. MySQL renders a ``FLOAT`` to about six
significant digits in the text protocol, and the client parses that string
back -- so even a value stored exactly came back rounded. That is how the
"no data" sentinel in ``hdf5_datasets.replace_nonfinite`` reached the
pipeline as -1.70141e38 instead of -1.7014117331926443e38, and from there
into every HDF5 fill on a server. It is not configurable: there is no
server variable for it, and ``FLOAT(24)`` renders the same (``FLOAT(25)``
only differs because MySQL promotes it to DOUBLE).

This widens the columns. It cannot recover digits already lost -- a
coordinate stored through a 32-bit column is rounded on the way in, and no
later ALTER knows what it used to be. Re-import from the original source if
exact values matter; the errors were of order an arcsecond for target and
observatory coordinates.

SQLite is migrated too, even though nothing changes numerically there --
its FLOAT and DOUBLE are both 64-bit REAL. What changes is the *declared*
type, so that a database says the same thing on every backend rather than
carrying a type name that only happens to be harmless. The alternative,
skipping SQLite, leaves the schema disagreeing with the models on that
backend for ever.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0009_float_columns_to_double"
down_revision = "0008_widen_lc_status_id"
branch_labels = None
depends_on = None

# (table, column, nullable) as of this revision. Frozen: the models are
# free to move, this list describes the schema at this point in history.
COLUMNS = (
    ("camera_type", "pixel_size", False),
    ("hdf5_datasets", "replace_nonfinite", True),
    ("observatory", "latitude", False),
    ("observatory", "longitude", False),
    ("observatory", "altitude", False),
    ("target", "ra", True),
    ("target", "dec", True),
    ("telescope_type", "f_ratio", False),
    ("telescope_type", "focal_length", False),
)


def _needs_widening(inspector, table, column):
    """Whether *column* exists and is still a 32-bit float."""

    if table not in inspector.get_table_names():
        return False
    for described in inspector.get_columns(table):
        if described["name"] == column:
            return not isinstance(described["type"], sqlalchemy.Double)
    return False


def upgrade():
    """Change each column to DOUBLE where it is not already.

    SQLite cannot alter a column in place, so this goes through
    ``batch_alter_table``, which rebuilds the table. That works only
    because ``0003`` repaired the timestamp triggers first: SQLite
    re-parses every trigger when a table is renamed, and one that does not
    parse blocks the rebuild.
    """

    connection = alembic.op.get_bind()
    inspector = sqlalchemy.inspect(connection)
    for table, column, nullable in COLUMNS:
        if not _needs_widening(inspector, table, column):
            continue
        with alembic.op.batch_alter_table(table) as batch:
            batch.alter_column(
                column,
                existing_type=sqlalchemy.Float(),
                type_=sqlalchemy.Double(),
                existing_nullable=nullable,
            )


def downgrade():
    """Deliberately not narrowed again.

    Going back to FLOAT would round every value already stored, which is
    the loss this revision exists to stop.
    """
