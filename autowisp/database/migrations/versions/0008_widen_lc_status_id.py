"""Widen ``light_curve_status.id`` to hold a Gaia source id.

The column is not a row counter: ``LightCurveProcessingManager`` stores the
star's Gaia source id in it. Those run to ~8e17, which overflows the 32-bit
INT that ``Integer`` becomes on MySQL and MariaDB -- resuming interrupted
light-curve processing failed there with "Out of range value for column
'id'", or, on a server not in strict mode, silently truncated the id.

SQLite hid it completely: its INTEGER is 64-bit and it does not enforce
declared types anyway.
"""

import sqlalchemy
from sqlalchemy.dialects import mysql, postgresql, sqlite
import alembic

# revision identifiers, used by Alembic.
revision = "0008_widen_lc_status_id"
down_revision = "0007_drop_condition_fks"
branch_labels = None
depends_on = None

TABLE = "light_curve_status"
COLUMN = "id"

# A frozen copy of data_model.light_curve_status.GaiaIDType. Copied rather
# than imported: a revision describes one schema for ever, and the model is
# free to move. The SQLite variant is why this is not plain BigInteger --
# SQLite renders that as BIGINT, and BIGINT PRIMARY KEY is not a rowid
# alias, so the column has to stay INTEGER there.
GAIA_ID_TYPE = (
    sqlalchemy.BigInteger()
    .with_variant(postgresql.BIGINT(), "postgresql")
    .with_variant(mysql.BIGINT(), "mysql")
    .with_variant(sqlite.INTEGER(), "sqlite")
)


def _is_already_wide(connection):
    """Whether the column already holds 64-bit values, or is absent."""

    for described in sqlalchemy.inspect(connection).get_columns(TABLE):
        if described["name"] == COLUMN:
            return isinstance(described["type"], sqlalchemy.BigInteger)
    return True


def upgrade():
    """Widen the column where the rendered type actually differs.

    On SQLite the type resolves to INTEGER, which is what the column
    already is, so rebuilding the table there would be work for no change.
    """

    connection = alembic.op.get_bind()
    if connection.dialect.name == "sqlite" or _is_already_wide(connection):
        return

    alembic.op.alter_column(
        TABLE,
        COLUMN,
        existing_type=sqlalchemy.Integer(),
        type_=GAIA_ID_TYPE,
        existing_nullable=False,
    )


def downgrade():
    """Deliberately not narrowed again.

    Going back to INT would truncate every id already stored, which is the
    corruption this revision exists to prevent.
    """
