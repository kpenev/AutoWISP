"""Add the diagnostic expression library to the project database.

The library used to live in the browser interface's own database, shared
by every project. Exclusion rules make expressions part of how a project is
processed, so each project now keeps its own; a shared one would let an
edit made for one project change what another one's rules exclude.

Nothing is carried over from the browser interface: its library was never
in use, and projects exchange expressions by export and import.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0011_diagnostic_expression"
down_revision = "0010_unique_image_type_name"
branch_labels = None
depends_on = None

TABLE_NAME = "diagnostic_expression"


def _has_table(connection):
    """Whether the library table is already there.

    It can be without this revision having run: a database older than
    Alembic is first brought up with ``create_all``, which creates every
    table the models have today, before the revisions are applied.
    """

    return sqlalchemy.inspect(connection).has_table(TABLE_NAME)


def upgrade():
    """Create the table, unless it exists already.

    Its timestamp trigger is not created here: ``migrate_project``
    installs any missing one once the revisions are done.
    """

    if _has_table(alembic.op.get_bind()):
        return

    alembic.op.create_table(
        TABLE_NAME,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column(
            "timestamp",
            sqlalchemy.TIMESTAMP,
            nullable=False,
            server_default=sqlalchemy.text("CURRENT_TIMESTAMP"),
        ),
        sqlalchemy.Column(
            "name", sqlalchemy.String(100), nullable=False, unique=True
        ),
        sqlalchemy.Column("expression", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("description", sqlalchemy.Text, nullable=False),
    )


def downgrade():
    """Drop the table, and every expression in it, if it is there."""

    if not _has_table(alembic.op.get_bind()):
        return

    alembic.op.drop_table(TABLE_NAME)
