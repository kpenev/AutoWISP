"""Add the table recording a project's experiments.

An experiment is a copy of a project, processed on its own and possibly
merged back. The original records its experiments in this table; an
experiment's own database has none, and that absence is what makes it an
experiment.

Every project at the previous revision is an original, since experiments are
created at this revision or later. The exception is an experiment downgraded
below this revision, which the older code takes for an ordinary project
anyway, and which becomes one here when it is upgraded again. A later
revision changing the table must not create it where it is missing: that
would turn an experiment into an ordinary project.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0021_experiment"
down_revision = "0020_pipeline_run_identity"
branch_labels = None
depends_on = None

TABLE_NAME = "experiment"


def _has_table(connection):
    """Whether the experiment table is already there."""

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
        sqlalchemy.Column("home", sqlalchemy.String(1000), nullable=False),
        sqlalchemy.Column("created", sqlalchemy.TIMESTAMP, nullable=False),
        sqlalchemy.Column("fingerprint", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("state", sqlalchemy.String(20), nullable=False),
    )


def downgrade():
    """Drop the table, and the record of every experiment, if it is there.

    The experiments' directories are left in place: they are ordinary
    projects, which the code being downgraded to opens like any other.
    """

    if not _has_table(alembic.op.get_bind()):
        return

    alembic.op.drop_table(TABLE_NAME)
