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

Operations on experiments are pipeline runs of the original, so runs now
record their task: 'processing' for every run before this revision, and for
every run of the pipeline's steps since. Each experiment refers to the run
that last copied the original into it.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0021_experiment"
down_revision = "0020_pipeline_run_identity"
branch_labels = None
depends_on = None

TABLE_NAME = "experiment"
RUN_TABLE_NAME = "pipeline_run"
TASK_COLUMN = "task"
# Frozen: the model's default is free to move.
PROCESSING_TASK = "processing"


def _has_table(connection):
    """Whether the experiment table is already there."""

    return sqlalchemy.inspect(connection).has_table(TABLE_NAME)


def _has_task_column(connection):
    """Whether the runs already record their task."""

    return TASK_COLUMN in {
        described["name"]
        for described in sqlalchemy.inspect(connection).get_columns(
            RUN_TABLE_NAME
        )
    }


def upgrade():
    """Give runs their task, and create the table, unless already done.

    The table's timestamp trigger is not created here: ``migrate_project``
    installs any missing one once the revisions are done.
    """

    connection = alembic.op.get_bind()
    if not _has_task_column(connection):
        alembic.op.add_column(
            RUN_TABLE_NAME,
            sqlalchemy.Column(
                TASK_COLUMN,
                sqlalchemy.String(50),
                nullable=False,
                server_default=PROCESSING_TASK,
            ),
        )

    if _has_table(connection):
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
        sqlalchemy.Column(
            "copy_run_id",
            sqlalchemy.Integer,
            sqlalchemy.ForeignKey(f"{RUN_TABLE_NAME}.id", onupdate="CASCADE"),
            nullable=False,
        ),
    )


def downgrade():
    """Drop the table, and the record of every experiment, and the tasks.

    The experiments' directories are left in place: they are ordinary
    projects, which the code being downgraded to opens like any other. The
    runs of operations on experiments stay, which that code takes for
    processing runs; being finished, they change nothing it decides.
    """

    connection = alembic.op.get_bind()
    # First, since it refers to the runs.
    if _has_table(connection):
        alembic.op.drop_table(TABLE_NAME)

    if _has_task_column(connection):
        # SQLite cannot drop a column in place, so the table is rebuilt
        # there; migrate_project reinstates the timestamp trigger the
        # rebuild loses.
        with alembic.op.batch_alter_table(RUN_TABLE_NAME) as batch:
            batch.drop_column(TASK_COLUMN)
