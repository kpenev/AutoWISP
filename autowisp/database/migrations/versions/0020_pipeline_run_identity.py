"""Record which machine and which process is running a pipeline run.

Whether a run is still going was decided by its host name and process id
alone. A machine's name can change while a run is going -- macOS renames
itself with the network -- and the run is then missed: the BUI offered to
start processing while a run was going (SUP-491). And a process id can be
reused once the run has died, which makes a crashed run look alive.

So a run now also records a hashed identifier of the machine, which survives
renaming it, and when its process started, which a process reusing the id
does not share. Both are nullable: runs recorded before have neither, and are
judged by host name as before.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0020_pipeline_run_identity"
down_revision = "0019_restore_lc_fnum_fill"
branch_labels = None
depends_on = None

TABLE = "pipeline_run"

# (column, type) as of this revision. Frozen: the model is free to move.
COLUMNS = (
    ("machine_id", sqlalchemy.String(64)),
    ("process_started", sqlalchemy.Double()),
)


def _get_present_columns(connection):
    """Return the names of the table's columns, as the database has them."""

    return {
        described["name"]
        for described in sqlalchemy.inspect(connection).get_columns(TABLE)
    }


def upgrade():
    """Add whichever of the columns the table lacks."""

    present = _get_present_columns(alembic.op.get_bind())
    for name, column_type in COLUMNS:
        if name not in present:
            alembic.op.add_column(
                TABLE, sqlalchemy.Column(name, column_type, nullable=True)
            )


def downgrade():
    """Drop the columns again; runs are then judged by host name alone."""

    present = _get_present_columns(alembic.op.get_bind())
    # SQLite cannot drop a column in place, so the table is rebuilt there;
    # migrate_project reinstates the timestamp trigger the rebuild loses.
    with alembic.op.batch_alter_table(TABLE) as batch:
        for name, _ in COLUMNS:
            if name in present:
                batch.drop_column(name)
