"""Repair update-timestamp triggers that assume an ``id`` column.

AutoWISP generates an ``update_<table>_timestamp`` trigger per table from
one template, and on SQLite that template ended ``WHERE id = NEW.id``. Most
tables have an ``id``, but not all: ``image_master_selection`` is keyed on
(image_id, channel, master_type_id), so its trigger never parsed.

That went unnoticed because SQLite parses a trigger only when it fires --
so any ``UPDATE image_master_selection`` raised "no such column: id" -- or
when a table is renamed, which is how SQLite performs the column and
constraint changes in the revisions that follow this one. A single invalid
trigger anywhere in the schema blocks every one of those.

Hence this runs first. It repairs the trigger from the primary key the
database actually reports, rather than from the models, so it fixes
whatever a given project happens to have.

MySQL is untouched: its template sets ``NEW.timestamp`` directly and never
referred to ``id``.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0003_repair_timestamp_triggers"
down_revision = "0002_image_session_index"
branch_labels = None
depends_on = None


def _rebuild_trigger(connection, table, key_columns):
    """Drop and recreate one table's timestamp trigger."""

    name = f"update_{table}_timestamp"
    match = " AND ".join(f"{column} = NEW.{column}" for column in key_columns)
    # Table and column names come from the database's own catalogue, not
    # from user input, so interpolating them is safe here.
    connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}")
    connection.exec_driver_sql(
        f"CREATE TRIGGER {name} AFTER UPDATE ON {table} FOR EACH ROW "
        f"BEGIN UPDATE {table} SET timestamp = CURRENT_TIMESTAMP "
        f"WHERE {match}; END"
    )


def upgrade():
    """Recreate any trigger whose table has no ``id`` column."""

    connection = alembic.op.get_bind()
    if connection.dialect.name != "sqlite":
        return

    inspector = sqlalchemy.inspect(connection)
    for table in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "id" in columns or "timestamp" not in columns:
            continue
        key_columns = inspector.get_pk_constraint(table)["constrained_columns"]
        if key_columns:
            _rebuild_trigger(connection, table, key_columns)


def downgrade():
    """Deliberately not reinstated.

    Putting back a trigger that cannot parse would serve nobody, and would
    block any later attempt to alter these tables on SQLite.
    """
