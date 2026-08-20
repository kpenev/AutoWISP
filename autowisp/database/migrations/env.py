"""Alembic environment for the AutoWISP project database.

Unlike a stock Alembic setup this never creates its own engine. The caller
(:mod:`autowisp.database.migrate`) already owns one -- projects live in
different databases, chosen at runtime -- and passes an open connection in
through ``config.attributes``, so the migration runs inside the caller's
transaction and its lock.

Offline (``--sql``) mode is deliberately unsupported: project databases are
always migrated against a live connection, and emitting SQL for a schema we
cannot inspect would be misleading.
"""

from alembic import context

from autowisp.database.data_model.base import DataModelBase


def run_migrations_online():
    """Run the migrations against the connection supplied by the caller."""

    connection = context.config.attributes.get("connection")
    if connection is None:
        raise RuntimeError(
            "No connection supplied to the Alembic environment. Project "
            "databases must be migrated through "
            "autowisp.database.migrate.migrate_project(), which passes the "
            "engine's connection in via Config.attributes."
        )

    context.configure(
        connection=connection,
        target_metadata=DataModelBase.metadata,
        # SQLite cannot ALTER most things in place; batch mode renders such
        # operations as create/copy/drop instead. Harmless on backends that
        # do not need it.
        render_as_batch=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError(
        "Offline (--sql) migration is not supported for project databases."
    )

run_migrations_online()
