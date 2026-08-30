"""Keep ``modified`` current for writes that bypass Django's ORM.

``auto_now`` only fires where Django is doing the writing, and not even
everywhere it is (see :class:`core.models.BuiModelBase`).  A per-table
trigger covers the rest, and is installed for every model deriving from
that base rather than per model, so a table added later cannot be missed.

Modelled on the project database's ``update_<table>_timestamp`` triggers
(``autowisp/database/data_model/__init__.py``), including the lesson its
``0003_repair_timestamp_triggers`` revision had to teach: key the trigger on
the primary key the model actually declares, never on an assumed ``id``.
"""

import logging

from django.core.exceptions import FieldDoesNotExist
from django.db import connections

_logger = logging.getLogger(__name__)

#: The column these triggers maintain.  Models get it from
#: :class:`core.models.BuiModelBase`, but the trigger is installed on the
#: strength of the column being present rather than of where it came from.
#: That is both the honest test -- the trigger maintains a column, not a
#: class -- and what keeps this module free of a model import, which at
#: ``AppConfig`` import time would raise ``AppRegistryNotReady``.
_column = "modified"

# SQLite has no BEFORE-UPDATE assignment, so the row is written a second
# time.  The WHEN guard stops that recursing, and additionally lets a
# statement that set ``modified`` deliberately -- a data migration
# backdating a row -- keep the value it chose.
#
# strftime rather than CURRENT_TIMESTAMP because the latter truncates to
# whole seconds, while Django writes microseconds; matching the format
# keeps rows ordered correctly within a second.
_sqlite_trigger = (
    "CREATE TRIGGER {name} AFTER UPDATE ON {table} FOR EACH ROW"
    " WHEN NEW.{column} = OLD.{column}"
    " BEGIN"
    " UPDATE {table} SET {column} = strftime('%Y-%m-%d %H:%M:%f', 'now')"
    " WHERE {key} = NEW.{key};"
    " END"
)

# MySQL can assign to NEW directly, so no second write and no recursion.
# The IF mirrors SQLite's WHEN so that an explicit value survives on both.
_mysql_trigger = (
    "CREATE TRIGGER {name} BEFORE UPDATE ON {table} FOR EACH ROW"
    " BEGIN"
    " IF NEW.{column} = OLD.{column} THEN"
    " SET NEW.{column} = CURRENT_TIMESTAMP(6);"
    " END IF;"
    " END"
)

_templates = {"sqlite": _sqlite_trigger, "mysql": _mysql_trigger}


def timestamped_models(app_config):
    """Yield the app's concrete models that carry a ``modified`` column."""

    for model in app_config.get_models():
        try:
            model._meta.get_field(_column)  # pylint: disable=W0212
        except FieldDoesNotExist:
            continue
        yield model


def _trigger_name(connection, model):
    """Return the quoted trigger name for a model's table."""

    table = model._meta.db_table  # pylint: disable=protected-access
    return connection.ops.quote_name(f"update_{table}_modified")


def drop_modified_triggers(sender, using, **_kwargs):
    """
    Remove the triggers before an app's migrations run.

    Necessary, not tidiness: SQLite validates existing triggers when a
    table is altered, so a trigger naming ``modified`` makes any migration
    touching that column fail with "no such column: NEW.modified" -- the
    same way an unparseable trigger blocked migrations of the project
    database until ``0003_repair_timestamp_triggers``.  ``post_migrate``
    puts them back, so the pair is what makes the schema alterable.

    Args:
        sender:    The ``AppConfig`` whose migrations are about to run.

        using(str):    Alias of the database they will run against.

    Returns:
        None
    """

    connection = connections[using]
    if connection.vendor not in _templates:
        return

    models = list(timestamped_models(sender))
    if not models:
        return

    with connection.cursor() as cursor:
        for model in models:
            cursor.execute(
                f"DROP TRIGGER IF EXISTS {_trigger_name(connection, model)}"
            )


def install_modified_triggers(sender, using, **_kwargs):
    """
    Create the ``modified`` trigger for each of an app's models.

    Connected to ``post_migrate``, so it runs after every ``migrate`` --
    which the browser interface performs on every launch -- and therefore
    repairs a trigger that a schema change dropped.  SQLite rebuilds a table
    to alter it, taking its triggers with it, so that is not hypothetical.

    Args:
        sender:    The ``AppConfig`` whose migrations just ran.

        using(str):    Alias of the database they ran against.

    Returns:
        None
    """

    connection = connections[using]
    template = _templates.get(connection.vendor)
    models = list(timestamped_models(sender))
    if not models:
        return

    if template is None:
        _logger.warning(
            "No modified-timestamp trigger for database vendor %r; "
            "modified will only be set by Django itself on %s.",
            connection.vendor,
            ", ".join(model.__name__ for model in models),
        )
        return

    quote = connection.ops.quote_name
    with connection.cursor() as cursor:
        for model in models:
            name = _trigger_name(connection, model)
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
            cursor.execute(
                template.format(
                    name=name,
                    table=quote(model._meta.db_table),  # pylint: disable=W0212
                    column=quote(_column),
                    # Derived, never assumed to be `id`: that assumption
                    # is exactly what broke the project database's
                    # triggers.
                    key=quote(model._meta.pk.column),  # pylint: disable=W0212
                )
            )
