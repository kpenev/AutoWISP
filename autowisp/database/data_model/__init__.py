"""Add all tables to __all__."""

import sys
from glob import glob
from os.path import dirname, join, basename
from importlib import import_module
from inspect import isclass

from sqlalchemy import event, DDL

from autowisp.database.data_model.base import DataModelBase, DataModelSubBase
from autowisp.database.data_model.steps_and_parameters import (
    step_param_association,
)

__all__ = []


def timestamp_trigger_ddl(table, key_columns, dialect):
    """
    Return the DDL creating one table's update-timestamp trigger.

    The single definition of what these triggers are.  They are installed
    on ``CREATE TABLE`` by the event listeners below, but a table rebuilt
    by a migration is created without going through those, so
    :mod:`autowisp.database.migrate` reinstates them afterwards from here
    rather than from a copy of its own.

    Args:
        table(str):    Name of the table the trigger belongs to.

        key_columns:    The table's primary key columns, used by the SQLite
            form to address the updated row.  Derived rather than assumed
            to be ``id``: assuming it is what
            ``0003_repair_timestamp_triggers`` had to repair.

        dialect(str):    Name of the dialect to render for.

    Returns:
        str or None:    The ``CREATE TRIGGER`` statement, or ``None`` if
            this dialect needs no trigger or the table cannot have one.
    """

    name = f"update_{table}_timestamp"
    if dialect == "mysql":
        # Assigning to NEW in a BEFORE trigger writes the row once, so
        # there is nothing to key on and no recursion to avoid.
        return (
            f"CREATE TRIGGER {name} BEFORE UPDATE ON {table} "
            "FOR EACH ROW SET NEW.timestamp = CURRENT_TIMESTAMP"
        )
    if dialect == "sqlite":
        if not key_columns:
            return None
        match = " AND ".join(
            f"{column} = NEW.{column}" for column in key_columns
        )
        # SQLite has no BEFORE-UPDATE assignment, so the row is written a
        # second time and has to be addressed by its key.
        return (
            f"CREATE TRIGGER {name} AFTER UPDATE ON {table} FOR EACH ROW "
            f"BEGIN UPDATE {table} SET timestamp = CURRENT_TIMESTAMP "
            f"WHERE {match}; END"
        )
    return None


# TODO: merge with data_model/provenance/__init__.py
def import_table_definitions():
    """Import all table definitions directly to data_model."""

    this_module = sys.modules[__name__]
    table_modules = filter(
        lambda module_name: module_name not in ["__init__", "base"],
        (
            basename(module_path)[:-3]
            for module_path in glob(join(dirname(__file__), "*.py"))
        ),
    )
    for module_name in table_modules:
        module = import_module("autowisp.database.data_model." + module_name)

        # Pylint false positive
        # pylint: disable=cell-var-from-loop
        def is_table(mod_attr):
            return (
                mod_attr[0] != "_"
                and mod_attr != "DataModelBase"
                and mod_attr != "DataModelSubBase"
                and isclass(getattr(module, mod_attr))
                and issubclass(getattr(module, mod_attr), DataModelSubBase)
            )

        # pylint: enable=cell-var-from-loop
        table_class_names = list(
            filter(is_table, getattr(module, "__all__", []))
        )

        for class_name in table_class_names:
            setattr(this_module, class_name, getattr(module, class_name))
            __all__.append(class_name)


def attach_timestamp_triggers():
    """
    Arrange for every table carrying ``timestamp`` to keep it up to date.

    Driven off the metadata shared by every subclass of
    :class:`DataModelSubBase` rather than off the classes
    :func:`import_table_definitions` happens to bind.  Discovery there is by
    filesystem glob, which does not descend into ``data_model/provenance``,
    so attaching per discovered class silently left the twelve provenance
    tables maintaining a ``timestamp`` column nothing ever wrote.  The
    metadata has them regardless of which loop imported them, and will have
    any future subpackage too.
    """

    for table in DataModelSubBase.metadata.tables.values():
        if "timestamp" not in table.columns:
            continue
        key_columns = [column.name for column in table.primary_key.columns]
        for dialect in ("mysql", "sqlite"):
            statement = timestamp_trigger_ddl(str(table), key_columns, dialect)
            if statement is not None:
                event.listen(
                    table,
                    "after_create",
                    DDL(statement).execute_if(dialect=dialect),
                )


import_table_definitions()
attach_timestamp_triggers()
