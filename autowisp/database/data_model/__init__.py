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
            print("Checking", mod_attr)
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

        update_timestamp_mysql = """
            CREATE TRIGGER update_{table}_timestamp
            BEFORE UPDATE
            ON %(table)s
            FOR EACH ROW
                SET NEW.timestamp = CURRENT_TIMESTAMP
            """

        # Keyed on the table's actual primary key. It used to be hardcoded
        # to `id`, which most tables have but not all: the trigger on
        # image_master_selection (primary key image_id, channel,
        # master_type_id) could not parse. That went unnoticed because
        # SQLite only parses a trigger when it fires -- or when something
        # renames a table, which is how a batch migration rebuilds one.
        update_timestamp_sqlite = """
            CREATE TRIGGER update_{table}_timestamp
            AFTER UPDATE
            ON {table}
            FOR EACH ROW
            BEGIN
                UPDATE {table} SET timestamp = CURRENT_TIMESTAMP
                WHERE {match};
            END
            """

        for class_name in table_class_names:
            table_class = getattr(module, class_name)
            setattr(this_module, class_name, table_class)
            table = table_class.__table__
            event.listen(
                table,
                "after_create",
                DDL(update_timestamp_mysql.format(table=table)).execute_if(
                    dialect="mysql"
                ),
            )
            key_columns = [column.name for column in table.primary_key.columns]
            if key_columns:
                event.listen(
                    table,
                    "after_create",
                    DDL(
                        update_timestamp_sqlite.format(
                            table=table,
                            match=" AND ".join(
                                f"{name} = NEW.{name}" for name in key_columns
                            ),
                        )
                    ).execute_if(dialect="sqlite"),
                )

            __all__.append(class_name)


import_table_definitions()
