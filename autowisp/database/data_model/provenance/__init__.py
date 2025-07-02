"""Add all tables to __all__."""

import sys
from glob import glob
from os.path import dirname, join, basename
from importlib import import_module
from inspect import isclass

from autowisp.database.data_model.base import DataModelBase

# TODO separate impoprt table definitions as a new file and make two inits for
# provenance and datamodel and then separate __all__ from the import_module lists
# generated and combine them run an end step keep them separate (keep
# frames_provenance and frames separate, keep in that modules namespace only)

__all__ = []


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
        module = import_module(
            "autowisp.database.data_model.provenance." + module_name
        )

        # Pylint false positive
        # pylint: disable=cell-var-from-loop
        def is_table(mod_attr):
            """Check if module attribute is a table."""
            return (
                mod_attr[0] != "_"
                and mod_attr != "DataModelBase"
                and isclass(getattr(module, mod_attr))
                and issubclass(getattr(module, mod_attr), DataModelBase)
            )

        # pylint: enable=cell-var-from-loop
        table_class_name = list(
            filter(is_table, getattr(module, "__all__", []))
        )
        assert len(table_class_name) == 1
        table_class_name = table_class_name[0]
        setattr(
            this_module, table_class_name, getattr(module, table_class_name)
        )
        __all__.append(table_class_name)


import_table_definitions()
