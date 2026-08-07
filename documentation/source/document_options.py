#!/usr/bin/env python3

"""Regenerate ``wisp_options.rst`` from the parameters a project defines.

Option descriptions live in the ``parameter`` table, which is filled in when
a project is created, so producing the list needs a project to read from.
This script makes a throwaway one in a temporary directory and discards it
afterwards. That is simpler than pointing at a project of your own, and
safer: the generated file then reflects the options the current code
defines, rather than whatever the code looked like when some particular
project happened to be created.

Run it after adding, removing or re-wording any pipeline option::

    python3 documentation/source/document_options.py
"""

from argparse import Namespace
from os import path
from tempfile import TemporaryDirectory

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import Parameter

# pylint: enable=no-name-in-module
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import (
    get_db_engine,
    set_project_home,
    start_db_session,
)

OUTPUT_FNAME = path.join(
    path.dirname(path.abspath(__file__)), "wisp_options.rst"
)


def format_parameter(parameter):
    """Return the rst directive documenting a single parameter.

    Args:
        parameter(Parameter):    The database entry to document.

    Returns:
        str:    An ``option`` directive, indented body included.
    """

    # Blank lines separate paragraphs in rst, and the body has to stay
    # indented, so every newline in the description becomes both.
    body = (parameter.description or "").replace("\n", "\n\n\t")
    return (
        f".. option:: {parameter.name} (--{parameter.name} on command line)"
        f"\n\n\t{body}\n\n"
    )


def main():
    """Write the options page for the parameters of a freshly made project."""

    with TemporaryDirectory() as project_home:
        try:
            set_project_home(project_home)
            initialize_database(
                Namespace(
                    drop_hdf5_structure_tables=False, drop_all_tables=True
                )
            )
            with start_db_session() as db_session:
                parameters = (
                    db_session.query(Parameter).order_by(Parameter.id).all()
                )
                with open(OUTPUT_FNAME, "w", encoding="utf-8") as options_rst:
                    options_rst.write(
                        "Configuration Options\n=====================\n\n"
                    )
                    for parameter in parameters:
                        options_rst.write(format_parameter(parameter))
        finally:
            # Release the sqlite file before the temporary directory is
            # removed, which Windows requires.
            engine = get_db_engine()
            if engine is not None:
                engine.dispose()

    print(f"Documented {len(parameters)} options in {OUTPUT_FNAME!r}")


if __name__ == "__main__":
    main()
