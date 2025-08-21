#!/usr/bin/env python3

"""Generate rst file with directives for each option."""

from os import path

from autowisp.database.data_model import (  # pylint: disable=no-name-in-module
    Parameter,
)
from autowisp.database.interface import start_db_session

if __name__ == "__main__":
    with start_db_session() as db_session:
        parameters = db_session.query(Parameter).all()

    with open(
        path.join(path.dirname(__file__), "wisp_options.rst"),
        "w",
        encoding="utf-8",
    ) as options_rst:
        options_rst.write(
            "Configuration Options\n"
            "=====================\n\n"
        )
        for p in parameters:
            options_rst.write(
                f".. option:: {p.name} (--{p.name} on command line)"
                f"\n\n\t{p.description.replace('\n', '\n\n\t')}\n\n"
            )
