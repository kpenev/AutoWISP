#!/usr/bin/env python3

"""Regenerate ``wisp_options.rst`` from the parameters a project defines.

Option descriptions live in the ``parameter`` table, which is filled in when
a project is created, so producing the list needs a project to read from.
This script makes a throwaway one in a temporary directory and discards it
afterwards. That is simpler than pointing at a project of your own, and
safer: the generated file then reflects the options the current code
defines, rather than whatever the code looked like when some particular
project happened to be created.

The options the pipeline sets for each batch (``engine_set_options``) are
not in that table, being no part of a configuration, but a step run by hand
takes them, so they follow in a section of their own, with the help the
steps' command-line parsers give them.

Run it after adding, removing or re-wording any pipeline option::

    python3 documentation/source/document_options.py
"""

from argparse import Namespace
from os import path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from autowisp import processing_steps

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import Parameter, Step

# pylint: enable=no-name-in-module
from autowisp.database.initialize_database import (
    engine_set_options,
    initialize_database,
)
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


def get_engine_set_help(step_names):
    """Return the help each step gives the options the pipeline sets.

    These are not in the ``parameter`` table, which holds what a project
    can configure, so they are read from the steps' command-line parsers,
    as project creation reads every option.

    Args:
        step_names:    The steps of the project, in processing order.

    Returns:
        dict:    ``{option: {help: [step, ...]}}``, in the order of
            ``engine_set_options``, for the options some step takes. One
            option can be described differently by different steps.
    """

    result = {option: {} for option in engine_set_options}
    for step_name in step_names:
        descriptions = getattr(processing_steps, step_name).parse_command_line(
            []
        )["argument_descriptions"]
        for option, by_help in result.items():
            if option not in descriptions:
                continue
            description = descriptions[option]
            if isinstance(description, dict):
                description = description["help"]
            by_help.setdefault(description or "", []).append(step_name)
    return {option: by_help for option, by_help in result.items() if by_help}


def format_engine_set_option(option, by_help):
    """Return the rst directive documenting an option the pipeline sets.

    Args:
        option(str):    The option's name.

        by_help(dict):    The steps taking it by the help each gives it, as
            :func:`get_engine_set_help` returns for the option.

    Returns:
        str:    An ``option`` directive, indented body included, giving each
            help with the steps it applies to where they differ.
    """

    paragraphs = (
        list(by_help)
        if len(by_help) == 1
        else [
            f"For {', '.join(step_names)}: {description}"
            for description, step_names in by_help.items()
        ]
    )
    body = "\n\n\t".join(
        paragraph.replace("\n", "\n\n\t") for paragraph in paragraphs
    )
    return f".. option:: {option} (--{option} on command line)\n\n\t{body}\n\n"


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
                engine_set_help = get_engine_set_help(
                    db_session.scalars(select(Step.name).order_by(Step.id))
                )
                with open(OUTPUT_FNAME, "w", encoding="utf-8") as options_rst:
                    options_rst.write(
                        "Configuration Options\n=====================\n\n"
                    )
                    for parameter in parameters:
                        options_rst.write(format_parameter(parameter))
                    options_rst.write(
                        "Options the Pipeline Sets\n"
                        "-------------------------\n\n"
                        "The pipeline gives these to a step for each batch: "
                        "the masters it selects for the images, and the "
                        "exclusion list it writes from the step's exclusion "
                        "rule. They are not part of the configuration and "
                        "cannot be set there. Give them only when running a "
                        "step by hand.\n\n"
                    )
                    for option, by_help in engine_set_help.items():
                        options_rst.write(
                            format_engine_set_option(option, by_help)
                        )
        finally:
            # Release the sqlite file before the temporary directory is
            # removed, which Windows requires.
            engine = get_db_engine()
            if engine is not None:
                engine.dispose()

    print(
        f"Documented {len(parameters)} configurable options and "
        f"{len(engine_set_help)} the pipeline sets in {OUTPUT_FNAME!r}"
    )


if __name__ == "__main__":
    main()
