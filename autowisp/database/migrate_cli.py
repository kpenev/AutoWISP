"""Command line entry point for migrating a project database.

Opening a project with an out-of-date schema fails rather than silently
repairing itself, because every pipeline worker opens the project and
concurrent DDL from dozens of processes is not survivable. This is the
single-process command that does the repair. The browser interface migrates
on project selection and a pipeline run migrates in its main process, so this
is mostly for non-BUI use and for scripting.
"""

from configargparse import ArgumentParser, DefaultsFormatter

from autowisp.database.interface import set_project_home, get_db_engine
from autowisp.database.migrate import (
    get_head_revision,
    get_project_revision,
)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True,
    )
    parser.add_argument(
        "project_home",
        help="The project directory whose database should be migrated.",
    )
    parser.add_argument(
        "--assume-backed-up",
        action="store_true",
        help="Confirm that a centralised (MySQL/MariaDB) database has been "
        "backed up. Required for those, since they cannot be copied aside "
        "automatically and MySQL commits DDL implicitly, so a failed "
        "migration cannot be rolled back. SQLite databases are copied "
        "automatically and do not need this.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether the database is up to date and exit without "
        "changing anything.",
    )
    return parser.parse_args(*args)


def main(config=None):
    """Migrate the project database, reporting what changed."""

    if config is None:
        config = parse_command_line()

    if config.check:
        # Opening without migrate= raises if the schema is out of date, which
        # is the report wanted here -- so say so instead of propagating.
        try:
            set_project_home(config.project_home)
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(str(error))
            return 1
        print(f"Up to date at {get_project_revision(get_db_engine())}.")
        return 0

    result = set_project_home(
        config.project_home,
        migrate=True,
        assume_backed_up=config.assume_backed_up,
    )

    if result is None:
        print(f"Created a new project database at {get_head_revision()}.")
        return 0

    if result["from"] == result["to"]:
        print(f"Already up to date at {result['to']}.")
        return 0

    if result["backup"]:
        print(f"Backed up to {result['backup']}")
    print(f"Migrated from {result['from']} to {result['to']}.")
    return 0
