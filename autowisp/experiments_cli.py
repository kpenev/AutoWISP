"""Create, list, reset, merge, restore and delete a project's experiments.

The command line of :class:`~autowisp.project_experiments.ProjectExperiments`
(``wisp-experiments``): each subcommand is one operation on the experiments of
the project at the given home, which is their original. The project is
migrated first if it is behind, as opening it anywhere does; an experiment
behind is refused by reset and merge, and is migrated by opening it.

Errors, refusals included, are recorded in the project and reported as any
command's (:func:`~autowisp.error_cli.cli_entry_point`), against the
operation's run once it has one, so that the browser interface, running a
command in the background, can show them.
"""

import os
from argparse import ArgumentParser

from autowisp.database.interface import (
    DB_URL_FNAME,
    get_db_engine,
    has_project_database,
    set_project_home,
)
from autowisp.database.migrate import describe_migration
from autowisp.error_cli import cli_entry_point
from autowisp.exceptions import Component, ExperimentError
from autowisp.project_experiments import ProjectExperiments


def _confirm(question, config):
    """Return whether the user confirms, or confirmed with ``--yes``."""

    if config.yes:
        return True
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _describe(described):
    """Return a line describing an experiment, as listed."""

    # By the task of an operation interrupted, what to do next.
    next_steps = {
        "create experiment": "delete it",
        "reset experiment": "delete it",
        "merge experiment": "restore the original (wisp-experiments restore)",
        "delete experiment": "delete it again",
    }
    line = (
        f"{described['name']}: {described['state']}, created "
        f"{described['created']:%Y-%m-%d %H:%M:%S} UTC"
    )
    operation = described["operation"]
    if operation is None:
        return line
    if operation["running"] is None:
        return (
            f"{line}; {operation['task']} interrupted: "
            f"{next_steps.get(operation['task'], 'delete it')}"
        )
    where = (
        "here"
        if operation["running"] == "here"
        else f"on {operation['running']}"
    )
    return f"{line}; {operation['task']} running {where}"


def _create(experiments, config):
    """Create the experiment."""

    state = experiments.create(config.name, config.ignored_runs)
    print(f"Created experiment {config.name}: {state}.")


def _list(experiments, config):
    """List the experiments, one per line."""

    described = experiments.describe_experiments()
    if described is None:
        print(
            f"{config.project_home} is an experiment: its original has the "
            "experiments."
        )
    elif not described:
        print("No experiments.")
    for line in map(_describe, described or []):
        print(line)


def _reset(experiments, config):
    """Reset the experiment's processing."""

    state = experiments.reset(config.name, config.ignored_runs)
    print(f"Reset experiment {config.name}: {state}.")


def _merge(experiments, config):
    """Merge the experiment, naming the backup made."""

    backup = experiments.merge(config.name, config.force, config.ignored_runs)
    print(
        f"Merged experiment {config.name}. The project as it was is in "
        f"{backup}: once the merge is known to be right, delete it "
        "(wisp-experiments delete-backup)."
    )


def _restore(experiments, config):
    """Restore the project from the backup of an interrupted merge."""

    state = experiments.restore_merge(config.name, config.ignored_runs)
    print(f"Restored the project; experiment {config.name} is {state}.")


def _delete(experiments, config):
    """Delete the experiment, once confirmed."""

    if _confirm(
        f"Delete experiment {config.name}, with everything under its "
        "directory?",
        config,
    ):
        experiments.delete(config.name, config.ignored_runs)
        print(f"Deleted experiment {config.name}.")


def _delete_backup(experiments, config):
    """Delete the merges' backups, once confirmed."""

    if _confirm("Delete every backup merges left?", config):
        for backup in experiments.delete_merge_backup():
            print(f"Deleted {backup}.")


def _unlock(experiments, _):
    """Mark every current experiment stale."""

    for name in experiments.unlock():
        print(f"Marked experiment {name} stale.")


def parse_command_line(argv=None):
    """Return the parsed command line arguments."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "project_home",
        help="The home of the project whose experiments to work on.",
    )
    parser.add_argument(
        "--assume-backed-up",
        action="store_true",
        help="Confirm that a server (MySQL/MariaDB) project database has "
        "been backed up, which migrating it requires if it is behind (see "
        "wisp-migrate).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    def add_subcommand(command, run, description, *, name=True, run_check=True):
        """Add a subcommand, run by the given function."""

        subcommand = subcommands.add_parser(
            command, help=description, description=description
        )
        subcommand.set_defaults(run=run)
        if name:
            subcommand.add_argument("name", help="The experiment's name.")
        if run_check:
            subcommand.add_argument(
                "--ignore-run",
                type=int,
                action="append",
                default=[],
                metavar="ID",
                dest="ignored_runs",
                help="A run known to have ended without recording it, e.g. "
                "on a machine that crashed, which is then not counted as "
                "going. Repeat for several.",
            )
        return subcommand

    add_subcommand(
        "create", _create, "Create an experiment: a copy of the project."
    )
    add_subcommand(
        "list",
        _list,
        "List the experiments, with their states and the operations going "
        "or interrupted on them.",
        name=False,
        run_check=False,
    )
    add_subcommand(
        "reset",
        _reset,
        "Reset an experiment's processing to the project's, keeping its "
        "configuration.",
    )
    add_subcommand(
        "merge",
        _merge,
        "Merge an experiment into the project, which becomes a copy of it. "
        "The project's files and database are backed up first.",
    ).add_argument(
        "--force",
        action="store_true",
        help="Merge even if the project changed after the experiment was "
        "copied, losing the change.",
    )
    add_subcommand(
        "restore",
        _restore,
        "Restore the project from the backup of an interrupted merge.",
    )
    for subcommand in [
        add_subcommand("delete", _delete, "Delete an experiment."),
        add_subcommand(
            "delete-backup",
            _delete_backup,
            "Delete the backups merges left, once the merge is known to be "
            "right.",
            name=False,
            run_check=False,
        ),
    ]:
        subcommand.add_argument(
            "--yes", action="store_true", help="Do not ask for confirmation."
        )
    add_subcommand(
        "unlock",
        _unlock,
        "Release the project for changes by marking every current "
        "experiment stale.",
        name=False,
        run_check=False,
    )
    return parser.parse_args(argv)


@cli_entry_point(component=Component.PIPELINE)
def main(argv=None):
    """Run one operation on the experiments of a project."""

    config = parse_command_line(argv)
    home = os.path.abspath(config.project_home)
    # Checked first: opening a project without a database would create one.
    if not has_project_database(home):
        raise ExperimentError(
            f"{home} holds no AutoWISP project: it has neither autowisp.db "
            f"nor {DB_URL_FNAME}."
        )
    migration = set_project_home(
        home, migrate=True, assume_backed_up=config.assume_backed_up
    )
    if migration is not None and migration["from"] != migration["to"]:
        print(describe_migration(migration))
    config.run(ProjectExperiments(get_db_engine(), home), config)


if __name__ == "__main__":
    main()
