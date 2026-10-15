"""Create, reset, merge and delete a project's experiments.

The operations copy whole projects: their files, and their databases with
:mod:`autowisp.database.project_copy`. What they record in the original's
database is managed by
:class:`~autowisp.database.experiments.ExperimentRecords`, which the database
layer uses on its own, being unable to import the steps and the copy.
"""

from sqlalchemy import select

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Configuration, Parameter

# pylint: enable=no-name-in-module
from autowisp.database.experiments import ExperimentRecords
from autowisp.exceptions import ConfigurationError
from autowisp.processing_steps import get_path_arguments
from autowisp.processing_steps.manual_util import split_path_values
from autowisp.project_paths import PROJHOME_MARKER


class ProjectExperiments(ExperimentRecords):
    """
    The experiments of one project, and every operation on them.

    Adds to the records what reaches beyond the database: the configured
    paths, and the projects' files.
    """

    @staticmethod
    def _refuse_external_paths(db_session):
        """
        Raise if a configured path lies outside the project.

        An experiment shares with its original whatever lies outside its home,
        so a step writing there would overwrite the other's products. Every
        configured value of every path option, under any condition and in any
        version, must start with ``{PROJHOME}``, except those of the options
        declared ``path="external"``: files of other tools, only read, and the
        temporary storage, under which each process makes a directory of its
        own. Inputs are not told from outputs: one step's outputs are
        another's inputs.

        Stored paths are compared as strings, so only the literal marker
        counts: an equivalent spelling, e.g. ``{PROJHOME:s}``, is refused.

        Args:
            db_session:    A session on the project's database.

        Raises:
            ConfigurationError:    Naming each parameter and value refused.
        """

        path_kinds = {
            name: path_kind
            for name, path_kind in get_path_arguments().items()
            if path_kind != "external"
        }
        refused = sorted(
            {
                (name, value)
                for name, value in db_session.execute(
                    select(Parameter.name, Configuration.value)
                    .join(Configuration.parameter)
                    .where(Parameter.name.in_(path_kinds))
                )
                if value
                and any(
                    not path.startswith(PROJHOME_MARKER)
                    for _, path in split_path_values(value, path_kinds[name])
                )
            }
        )
        if refused:
            raise ConfigurationError(
                "Paths are configured outside the project, where an "
                "experiment and its original would share them: "
                + ", ".join(f"{name} = {value}" for name, value in refused)
                + f". Configure them under {PROJHOME_MARKER}.",
                details={"parameters": sorted({name for name, _ in refused})},
            )

    def refuse_run(self):
        """
        Raise unless a processing run may start in the project.

        Refused in a locked project (see :meth:`refuse_if_locked`), and in an
        experiment whose configuration has a path outside it (see
        :meth:`_refuse_external_paths`). An ordinary project may write outside
        its home.

        Raises:
            ProjectLockedError:    If the project is locked.

            ConfigurationError:    If the project is an experiment with a path
                configured outside it.
        """

        self.refuse_if_locked()
        with self._begin_session() as db_session:
            if self._read_experiments(db_session) is None:
                self._refuse_external_paths(db_session)
