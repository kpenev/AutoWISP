"""Tests of merging an experiment into its original."""

import os
import shutil
import unittest
from contextlib import nullcontext
from unittest import mock

from sqlalchemy import create_engine, func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.pipeline_runs import record_run
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import (
    ConfigurationError,
    ExperimentError,
    ResourceError,
)
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Error,
    Experiment,
    ImageType,
    PipelineRun,
)

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestMerge(ProjectCopyCase):
    """Merging an experiment: what refuses it, what it replaces and keeps."""

    # The raw image inside the project, which a merge leaves in place.
    _raw_image = os.path.join("RAW", "frame1.fits")

    @classmethod
    def setUpClass(cls):
        """Give the project a raw image inside it, and a file it wrote."""

        super().setUpClass()
        cls._add_processed_files()

    @staticmethod
    def _process_experiment(experiment_engine, experiment_home):
        """
        Process the experiment: rewrite a file, add one, record two runs.

        The runs take the ids the original gives to the next runs it records:
        creating ``exp2``, and the merge.
        """

        for fname in ["frame1.h5", "frame2.h5"]:
            with open(
                os.path.join(experiment_home, "DR", fname),
                "w",
                encoding="ascii",
            ) as file:
                file.write("Processed in the experiment.")
        with start_engine_session(experiment_engine) as db_session:
            for _ in range(2):
                db_session.add(
                    PipelineRun(
                        host="test-host",
                        process_id=1,
                        started=func.now(),
                        finished=func.now(),
                    )
                )

    @staticmethod
    def _add_other_experiment(engine, project_home):
        """Create experiment ``exp2``, with an error against its create."""

        ProjectExperiments(engine, project_home).create("exp2")
        with start_engine_session(engine) as db_session:
            db_session.add(
                Error(
                    pipeline_run_id=db_session.scalar(
                        select(Experiment.copy_run_id).filter_by(name="exp2")
                    ),
                    component="cli",
                    exception_class="ExperimentError",
                    user_message="Recorded against creating exp2.",
                )
            )

    @staticmethod
    def _set_state(engine, state):
        """Set the state of experiment ``exp1``."""

        with start_engine_session(engine) as db_session:
            db_session.execute(
                update(Experiment).filter_by(name="exp1").values(state=state)
            )

    @staticmethod
    def _add_run(engine, **columns):
        """Record a finished run with the given columns."""

        with start_engine_session(engine) as db_session:
            db_session.add(
                PipelineRun(
                    host="test-host",
                    started=func.now(),
                    finished=func.now(),
                    **columns,
                )
            )

    @staticmethod
    def _add_live_operation(engine):
        """Record a run of this process, creating another experiment."""

        with start_engine_session(engine) as db_session:
            record_run(db_session, "create experiment")

    @staticmethod
    def _add_image_type(engine):
        """Change the original, as a step run despite the lock would."""

        with start_engine_session(engine) as db_session:
            db_session.add(
                ImageType(name="added", description="Added after the copy.")
            )

    @staticmethod
    def _list_backups(project_home):
        """Return the paths of the merge backups in the project, sorted."""

        directory = os.path.join(project_home, "experiments")
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.startswith(".pre-merge-")
        ]

    @staticmethod
    def _read_runs(engine):
        """Return the processing runs of a project, each as a tuple."""

        with Session(engine) as db_session:
            return set(
                db_session.execute(
                    select(
                        PipelineRun.id,
                        PipelineRun.host,
                        PipelineRun.process_id,
                        PipelineRun.started,
                    ).filter_by(task=PipelineRun.processing_task)
                )
            )

    @staticmethod
    def _find_owner(owner_contents, contents):
        """Return whose contents those are, the original's first; or None."""

        return next(
            (
                owner
                for owner in ["original", "experiment"]
                if owner_contents[owner] == contents
            ),
            None,
        )

    def _read_backup(self, backup, original_tree):
        """
        Return whose files a backup holds, and whether it holds the error.

        Args:
            backup(str):    The backup's directory.

            original_tree(dict):    The original's files before the merge, as
                :meth:`_read_tree` returns them.

        Returns:
            tuple:    ``"original"`` if the backup holds the original's files,
            but the raw image, which stays in place; None otherwise. And
            whether its database holds the error recorded against creating
            ``exp2``.
        """

        backup_engine = create_engine(
            f"sqlite:///{os.path.join(backup, 'autowisp.db')}",
            poolclass=NullPool,
        )
        try:
            with Session(backup_engine) as db_session:
                has_error = db_session.scalar(select(Error.id)) is not None
        finally:
            backup_engine.dispose()
        return (
            (
                "original"
                if self._read_tree(backup)
                == {
                    fname: contents
                    for fname, contents in original_tree.items()
                    if fname != self._raw_image
                }
                else None
            ),
            has_error,
        )

    def _read_owners(self, homes, engines):
        """
        Return the files and processing runs of the original and experiment.

        Args:
            homes(dict):    The homes of the original and the experiment, by
                owner.

            engines(dict):    Their engines, by owner.

        Returns:
            dict:    By owner, the files, as :meth:`_read_tree` returns them.

            dict:    By owner, the processing runs, as :meth:`_read_runs`
            returns them; None for the experiment if its database is gone.
        """

        return (
            {owner: self._read_tree(home) for owner, home in homes.items()},
            {
                owner: (
                    self._read_runs(engines[owner])
                    if os.path.exists(os.path.join(home, "autowisp.db"))
                    else None
                )
                for owner, home in homes.items()
            },
        )

    def _read_outcome(self, engine, homes, owner_trees, owner_runs):
        """
        Return what a merge left, but its result and whether it merges again.

        Args:
            engine:    The original's engine.

            homes(dict):    The homes of the original and the experiment, by
                owner.

            owner_trees(dict):    By owner, the files before the merge.

            owner_runs(dict):    By owner, the processing runs before the
                merge; None for the experiment if its database is gone.

        Returns:
            dict:    What the merge left, by name (see :meth:`test_merge`).
        """

        backups = self._list_backups(homes["original"])
        with Session(engine) as db_session:
            outcome = {
                "states": dict(
                    db_session.execute(
                        select(Experiment.name, Experiment.state)
                    ).all()
                ),
                "exp2's run": db_session.scalar(
                    select(PipelineRun.task)
                    .join(Experiment, Experiment.copy_run_id == PipelineRun.id)
                    .filter(Experiment.name == "exp2")
                ),
                "error kept": db_session.scalar(select(Error.id)) is not None,
                "unfinished": sorted(
                    db_session.scalars(
                        select(PipelineRun.task).where(
                            PipelineRun.task != PipelineRun.processing_task,
                            PipelineRun.finished.is_(None),
                        )
                    )
                ),
            }
        outcome.update(
            {
                "files": self._find_owner(
                    owner_trees, self._read_tree(homes["original"])
                ),
                "rows": self._find_owner(owner_runs, self._read_runs(engine)),
                "backups": len(backups),
                "backup": (
                    self._read_backup(backups[0], owner_trees["original"])
                    if len(backups) == 1
                    else None
                ),
                "raw image linked": os.path.samefile(
                    *(
                        os.path.join(home, self._raw_image)
                        for home in homes.values()
                    )
                ),
            }
        )
        return outcome

    def _try_merge(self, prepare, name, force, no_space):
        """
        Merge an experiment in a copy of the project, as a scenario says.

        The experiment is processed, and ``exp2`` created after it, before
        the scenario prepares the copy.

        Args:
            prepare:    Given the original's engine and home, and the
                experiment's, prepares them and returns the ids of the runs
                to ignore; or None.

            name(str):    The name of the experiment to merge.

            force(bool):    Whether to force the merge.

            no_space(bool):    Whether the experiment's files find no free
                space to be copied into.

        Returns:
            dict:    What the merge left, by name (see :meth:`test_merge`).
        """

        # The experiment's engine and home.
        engine, project_home, *experiment = self._create_experiment()
        self._process_experiment(*experiment)
        self._add_other_experiment(engine, project_home)
        ignored_runs = (
            prepare(engine, project_home, *experiment) if prepare else None
        ) or set()
        homes = {"original": project_home, "experiment": experiment[1]}
        before = self._read_owners(
            homes, {"original": engine, "experiment": experiment[0]}
        )
        experiments = ProjectExperiments(engine, project_home)
        with (
            mock.patch.object(
                shutil, "disk_usage", return_value=mock.Mock(free=0)
            )
            if no_space
            else nullcontext()
        ):
            try:
                experiments.merge(name, force, ignored_runs)
                result = "merged"
            except (
                ExperimentError,
                ConfigurationError,
                ResourceError,
            ) as error:
                result = type(error).__name__
        outcome = self._read_outcome(engine, homes, *before)
        outcome["result"] = result
        outcome["merges again"] = None
        if result == "merged":
            # Without force: refused unless the fingerprint recorded is the
            # new original's.
            try:
                experiments.merge(name, ignored_runs=ignored_runs)
                outcome["merges again"] = True
            except ExperimentError:
                outcome["merges again"] = False
        return outcome

    def test_merge(self):
        """Each merge on a copy of its own: refused, restored or completed.

        What each leaves: the result, ``"merged"`` or the name of the class
        of the exception raised; the experiments' states; the task of the run
        ``exp2`` refers to; whose files and processing runs the original
        holds; whether the error against creating ``exp2`` is still there;
        the number of backups, and whose files the backup holds and whether
        its database has the error; whether the experiment merges again
        without force; whether the raw image is still the experiment's too;
        and the operations' runs left unfinished.
        """

        merged = {
            "result": "merged",
            # Other current experiments become stale.
            "states": {"exp1": "current", "exp2": "stale"},
            # Renumbered, with the experiment referring to it.
            "exp2's run": "create experiment",
            "files": "experiment",
            "rows": "experiment",
            "error kept": False,
            "backups": 1,
            "backup": ("original", True),
            "merges again": True,
            "raw image linked": True,
            "unfinished": [],
        }
        untouched = {
            "result": "ExperimentError",
            "states": {"exp1": "current", "exp2": "current"},
            "exp2's run": "create experiment",
            "files": "original",
            "rows": "original",
            "error kept": True,
            "backups": 0,
            "backup": None,
            "merges again": None,
            "raw image linked": True,
            "unfinished": [],
        }

        def unlock(engine, project_home, *_):
            """Mark every current experiment stale."""

            ExperimentRecords(engine, project_home).unlock()

        # Each scenario: what prepares the copy, given the original's engine
        # and home and the experiment's engine and home, returning the ids of
        # the runs to ignore; the name given; whether forced; whether there
        # is no free space; and what the merge leaves.
        test_scenarios = {
            "merged": (None, "exp1", False, False, merged),
            "earlier backup": (
                lambda _, project_home, *__: os.mkdir(
                    os.path.join(
                        project_home, "experiments", ".pre-merge-earlier"
                    )
                ),
                "exp1",
                False,
                False,
                merged,
            ),
            "no such experiment": (None, "exp2x", False, False, untouched),
            "incomplete": (
                lambda engine, *_: self._set_state(engine, "creating"),
                "exp1",
                True,
                False,
                {
                    **untouched,
                    "states": {"exp1": "creating", "exp2": "current"},
                },
            ),
            "merge interrupted": (
                lambda engine, *_: self._set_state(engine, "merging"),
                "exp1",
                True,
                False,
                {**untouched, "states": {"exp1": "merging", "exp2": "current"}},
            ),
            "stale": (
                unlock,
                "exp1",
                False,
                False,
                {**untouched, "states": {"exp1": "stale", "exp2": "stale"}},
            ),
            "stale, forced": (unlock, "exp1", True, False, merged),
            "original changed": (
                lambda engine, *_: self._add_image_type(engine),
                "exp1",
                False,
                False,
                untouched,
            ),
            "original changed, forced": (
                lambda engine, *_: self._add_image_type(engine),
                "exp1",
                True,
                False,
                merged,
            ),
            "database gone": (
                lambda _, __, ___, experiment_home: os.remove(
                    os.path.join(experiment_home, "autowisp.db")
                ),
                "exp1",
                False,
                False,
                untouched,
            ),
            "original not migrated": (
                lambda engine, *_: stamp_previous_revision(engine),
                "exp1",
                False,
                False,
                untouched,
            ),
            "experiment not migrated": (
                lambda _, __, experiment_engine, ___: stamp_previous_revision(
                    experiment_engine
                ),
                "exp1",
                False,
                False,
                untouched,
            ),
            "path outside, in the experiment": (
                lambda _, __, experiment_engine, ___: self._configure_outside(
                    experiment_engine
                ),
                "exp1",
                False,
                False,
                {**untouched, "result": "ConfigurationError"},
            ),
            # Another experiment's create, left going.
            "operation going in the original": (
                lambda engine, *_: self._add_live_operation(engine),
                "exp1",
                False,
                False,
                {**untouched, "unfinished": ["create experiment"]},
            ),
            "run going in the experiment": (
                lambda _, __, experiment_engine, ___: self._add_live_run(
                    experiment_engine, ignored=False
                ),
                "exp1",
                False,
                False,
                untouched,
            ),
            "run going, ignored": (
                lambda _, __, experiment_engine, ___: self._add_live_run(
                    experiment_engine, ignored=True
                ),
                "exp1",
                False,
                False,
                merged,
            ),
            # Restored after the original's files were moved.
            "too little space": (
                None,
                "exp1",
                False,
                True,
                {**untouched, "result": "ResourceError"},
            ),
            # Refused even if forced: the experiment lacks the run, which
            # fails the merge after the original's files were moved.
            "processed since the copy": (
                lambda engine, *_: self._add_run(engine, process_id=2),
                "exp1",
                True,
                False,
                untouched,
            ),
        }

        self.assertEqual(
            {
                scenario: self._try_merge(*parameters[:4])
                for scenario, parameters in test_scenarios.items()
            },
            {
                scenario: parameters[4]
                for scenario, parameters in test_scenarios.items()
            },
        )

    def test_delete_merge_backup(self):
        """The backup is deleted, unless an experiment is being merged."""

        engine, project_home, *_ = self._create_experiment()
        experiments = ProjectExperiments(engine, project_home)
        backup = experiments.merge("exp1")

        self._set_state(engine, "merging")
        with self.assertRaises(ExperimentError):
            experiments.delete_merge_backup()
        self.assertEqual(self._list_backups(project_home), [backup])
        self._set_state(engine, "current")
        self.assertEqual(experiments.delete_merge_backup(), [backup])
        self.assertEqual(self._list_backups(project_home), [])


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
