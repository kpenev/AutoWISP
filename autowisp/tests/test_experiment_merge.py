"""Tests of merging an experiment into its original."""

import json
import os
import shutil
import subprocess
import sys
import unittest
from contextlib import ExitStack, nullcontext
from pkgutil import resolve_name
from unittest import mock

from sqlalchemy import create_engine, func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import create_project_engine
from autowisp.database.migrate import migrate_project
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
from autowisp.tests import server_test_url
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    get_project_url,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable

# The status a merge crashing as a test asks exits with, which tells the
# crash from a traceback or a merge that finished.
_crash_status = 17


# A stand-in, called in place of the function it stands in for.
# pylint: disable-next=too-few-public-methods
class _Crash:
    """Stand in for a function, crashing or failing on one of its calls."""

    def __init__(self, target, crash_ncalls, action):
        """
        Pass every call on to the target, but the given one.

        Args:
            target(str):    The import path of the function stood in for.

            crash_ncalls(int):    Which call, counting from 1, crashes or
                fails.

            action(str):    ``"exit"`` to end the process there, as a crash
                would, with nothing cleaning up; or ``"fail"`` to raise
                ``OSError``.
        """

        self._original = resolve_name(target)
        self._crash_ncalls = crash_ncalls
        self._action = action
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        """Call the target, unless this is the call to crash or fail."""

        self._call_count += 1
        if self._call_count == self._crash_ncalls:
            if self._action == "exit":
                os._exit(_crash_status)
            raise OSError("Failing, as the test asks.")
        return self._original(*args, **kwargs)


def merge_crashing(project_home, crashes):
    """
    Merge experiment ``exp1`` of the project, crashing as a test asks.

    Run by the tests in a process of its own, which the crash ends.

    Args:
        project_home(str):    The original's home.

        crashes:    Each ``[target, crash_ncalls, action]``, as
            :class:`_Crash` takes them.
    """

    with ExitStack() as stack:
        for target, crash_ncalls, action in crashes:
            stack.enter_context(
                mock.patch(target, _Crash(target, crash_ncalls, action))
            )
        engine = create_project_engine(get_project_url(project_home))
        try:
            ProjectExperiments(engine, project_home).merge("exp1")
        finally:
            engine.dispose()


class TestMerge(ProjectCopyCase):
    """Merging an experiment: what refuses it, what it replaces and keeps."""

    # The raw image inside the project, which a merge leaves in place.
    _raw_image = os.path.join("RAW", "frame1.fits")

    @classmethod
    def setUpClass(cls):
        """Give the project a raw image, files it wrote, and other files.

        Enough files that moving the project into a backup takes several
        renames, which a crash can stop part way.
        """

        super().setUpClass()
        cls._add_processed_files()
        for fname in ["notes.txt", "LOGS/run.log"]:
            path = os.path.join(cls._template_home, *fname.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="ascii") as file:
                file.write(fname)

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
                        select(Experiment.run_id).filter_by(name="exp2")
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
                    if owner == "original"
                    or os.path.exists(os.path.join(home, "autowisp.db"))
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
                    .join(Experiment, Experiment.run_id == PipelineRun.id)
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

        def migrate_both(*engines_and_homes):
            """Upgrade the original from the release before, then the copy."""

            for db_engine, home in zip(
                engines_and_homes[::2], engines_and_homes[1::2]
            ):
                stamp_previous_revision(db_engine)
                migrate_project(db_engine, home, assume_backed_up=True)

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
            # Each upgraded in its turn, with the original's fingerprint
            # recorded again in the original's.
            "both migrated": (migrate_both, "exp1", False, False, merged),
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

    @staticmethod
    def _crash_merge(project_home, crashes):
        """
        Merge experiment ``exp1`` in a process of its own, crashing it.

        Args:
            project_home(str):    The original's home.

            crashes:    How the merge crashes (see :func:`merge_crashing`).

        Returns:
            bool:    Whether the process ended as the crash ends it, rather
            than by a traceback or with the merge finished.
        """

        # The code under test, whether the checkout's or installed.
        code_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        merge = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, sys\n"
                "from autowisp.tests.test_experiment_merge import "
                "merge_crashing\n"
                "merge_crashing(sys.argv[1], json.loads(sys.argv[2]))",
                project_home,
                json.dumps(crashes),
            ],
            env={**os.environ, "PYTHONPATH": code_root},
            capture_output=True,
            check=False,
        )
        return merge.returncode == _crash_status

    @staticmethod
    def _mark_merge_going(engine):
        """Make ``exp1`` look merged by a run of this process, still going."""

        with start_engine_session(engine) as db_session:
            db_session.execute(
                update(Experiment)
                .filter_by(name="exp1")
                .values(
                    state="merging",
                    run_id=record_run(db_session, "merge experiment").id,
                )
            )

    def _try_restore(self, crashes, prepare):
        """
        Restore the original after a merge crashed, as a scenario says.

        Args:
            crashes:    How the merge crashes (see :func:`merge_crashing`);
                None for no merge.

            prepare:    Given the original's engine, prepares it after the
                merge, if any; or None.

        Returns:
            tuple:    What the crash left: whether the merge ended as the
            crash ends it (None if there was none), whether there is a
            backup, whether the marker of all files moved is in it, and whose
            files the original holds. Then what the restore left: the state
            it returned, or the name of the class of the exception it raised;
            the experiment's state; whose files and processing runs the
            original holds; the number of backups; and the operations' runs
            left unfinished.
        """

        engine, project_home, *experiment = self._create_experiment()
        self._process_experiment(*experiment)
        homes = {"original": project_home, "experiment": experiment[1]}
        owner_trees, owner_runs = self._read_owners(
            homes, {"original": engine, "experiment": experiment[0]}
        )
        crashed = self._crash_merge(project_home, crashes) if crashes else None
        if prepare:
            prepare(engine)
        backups = self._list_backups(project_home)
        crash_left = (
            crashed,
            bool(backups),
            any(
                os.path.exists(os.path.join(backup, ".all-moved"))
                for backup in backups
            ),
            self._find_owner(owner_trees, self._read_tree(project_home)),
        )

        try:
            result = ProjectExperiments(engine, project_home).restore_merge(
                "exp1"
            )
        except ExperimentError as error:
            result = type(error).__name__
        with Session(engine) as db_session:
            return crash_left + (
                result,
                db_session.scalar(
                    select(Experiment.state).filter_by(name="exp1")
                ),
                self._find_owner(owner_trees, self._read_tree(project_home)),
                self._find_owner(owner_runs, self._read_runs(engine)),
                len(self._list_backups(project_home)),
                sorted(
                    db_session.scalars(
                        select(PipelineRun.task).where(
                            PipelineRun.task != PipelineRun.processing_task,
                            PipelineRun.finished.is_(None),
                        )
                    )
                ),
            )

    def test_restore_merge(self):
        """Each merge crashed at a different point, then restored.

        A crash ends the merge's process wherever the scenario says, with
        nothing cleaning up, by standing in for a function the merge calls
        there. What the crash left is checked first, so that a change to
        merge moving the crash elsewhere fails here, rather than testing
        another point unnoticed.
        """

        restored = ("current", "current", "original", "original", 0, [])
        # Each scenario: how the merge crashes, each crash a function stood
        # in for, on which call, and whether it crashes or fails there; what
        # prepares the original after the merge; what the crash left; and
        # what the restore left (see _try_restore).
        test_scenarios = {
            "before the backup": (
                [["os.makedirs", 1, "exit"]],
                None,
                (True, False, False, "original"),
                restored,
            ),
            "before moving": (
                [["autowisp.project_experiments.move_tree", 1, "exit"]],
                None,
                (True, True, False, "original"),
                restored,
            ),
            "part way moving": (
                [["os.rename", 2, "exit"]],
                None,
                (True, True, False, None),
                restored,
            ),
            "part way copying": (
                [["shutil.copy2", 2, "exit"]],
                None,
                (True, True, True, None),
                restored,
            ),
            # On a server, the backup is the database's first copy.
            "in the database's transaction": (
                [
                    [
                        "autowisp.project_experiments.copy_project_database",
                        1 if server_test_url() is None else 2,
                        "exit",
                    ]
                ],
                None,
                (True, True, True, "experiment"),
                restored,
            ),
            # The copy fails, and its rollback crashes.
            "part way rolling back": (
                [
                    ["shutil.copy2", 2, "fail"],
                    ["autowisp.project_experiments.remove_path", 1, "exit"],
                ],
                None,
                (True, True, True, None),
                restored,
            ),
            "not merging": (
                None,
                None,
                (None, False, False, "original"),
                ("ExperimentError",) + restored[1:],
            ),
            "merge going": (
                None,
                self._mark_merge_going,
                (None, False, False, "original"),
                (
                    "ExperimentError",
                    "merging",
                    "original",
                    "original",
                    0,
                    ["merge experiment"],
                ),
            ),
        }

        self.assertEqual(
            {
                scenario: self._try_restore(crashes, prepare)
                for scenario, (crashes, prepare, *_) in test_scenarios.items()
            },
            {
                scenario: crash_left + restore_left
                for scenario, (
                    *_,
                    crash_left,
                    restore_left,
                ) in test_scenarios.items()
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
