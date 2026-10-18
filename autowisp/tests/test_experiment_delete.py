"""Tests of deleting an experiment."""

import os
import shutil
import subprocess
import sys
import time
import unittest

import psutil
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from autowisp.database.pipeline_runs import (
    get_machine_id,
    get_process_started,
)
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import ExperimentError, get_hostname
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import ProjectCopyCase

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestDelete(ProjectCopyCase):
    """Deleting an experiment: what refuses it, and what it stops."""

    # A run of another machine, which nothing here can stop.
    _other_machine = {"host": "other-host", "machine_id": "other-machine"}

    @classmethod
    def setUpClass(cls):
        """Give the project a raw image inside it, and a file it wrote."""

        super().setUpClass()
        cls._add_processed_files()

    def _start_process(self):
        """
        Start a stand-in for a run, with a child standing in for its workers.

        Both sleep, and are killed after the test if still there.

        Returns:
            subprocess.Popen:    The process.

            psutil.Process:    Its child, once started.
        """

        sleep = "import time; time.sleep(600)"
        # Released via addCleanup, since it must outlive this method.
        # pylint: disable-next=consider-using-with
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; "
                f"subprocess.Popen([sys.executable, '-c', {sleep!r}]); "
                + sleep,
            ]
        )
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)
        deadline = time.monotonic() + 30
        while not psutil.Process(process.pid).children():
            self.assertLess(
                time.monotonic(), deadline, "The child did not start."
            )
            time.sleep(0.05)
        child = psutil.Process(process.pid).children()[0]
        self.addCleanup(self._kill_if_alive, child)
        return process, child

    @staticmethod
    def _kill_if_alive(process):
        """Kill a process, unless it is already gone."""

        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass

    @staticmethod
    def _is_alive(process):
        """Is the process still running, rather than gone or a zombie?"""

        try:
            return process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    @staticmethod
    def _add_run(engine, **columns):
        """Record an unfinished run with the given columns; return its id."""

        with start_engine_session(engine) as db_session:
            pipeline_run = PipelineRun(started=func.now(), **columns)
            db_session.add(pipeline_run)
            db_session.flush()
            return pipeline_run.id

    @staticmethod
    def _mark_copy_run_going(engine, **columns):
        """Make the experiment's create look still going, as columns say."""

        with start_engine_session(engine) as db_session:
            db_session.execute(
                update(PipelineRun)
                .where(
                    PipelineRun.id
                    == select(Experiment.copy_run_id).scalar_subquery()
                )
                .values(finished=None, **columns)
            )

    @staticmethod
    def _set_state(engine, state):
        """Set the state of every experiment."""

        with start_engine_session(engine) as db_session:
            db_session.execute(update(Experiment).values(state=state))

    def _try_delete(self, prepare, name):
        """
        Delete an experiment in a copy of the project, as a scenario says.

        Args:
            prepare:    Given the original's engine, and the experiment's
                engine and home, prepares them and returns the ids of the
                runs to ignore; or None.

            name(str):    The name of the experiment to delete.

        Returns:
            tuple:    ``"deleted"``, or the name of the class of the exception
            delete raised; the state of experiment ``exp1``, None if it is
            gone; whether its directory is there; and the tasks of the
            operations' runs left unfinished.
        """

        engine, project_home, experiment_engine, experiment_home = (
            self._create_experiment()
        )
        experiments = ProjectExperiments(engine, project_home)
        ignored_runs = (
            prepare(engine, experiment_engine, experiment_home)
            if prepare
            else None
        ) or set()
        try:
            experiments.delete(name, ignored_runs)
            result = "deleted"
        except ExperimentError as error:
            result = type(error).__name__
        with Session(engine) as db_session:
            return (
                result,
                db_session.scalar(select(Experiment.state)),
                os.path.isdir(experiment_home),
                sorted(
                    db_session.scalars(
                        select(PipelineRun.task).where(
                            PipelineRun.task != PipelineRun.processing_task,
                            PipelineRun.finished.is_(None),
                        )
                    )
                ),
            )

    def test_delete(self):
        """Each delete on a copy of its own: refused, restored or completed."""

        processes = {
            "run going here, in the experiment": self._start_process(),
            "create going here": self._start_process(),
        }

        def this_machine(scenario):
            """Return the columns of a run of the scenario's process."""

            process_id = processes[scenario][0].pid
            return {
                "host": get_hostname(),
                "machine_id": get_machine_id(),
                "process_id": process_id,
                "process_started": get_process_started(process_id),
            }

        # Each scenario: what prepares the copy, given the original's engine
        # and the experiment's engine and home, returning the ids of the runs
        # to ignore; the name given; and the result, the experiment's state,
        # and whether its directory is left.
        test_scenarios = {
            "deleted": (None, "exp1", "deleted", None, False),
            "incomplete": (
                lambda engine, *_: self._set_state(engine, "creating"),
                "exp1",
                "deleted",
                None,
                False,
            ),
            # Deleting again completes an interrupted delete.
            "interrupted delete": (
                lambda engine, *_: self._set_state(engine, "deleting"),
                "exp1",
                "deleted",
                None,
                False,
            ),
            "directory gone": (
                lambda _, __, experiment_home: shutil.rmtree(experiment_home),
                "exp1",
                "deleted",
                None,
                False,
            ),
            "database gone": (
                lambda _, __, experiment_home: os.remove(
                    os.path.join(experiment_home, "autowisp.db")
                ),
                "exp1",
                "deleted",
                None,
                False,
            ),
            "no such experiment": (
                None,
                "exp2",
                "ExperimentError",
                "current",
                True,
            ),
            # Restored: refused before anything was removed.
            "run going elsewhere, in the experiment": (
                lambda _, experiment_engine, __: self._add_run(
                    experiment_engine, process_id=1, **self._other_machine
                )
                and set(),
                "exp1",
                "ExperimentError",
                "current",
                True,
            ),
            "create going elsewhere": (
                lambda engine, *_: self._mark_copy_run_going(
                    engine, **self._other_machine
                ),
                "exp1",
                "ExperimentError",
                "current",
                True,
            ),
            "run going elsewhere, ignored": (
                lambda _, experiment_engine, __: {
                    self._add_run(
                        experiment_engine, process_id=1, **self._other_machine
                    )
                },
                "exp1",
                "deleted",
                None,
                False,
            ),
            # Stopped, then deleted.
            "run going here, in the experiment": (
                lambda _, experiment_engine, __: self._add_run(
                    experiment_engine,
                    **this_machine("run going here, in the experiment"),
                )
                and set(),
                "exp1",
                "deleted",
                None,
                False,
            ),
            "create going here": (
                lambda engine, *_: self._mark_copy_run_going(
                    engine, **this_machine("create going here")
                ),
                "exp1",
                "deleted",
                None,
                False,
            ),
        }
        # The operations' runs each scenario leaves unfinished: only a create
        # going elsewhere, which the delete refuses to touch.
        left_unfinished = {"create going elsewhere": ["create experiment"]}

        self.assertEqual(
            {
                scenario: self._try_delete(prepare, name)
                for scenario, (prepare, name, *_) in test_scenarios.items()
            },
            {
                scenario: (
                    result,
                    state,
                    directory_left,
                    left_unfinished.get(scenario, []),
                )
                for scenario, (
                    *_,
                    result,
                    state,
                    directory_left,
                ) in test_scenarios.items()
            },
        )
        self.assertEqual(
            [
                scenario
                for scenario, (process, child) in processes.items()
                if process.poll() is None or self._is_alive(child)
            ],
            [],
            "Runs going here, or their children, were not stopped.",
        )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
