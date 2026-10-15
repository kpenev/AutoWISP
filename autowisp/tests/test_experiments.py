"""Tests of a project's experiments: what locks it, what makes them stale."""

import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime

from sqlalchemy import (
    DDL,
    create_engine,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp import run_pipeline
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import set_project_home, start_db_session
from autowisp.database.migrate import migrate_project
from autowisp.database.pipeline_runs import get_machine_id, get_process_started
from autowisp.error_context import get_error_context, set_error_context
from autowisp.exceptions import (
    ConfigurationError,
    ExperimentError,
    ProjectLockedError,
    get_hostname,
)
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    Configuration,
    DiagnosticExpression,
    Error,
    Experiment,
    ImageType,
    Parameter,
    PipelineRun,
    Step,
)
from autowisp.database.data_model.provenance import Camera

# pylint: enable=no-name-in-module
from autowisp.database.data_model.steps_and_parameters import (
    step_param_association,
)
from autowisp.tests.database_fixtures import (
    add_cameras,
    add_create_run,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable


def _drop_timestamp_trigger(db_session, model):
    """Stop the database restamping the rows of the model's table.

    Triggers have no ORM form; :mod:`autowisp.database.data_model` creates
    them as DDL too.
    """

    db_session.connection().execute(
        DDL(f"DROP TRIGGER update_{model.__tablename__}_timestamp")
    )


class _ProjectCopyCase(unittest.TestCase):
    """A new project, created once, which each test changes copies of."""

    @classmethod
    def setUpClass(cls):
        """Create the project, leaving it the current one."""

        # Released via addClassCleanup, since it must outlive setUpClass.
        # pylint: disable-next=consider-using-with
        template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(template.cleanup)
        cls._template_home = template.name
        set_project_home(cls._template_home)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True)
        )

    def _copy_project(self):
        """Return an engine on a copy of the project, and the copy's home."""

        copy_home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, copy_home)
        db_fname = os.path.join(copy_home, "autowisp.db")
        shutil.copy(os.path.join(self._template_home, "autowisp.db"), db_fname)
        # No pool, so that nothing holds the file open when it is removed.
        engine = create_engine(f"sqlite:///{db_fname}", poolclass=NullPool)
        self.addCleanup(engine.dispose)
        return engine, copy_home

    @classmethod
    def _add_experiments(
        cls, engine, states, fingerprint="{}", db_session=None
    ):
        """Record experiments in the given states, ``{name: state}``.

        Each with the finished run that created it, as creating one leaves
        it. Within ``db_session`` if given, else in a transaction of its own
        on ``engine``.
        """

        if db_session is None:
            with Session(engine) as own_session, own_session.begin():
                cls._add_experiments(engine, states, fingerprint, own_session)
            return
        for name, state in states.items():
            db_session.add(
                Experiment(
                    name=name,
                    home=f"{{PROJHOME}}/experiments/{name}",
                    created=func.now(),
                    fingerprint=fingerprint,
                    state=state,
                    copy_run_id=add_create_run(db_session).id,
                )
            )


class TestStaleness(_ProjectCopyCase):
    """Migrating marks stale the experiments a merge would lose changes of."""

    @classmethod
    def setUpClass(cls):
        """Give the project a row in each table the changes touch."""

        super().setUpClass()
        with start_db_session() as db_session:
            add_cameras(db_session)
            run = PipelineRun(
                host="test-host", process_id=1, started=func.now()
            )
            db_session.add(run)
            db_session.flush()
            db_session.add(
                Error(
                    pipeline_run_id=run.id,
                    component="step",
                    exception_class="TestError",
                    user_message="Failed.",
                )
            )

    @staticmethod
    def _get_fingerprint(engine, project_home):
        """Return the project's fingerprint, as an experiment records it.

        Only creating an experiment produces one, and create does not exist
        yet (SUP-572); once it does, the tests create their experiments.
        """

        with engine.connect() as connection:
            return json.dumps(
                # pylint: disable-next=protected-access
                ExperimentRecords(engine, project_home)._get_fingerprint(
                    connection
                )
            )

    def test_changes_seen(self):
        """Each change, made to its own copy, makes stale only what it must.

        Every copy holds an experiment of each state, recorded as the project
        stood: the current one becomes stale if the change is one a merge
        would lose, while the others keep their states whatever it is.
        """

        def update_within_the_second(db_session):
            # Without the trigger the timestamp stays as it was, as it does
            # for an update within the same second.
            _drop_timestamp_trigger(db_session, Step)
            return db_session.execute(
                update(Step)
                .where(Step.id == select(func.min(Step.id)).scalar_subquery())
                .values(description=Step.description + " (edited)")
            )

        def delete_unkeyed_row(db_session):
            first = db_session.execute(
                select(step_param_association)
                .order_by(*step_param_association.columns)
                .limit(1)
            ).one()
            return db_session.execute(
                delete(step_param_association).where(
                    step_param_association.c.step_id == first.step_id,
                    step_param_association.c.param_id == first.param_id,
                )
            )

        def restamp(db_session):
            # Without the trigger the timestamp certainly changes, rather
            # than being set to the second it already holds.
            _drop_timestamp_trigger(db_session, Parameter)
            return db_session.execute(
                update(Parameter).values(timestamp=datetime(2000, 1, 1))
            )

        # Each change: the function making it, and whether a merge would
        # lose it, making the current experiment stale. The function returns
        # the result of its update or delete, if it has one.
        changes = {
            "value updated within the second": (update_within_the_second, True),
            "row deleted": (
                lambda db_session: db_session.execute(
                    delete(DiagnosticExpression)
                ),
                True,
            ),
            "row inserted": (
                lambda db_session: db_session.add(
                    ImageType(name="test", description="Added by the test.")
                ),
                True,
            ),
            "row deleted without a primary key": (delete_unkeyed_row, True),
            "notes edited": (
                lambda db_session: db_session.execute(
                    update(Camera).values(notes="Edited.")
                ),
                True,
            ),
            "run finished": (
                lambda db_session: db_session.execute(
                    update(PipelineRun).values(finished=func.now())
                ),
                True,
            ),
            "restamped": (restamp, False),
            "error added": (
                lambda db_session: db_session.add(
                    Error(
                        component="bui",
                        exception_class="TestError",
                        user_message="Failed.",
                    )
                ),
                False,
            ),
            "error resolved": (
                lambda db_session: db_session.execute(
                    update(Error).values(resolved=func.now())
                ),
                False,
            ),
            "experiment added, with the run creating it": (
                lambda db_session: self._add_experiments(
                    None, {"exp4": "creating"}, db_session=db_session
                ),
                False,
            ),
            "processing run added": (
                lambda db_session: db_session.add(
                    PipelineRun(
                        host="test-host", process_id=2, started=func.now()
                    )
                ),
                True,
            ),
        }
        states = {"exp1": "current", "exp2": "stale", "exp3": "creating"}

        outcome = {}
        for change, (make_change, _) in changes.items():
            engine, project_home = self._copy_project()
            stamp_previous_revision(engine)
            self._add_experiments(
                engine, states, self._get_fingerprint(engine, project_home)
            )
            with Session(engine) as db_session, db_session.begin():
                result = make_change(db_session)
                if result is not None:
                    self.assertGreater(
                        result.rowcount, 0, f"{change!r} changed nothing."
                    )

            migration = migrate_project(engine, project_home)
            outcome[change] = (
                {
                    experiment.name: experiment.state
                    for experiment in ExperimentRecords(
                        engine, project_home
                    ).get_experiments()
                    if experiment.name in states
                },
                migration["stale"],
            )

        self.assertEqual(
            outcome,
            {
                change: (
                    dict(states, exp1="stale" if seen else "current"),
                    ["exp1"] if seen else [],
                )
                for change, (_, seen) in changes.items()
            },
        )


class TestProjectLock(_ProjectCopyCase):
    """A project is locked while it has experiments that are not stale."""

    @staticmethod
    def _get_locking(experiments):
        """Return the states of the experiments a refusal names, by name."""

        try:
            experiments.refuse_if_locked()
        except ProjectLockedError as error:
            return error.details["experiments"]
        return {}

    def test_lock_and_unlock(self):
        """Experiments not yet stale lock the project; unlocking frees it of
        the current ones, while one still being created keeps it locked."""

        # Each project: the states of its experiments, None for a project
        # that is itself an experiment, those that must lock it, and those
        # that must still lock it once unlocked.
        projects = {
            "an experiment": (None, {}, {}),
            "without experiments": ({}, {}, {}),
            "with stale experiments": (
                {"exp1": "stale", "exp2": "stale"},
                {},
                {},
            ),
            "with experiments in every state": (
                {"exp1": "current", "exp2": "creating", "exp3": "stale"},
                {"exp1": "current", "exp2": "creating"},
                {"exp2": "creating"},
            ),
        }

        locking = {}
        unlocked = {}
        locking_after = {}
        for project, (states, _, _) in projects.items():
            engine, project_home = self._copy_project()
            if states is None:
                Experiment.__table__.drop(engine)
            else:
                self._add_experiments(engine, states)
            experiments = ExperimentRecords(engine, project_home)
            locking[project] = self._get_locking(experiments)
            unlocked[project] = experiments.unlock()
            locking_after[project] = self._get_locking(experiments)

        self.assertEqual(
            locking,
            {project: before for project, (_, before, _) in projects.items()},
        )
        self.assertEqual(
            unlocked,
            {
                project: sorted(set(before) - set(after))
                for project, (_, before, after) in projects.items()
            },
        )
        self.assertEqual(
            locking_after,
            {project: after for project, (_, _, after) in projects.items()},
        )

    def test_run_refused(self):
        """A run in a locked project is refused as it starts, and ended."""

        engine, project_home = self._copy_project()
        self._add_experiments(engine, {"exp1": "current"})
        image_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, image_dir)
        # The run makes itself the context of every error recorded later,
        # which a test of another project must not inherit.
        self.addCleanup(set_error_context, get_error_context())

        # No steps, as the BUI adds images: only the check at the start of
        # the run can refuse it.
        with self.assertRaises(ProjectLockedError):
            run_pipeline.main(
                Namespace(
                    project_home=project_home,
                    add_raw_images=[image_dir],
                    steps=[],
                    step_imtypes=[],
                )
            )
        # The refused run, not the one that created the experiment.
        with Session(engine) as db_session:
            self.assertEqual(
                [
                    run.finished is not None
                    for run in db_session.scalars(
                        select(PipelineRun).filter_by(
                            task=PipelineRun.processing_task
                        )
                    )
                ],
                [True],
            )


class TestRefusals(_ProjectCopyCase):
    """What refuses the operations on experiments, and runs in them."""

    def test_active_runs(self):
        """Runs going here and elsewhere refuse, unless ignored by their ids.

        Those found ended stay recorded as finished, though the check refuses.
        """

        engine = self._copy_project()[0]
        this_machine = {
            "host": get_hostname(),
            "machine_id": get_machine_id(),
            "process_id": os.getpid(),
            "process_started": get_process_started(),
        }
        other_machine = {"host": "other-host", "machine_id": "other-machine"}
        # Each run's columns besides its start; all unfinished but the first.
        runs = {
            "finished": {**this_machine, "finished": func.now()},
            "going": this_machine,
            "operation going": {**this_machine, "task": "create experiment"},
            # Its process id since taken by this test's, which started later.
            "ended": {
                **this_machine,
                "process_started": this_machine["process_started"] - 3600,
            },
            "elsewhere": {**other_machine, "process_id": 1},
            "ignored": {**other_machine, "process_id": 2},
        }
        ids = {}
        with Session(engine) as db_session, db_session.begin():
            for name, columns in runs.items():
                pipeline_run = PipelineRun(started=func.now(), **columns)
                db_session.add(pipeline_run)
                db_session.flush()
                ids[name] = pipeline_run.id

        # pylint: disable=protected-access
        with self.assertRaises(ExperimentError) as refusal:
            ExperimentRecords._refuse_active_runs(engine, {ids["ignored"]})
        with Session(engine) as db_session:
            finished = set(
                db_session.scalars(
                    select(PipelineRun.id).where(
                        PipelineRun.finished.is_not(None)
                    )
                )
            )
        # Raises if ignoring fails for any of them.
        ExperimentRecords._refuse_active_runs(engine, set(ids.values()))
        # pylint: enable=protected-access

        self.assertEqual(
            refusal.exception.details["runs"],
            sorted(
                ids[name] for name in ("going", "operation going", "elsewhere")
            ),
        )
        self.assertEqual(finished, {ids["finished"], ids["ended"]})

    def test_external_paths(self):
        """A run in an experiment is refused for each path configured outside
        it, under any condition; in its original, or as created, it is not."""

        outside = os.path.abspath("outside")
        # Each value configured: the parameter, whether the value is added
        # under another condition rather than replacing the parameter's, the
        # value, and whether it is refused.
        configured = [
            ("lc-fname", False, os.path.join(outside, "{:d}.h5"), True),
            (
                "master-mask",
                False,
                "[R:{PROJHOME}/mask_R.fits, "
                f"G:{os.path.join(outside, 'mask_G.fits')}]",
                True,
            ),
            ("subpixmap", False, os.path.join(outside, "subpixmap"), False),
            (
                "epd-statistics-fname",
                True,
                os.path.join(outside, "epd.txt"),
                True,
            ),
            ("tfa-statistics-fname", True, "{PROJHOME}/tfa.txt", False),
        ]

        def configure(db_session):
            condition_id = db_session.scalar(select(func.max(Condition.id)))
            for name, added, value, _ in configured:
                parameter_id = db_session.scalar(
                    select(Parameter.id).filter_by(name=name)
                )
                if added:
                    db_session.add(
                        Configuration(
                            parameter_id=parameter_id,
                            version=0,
                            condition_id=condition_id,
                            value=value,
                        )
                    )
                else:
                    db_session.execute(
                        update(Configuration)
                        .filter_by(parameter_id=parameter_id)
                        .values(value=value)
                    )

        # Each project: whether it is an experiment, whether it is configured
        # as above, and the parameters refused.
        projects = {
            "original": (False, True, []),
            "experiment": (
                True,
                True,
                sorted(name for name, _, _, refused in configured if refused),
            ),
            "experiment as created": (True, False, []),
        }

        refused = {}
        for project, (is_experiment, is_configured, _) in projects.items():
            engine, project_home = self._copy_project()
            if is_configured:
                with Session(engine) as db_session, db_session.begin():
                    configure(db_session)
            if is_experiment:
                Experiment.__table__.drop(engine)
            try:
                ProjectExperiments(engine, project_home).refuse_run()
                refused[project] = []
            except ConfigurationError as error:
                refused[project] = error.details["parameters"]

        self.assertEqual(
            refused,
            {
                project: expected
                for project, (_, _, expected) in projects.items()
            },
        )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
