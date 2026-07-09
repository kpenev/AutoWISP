"""Unit tests for the crash-report builder and its scrubbing helpers."""

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import select

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    Error,
    Image,
    ImageProcessingProgress,
    LightCurveProcessingProgress,
    Parameter,
    Step,
)

# pylint: enable=no-name-in-module
from autowisp.crash_report import (
    REDACTED,
    build_crash_report,
    collect_provenance,
    crash_report_main,
    find_error_progress,
    latest_error_id,
    scrub_config_values,
    scrub_mapping,
    scrub_text,
    select_error_logs,
)
from autowisp.error_persistence import persist_error
from autowisp.tests.error_fixtures import make_find_stars_error


class TestScrubText(unittest.TestCase):
    """scrub_text redacts secret values, leaves everything else intact."""

    def test_dict_repr_line(self):
        """Quoted secret values in a dict repr are redacted, others kept."""

        text = (
            "{'gaia_user': 'kpenev', 'gaia_password': 'hunter2', "
            "'project_home': '/data/proj'}"
        )
        out = scrub_text(text)
        self.assertNotIn("hunter2", out)
        self.assertNotIn("kpenev", out)
        self.assertIn(REDACTED, out)
        # Non-secret values survive.
        self.assertIn("/data/proj", out)
        self.assertIn("gaia_password", out)  # the key stays

    def test_ini_assignment(self):
        """An ini-style `key = value` secret is redacted to end of line."""

        out = scrub_text("gaia-password = my secret phrase\ngain = 1.0")
        self.assertNotIn("my secret phrase", out)
        self.assertIn("gaia-password = " + REDACTED, out)
        # An unrelated key on the next line is untouched.
        self.assertIn("gain = 1.0", out)

    def test_json_assignment(self):
        """A JSON-style secret value is redacted."""

        out = scrub_text('"api_key": "abcd1234"')
        self.assertNotIn("abcd1234", out)
        self.assertIn(REDACTED, out)

    def test_non_secret_untouched(self):
        """A non-secret key (even one containing 'pass' fragments) stays."""

        text = "password_hint = enabled\nusername = kpenev"
        out = scrub_text(text)
        # 'password_hint' is not the secret word 'password' (no boundary).
        self.assertEqual(out, text)

    def test_empty_input(self):
        self.assertEqual(scrub_text(""), "")
        self.assertIsNone(scrub_text(None))


class TestScrubMapping(unittest.TestCase):
    """scrub_mapping redacts values whose key names a secret."""

    def test_redacts_secret_keys(self):
        scrubbed = scrub_mapping(
            {
                "gaia_user": "kpenev",
                "gaia_password": "hunter2",
                "project_home": "/data/proj",
                "num_parallel_processes": 4,
            }
        )
        self.assertEqual(scrubbed["gaia_user"], REDACTED)
        self.assertEqual(scrubbed["gaia_password"], REDACTED)
        self.assertEqual(scrubbed["project_home"], "/data/proj")
        self.assertEqual(scrubbed["num_parallel_processes"], 4)

    def test_recurses_into_nested(self):
        scrubbed = scrub_mapping(
            {"credentials": {"token": "t0 ken", "user": "kpenev"}}
        )
        # The 'credentials' key itself is a secret name -> whole value
        # redacted.
        self.assertEqual(scrubbed["credentials"], REDACTED)

    def test_nested_non_secret_parent(self):
        scrubbed = scrub_mapping({"config": {"api_key": "abcd", "gain": 1.0}})
        self.assertEqual(scrubbed["config"]["api_key"], REDACTED)
        self.assertEqual(scrubbed["config"]["gain"], 1.0)

    def test_original_not_mutated(self):
        original = {"gaia_password": "hunter2"}
        scrub_mapping(original)
        self.assertEqual(original["gaia_password"], "hunter2")


class TestScrubConfigValues(unittest.TestCase):
    """scrub_config_values redacts secret configuration values in the DB."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_redacts_secret_config_only(self):
        """Secret-named config values are redacted; others are kept."""

        # pylint: disable=not-callable
        with start_db_session() as db_session:
            gaia_pw = Parameter(name="gaia-password", description="creds")
            gain = Parameter(name="gain", description="detector gain")
            db_session.add_all([gaia_pw, gain])
            db_session.flush()
            db_session.add_all(
                [
                    Configuration(
                        parameter_id=gaia_pw.id, version=0, value="Secret123"
                    ),
                    Configuration(parameter_id=gain.id, version=0, value="1.0"),
                ]
            )
            db_session.flush()
            gaia_param_id, gain_param_id = gaia_pw.id, gain.id
        # pylint: enable=not-callable

        with start_db_session() as db_session:
            redacted = scrub_config_values(db_session)

        self.assertEqual(redacted, 1)
        with start_db_session() as db_session:
            values = dict(
                db_session.execute(
                    # pylint: disable=no-member
                    select(Configuration.parameter_id, Configuration.value)
                ).all()
            )
        self.assertEqual(values[gaia_param_id], REDACTED)
        self.assertEqual(values[gain_param_id], "1.0")


class TestFindErrorProgress(unittest.TestCase):
    """find_error_progress resolves a step error to its processing record."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        set_project_home(self._tmp.name)

    def _add_progress(self, step_name, run_id, image_type_id):
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            step = db_session.scalar(select(Step).where(Step.name == step_name))
            if step is None:
                step = Step(name=step_name, description=step_name + " step")
                db_session.add(step)
                db_session.flush()
            progress = ImageProcessingProgress(
                run_id=run_id,
                step_id=step.id,
                image_type_id=image_type_id,
                configuration_version=0,
            )
            db_session.add(progress)
            db_session.flush()
            return progress.id
        # pylint: enable=not-callable

    def _add_lc_progress(self, step_name, run_id):
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            step = db_session.scalar(select(Step).where(Step.name == step_name))
            if step is None:
                step = Step(name=step_name, description=step_name + " step")
                db_session.add(step)
                db_session.flush()
            progress = LightCurveProcessingProgress(
                run_id=run_id,
                step_id=step.id,
                single_photref_id=None,
                configuration_version=0,
            )
            db_session.add(progress)
            db_session.flush()
            return progress.id
        # pylint: enable=not-callable

    def test_resolves_by_run_and_step(self):
        """An error's run + step pick out the matching progress."""

        progress_id = self._add_progress("calibrate", run_id=7, image_type_id=3)
        error = SimpleNamespace(
            pipeline_run_id=7, step_name="calibrate", image_id=None
        )
        progress = find_error_progress(error)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.id, progress_id)

    def test_resolves_lightcurve_step(self):
        """A lightcurve-step error resolves via the LC progress table.

        Regression for the crash-report gap where LC steps (tfa/epd/...)
        recorded in ``light_curve_processing_progress`` -- a different
        table than image steps -- could never resolve, so their logs were
        never collected.
        """

        progress_id = self._add_lc_progress("tfa", run_id=7)
        error = SimpleNamespace(
            pipeline_run_id=7, step_name="tfa", image_id=None
        )
        progress = find_error_progress(error)
        self.assertIsInstance(progress, LightCurveProcessingProgress)
        self.assertEqual(progress.id, progress_id)

    def test_none_for_stepless_or_runless(self):
        """A pipeline/BUI error (no step or run) resolves to nothing."""

        self._add_progress("calibrate", run_id=7, image_type_id=3)
        self.assertIsNone(
            find_error_progress(
                SimpleNamespace(
                    pipeline_run_id=7, step_name=None, image_id=None
                )
            )
        )
        self.assertIsNone(
            find_error_progress(
                SimpleNamespace(
                    pipeline_run_id=None, step_name="calibrate", image_id=None
                )
            )
        )

    def test_filters_by_image_type_when_image_known(self):
        """When the error names an image, the image's type disambiguates."""

        self._add_progress("calibrate", run_id=7, image_type_id=3)
        self._add_progress("calibrate", run_id=7, image_type_id=5)
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            image = Image(
                raw_fname="/data/raw/x.fits",
                image_type_id=5,
                observing_session_id=1,
            )
            db_session.add(image)
            db_session.flush()
            image_id = image.id
        # pylint: enable=not-callable

        progress = find_error_progress(
            SimpleNamespace(
                pipeline_run_id=7, step_name="calibrate", image_id=image_id
            )
        )
        self.assertEqual(progress.image_type_id, 5)

    def test_select_logs_empty_without_progress(self):
        """A stepless error selects no logs (and does not raise)."""

        error = SimpleNamespace(
            pipeline_run_id=None, step_name=None, image_id=None
        )
        self.assertEqual(select_error_logs(error), [])


class TestCollectProvenance(unittest.TestCase):
    """collect_provenance captures the current environment, JSON-safe."""

    def test_fields_present_and_serializable(self):
        """The provenance dict has the expected fields and serializes."""

        provenance = collect_provenance()
        for key in (
            "report_generated",
            "hostname",
            "platform",
            "python_version",
            "code_version",
            "packages",
        ):
            self.assertIn(key, provenance)
        # numpy is a hard dependency, so its version is always recorded.
        self.assertIn("numpy", provenance["packages"])
        self.assertIsInstance(provenance["packages"]["numpy"], str)
        # Everything must be JSON-serializable for the manifest.
        json.dumps(provenance)

    def test_unknown_package_omitted(self):
        """A package that is not installed is simply absent (no error)."""

        provenance = collect_provenance()
        self.assertNotIn(
            "definitely-not-a-real-package", provenance["packages"]
        )


class TestBuildCrashReport(unittest.TestCase):
    """build_crash_report assembles a scrubbed, best-effort zip."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        set_project_home(self._tmp.name)
        self._out = os.path.join(self._tmp.name, "report.zip")

    def _add_secret_config(self):
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            secret = Parameter(name="gaia-password", description="creds")
            db_session.add(secret)
            db_session.flush()
            db_session.add(
                Configuration(
                    parameter_id=secret.id, version=0, value="Secret123"
                )
            )
        # pylint: enable=not-callable

    def test_report_contents_and_scrubbing(self):
        """The zip holds the expected members; the DB copy is scrubbed."""

        self._add_secret_config()
        error_id = persist_error(make_find_stars_error())

        path = build_crash_report(error_id, self._out)

        with zipfile.ZipFile(path) as report:
            names = set(report.namelist())
            self.assertIn("error.json", names)
            self.assertIn("sidecar.json", names)
            self.assertIn("provenance.json", names)
            self.assertIn("manifest.json", names)
            self.assertIn("database/autowisp.db", names)

            manifest = json.loads(report.read("manifest.json"))
            self.assertEqual(manifest["error_id"], error_id)
            self.assertIn("error.json", manifest["collected"])

            # The live secret is redacted in the bundled database copy.
            db_copy = os.path.join(self._tmp.name, "from_zip.db")
            with open(db_copy, "wb") as out_db:
                out_db.write(report.read("database/autowisp.db"))
        connection = sqlite3.connect(db_copy)
        value = connection.execute(
            "SELECT c.value FROM configuration c JOIN parameter p "
            "ON c.parameter_id = p.id WHERE p.name = 'gaia-password'"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(value, REDACTED)

        # The live database is untouched (still holds the real secret).
        with start_db_session() as db_session:
            live = db_session.scalar(
                select(Configuration.value)  # pylint: disable=no-member
                .join(Parameter)
                .where(Parameter.name == "gaia-password")
            )
        self.assertEqual(live, "Secret123")

    def test_missing_sidecar_is_a_gap(self):
        """A sidecar-less error still builds, with the gap recorded."""

        error_id = persist_error(make_find_stars_error())
        with start_db_session() as db_session:
            db_session.get(Error, error_id).sidecar_path = None

        path = build_crash_report(error_id, self._out)

        with zipfile.ZipFile(path) as report:
            self.assertNotIn("sidecar.json", report.namelist())
            manifest = json.loads(report.read("manifest.json"))
        gaps = {gap["artifact"] for gap in manifest["gaps"]}
        self.assertIn("sidecar.json", gaps)

    def test_unknown_error_raises(self):
        """An unknown error id raises ValueError."""

        with self.assertRaises(ValueError):
            build_crash_report(999999, self._out)


class TestCrashReportCli(unittest.TestCase):
    """The wisp-crash-report CLI: latest-error lookup and the entry point."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        set_project_home(self._tmp.name)

    def _run_cli(self, *args):
        argv = ["wisp-crash-report", self._tmp.name, *args]
        with mock.patch("sys.argv", argv):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                crash_report_main()
        return out.getvalue()

    def test_latest_error_id(self):
        """latest_error_id returns the most recent error, None when empty."""

        self.assertIsNone(latest_error_id())
        first = persist_error(make_find_stars_error())
        second = persist_error(make_find_stars_error())
        self.assertEqual(latest_error_id(), max(first, second))

    def test_cli_writes_report_for_id(self):
        """Passing an id builds the report and prints where it went."""

        error_id = persist_error(make_find_stars_error())
        out_path = os.path.join(self._tmp.name, "report.zip")
        output = self._run_cli(str(error_id), "--out", out_path)

        self.assertTrue(os.path.exists(out_path))
        self.assertIn(str(error_id), output)
        with zipfile.ZipFile(out_path) as report:
            self.assertIn("manifest.json", report.namelist())

    def test_cli_last_resolves_most_recent(self):
        """--last targets the most recently recorded error."""

        persist_error(make_find_stars_error())
        latest = persist_error(make_find_stars_error())
        out_path = os.path.join(self._tmp.name, "last.zip")
        output = self._run_cli("--last", "--out", out_path)

        self.assertTrue(os.path.exists(out_path))
        self.assertIn(str(latest), output)

    def test_cli_errors_without_id_or_last(self):
        """Omitting both an id and --last exits with an argparse error."""

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self._run_cli()

    def test_cli_last_with_no_errors_exits(self):
        """--last with no recorded errors exits with an error."""

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self._run_cli("--last")


if __name__ == "__main__":
    unittest.main()
