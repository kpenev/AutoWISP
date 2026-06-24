"""Unit tests for the crash-report credential scrubber."""

import tempfile
import unittest

from sqlalchemy import select

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Configuration, Parameter

# pylint: enable=no-name-in-module
from autowisp.crash_report import (
    REDACTED,
    scrub_config_values,
    scrub_mapping,
    scrub_text,
)


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


if __name__ == "__main__":
    unittest.main()
