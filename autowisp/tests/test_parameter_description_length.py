"""Check that step argument help fits the parameter description column.

Project initialization stores the command line help of every processing
step argument in ``parameter.description``. SQLite ignores declared
``VARCHAR`` lengths, so an over-long help string is invisible locally and
only breaks initialization against a centralised MySQL/MariaDB database,
where it aborts the insert with "Data too long for column
'description'". This keeps the help strings within the column instead.
"""

import contextlib
import io
import unittest

from autowisp import processing_steps
from autowisp.database.data_model.steps_and_parameters import Parameter, Step


def _iter_step_modules():
    """Yield ``(name, module)`` for each processing step of the pipeline."""

    for name in sorted(dir(processing_steps)):
        if name.startswith("_"):
            continue
        module = getattr(processing_steps, name)
        if hasattr(module, "parse_command_line"):
            yield name, module


class TestParameterDescriptionLength(unittest.TestCase):
    """Everything initialization inserts must fit its column."""

    @classmethod
    def setUpClass(cls):
        """Collect the descriptions initialization would store."""

        cls.step_docs = {}
        cls.descriptions = {}
        for step_name, module in _iter_step_modules():
            cls.step_docs[step_name] = module.__doc__ or ""
            # The parsers report their defaults on stdout.
            with contextlib.redirect_stdout(io.StringIO()):
                config = module.parse_command_line([])
            for param, description in config["argument_descriptions"].items():
                if isinstance(description, dict):
                    description = description["help"]
                cls.descriptions[(step_name, param)] = description or ""

    def test_argument_help_fits_description_column(self):
        """No argument help exceeds ``Parameter.description``."""

        max_length = Parameter.__table__.c.description.type.length
        too_long = {
            key: len(description)
            for key, description in self.descriptions.items()
            if len(description) > max_length
        }
        self.assertEqual(
            too_long,
            {},
            f"Argument help longer than the {max_length} character "
            "parameter.description column (breaks project initialization "
            "on MySQL/MariaDB): "
            + ", ".join(
                f"{step}: --{param} ({length} chars)"
                for (step, param), length in sorted(too_long.items())
            ),
        )

    def test_step_docstring_fits_description_column(self):
        """No step docstring exceeds ``Step.description``."""

        max_length = Step.__table__.c.description.type.length
        too_long = {
            step: len(doc)
            for step, doc in self.step_docs.items()
            if len(doc) > max_length
        }
        self.assertEqual(
            too_long,
            {},
            f"Step docstring longer than the {max_length} character "
            "step.description column: "
            + ", ".join(
                f"{step} ({length} chars)"
                for step, length in sorted(too_long.items())
            ),
        )


if __name__ == "__main__":
    unittest.main()
