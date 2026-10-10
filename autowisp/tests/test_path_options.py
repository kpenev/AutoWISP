"""Check that path options are marked, and that their form is enforced."""

import contextlib
import io
import ntpath
import os
from os import path
import posixpath
import re
import unittest

from autowisp.exceptions import ConfigurationError
from autowisp.processing_steps import calibrate, fit_magnitudes
from autowisp.processing_steps.manual_util import (
    raise_config_parse_errors,
    stored_path_value,
)
from autowisp.project_paths import (
    check_path_form,
    get_project_home,
    set_project_home_path,
)
from autowisp.tests.test_parameter_description_length import (
    iter_step_modules,
)


def _looks_like_path(value):
    """True iff an option's default looks like a file or directory name."""

    # How the parser records the default of a multi-valued option.
    if value.startswith("[") and value.endswith("]"):
        entries = value[1:-1].split(", ")
    else:
        entries = [value]
    return any(
        entry.startswith(("{PROJHOME}", "{project_home}"))
        # Either system's rules: the defaults differ between the two.
        or posixpath.isabs(entry)
        or ntpath.isabs(entry)
        or re.search(r"\.(fits(\.fz)?|h5|hdf5|txt|log|outerr)$", entry)
        for entry in entries
    )


class TestPathOptions(unittest.TestCase):
    """Path options must say where they are: absolute or in the project."""

    def test_path_defaults_are_marked(self):
        """Every option whose default looks like a path is marked as one."""

        unmarked = set()
        found = set()
        for step_name, module in iter_step_modules():
            # The parsers report their defaults on stdout.
            with contextlib.redirect_stdout(io.StringIO()):
                config = module.parse_command_line([])
            for param, default in config["argument_defaults"].items():
                if isinstance(default, str) and _looks_like_path(default):
                    found.add(param)
                    if param not in config["path_arguments"]:
                        unmarked.add((step_name, param))

        # Guards against a detector that matches nothing.
        self.assertIn("stacked-master-fname", found)
        self.assertIn("anet-indices", found)
        self.assertEqual(
            unmarked,
            set(),
            "Options with path-like defaults not marked with path=: "
            + ", ".join(f"{step}: --{param}" for step, param in unmarked),
        )

    def test_path_forms(self):
        """Project paths may use {PROJHOME}; external ones must not."""

        absolute = path.abspath("MASTERS/x.fits")
        cases = [
            ("{PROJHOME}/MASTERS/x.fits", False, True),
            (absolute, False, True),
            ("MASTERS/x.fits", False, False),
            ("{PROJHOME}/anet/narrow", True, False),
            (absolute, True, True),
            ("anet/narrow", True, False),
        ]
        for fname, external, accepted in cases:
            with self.subTest(fname=fname, external=external):
                if accepted:
                    check_path_form(fname, external=external)
                else:
                    with self.assertRaises(ValueError):
                        check_path_form(fname, external=external)

    def test_parser_enforces_form(self):
        """A relative value is refused when parsed, whatever the option."""

        absolute = path.abspath("tmp")
        cases = [
            (fit_magnitudes, ["--qc-exclude-file", "exclude.txt"], False),
            (fit_magnitudes, ["--qc-exclude-file", "{PROJHOME}/x.txt"], True),
            (fit_magnitudes, ["--tempstore-dir", "{PROJHOME}/tmp"], False),
            (fit_magnitudes, ["--tempstore-dir", absolute], True),
            (calibrate, ["--master-bias", "R:MASTERS/zero_R.fits"], False),
            (
                calibrate,
                ["--master-bias", "R:{PROJHOME}/MASTERS/zero_R.fits"],
                True,
            ),
            (calibrate, ["--master-bias", "{PROJHOME}/zero_R.fits"], False),
        ]
        for module, args, accepted in cases:
            with self.subTest(args=args):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    raise_config_parse_errors(),
                ):
                    if accepted:
                        module.parse_command_line(args)
                    else:
                        with self.assertRaises(ConfigurationError):
                            module.parse_command_line(args)

    def test_stored_path_value(self):
        """Saved values move into {PROJHOME} form unless external."""

        home = path.abspath("project")
        inside = path.join(home, "MASTERS", "m.fits")
        outside = path.abspath(path.join("elsewhere", "m.fits"))
        cases = [
            (inside, True, "{PROJHOME}/MASTERS/m.fits"),
            (outside, True, outside),
            ("{PROJHOME}/DR/x.h5", True, "{PROJHOME}/DR/x.h5"),
            ("MASTERS/m.fits", True, None),
            (inside, "external", inside),
            ("{PROJHOME}/anet/narrow", "external", None),
            (
                f"[{inside}, {outside}]",
                True,
                f"[{{PROJHOME}}/MASTERS/m.fits, {outside}]",
            ),
            (f"[{inside}, MASTERS/m.fits]", True, None),
            ("R:" + inside, "channel", "R:{PROJHOME}/MASTERS/m.fits"),
            # A channel followed by a drive: absolute only on Windows.
            (
                r"B:C:\some\path",
                "channel",
                r"B:C:\some\path" if os.name == "nt" else None,
            ),
            (inside, "channel", None),
            (None, True, None),
        ]
        previous_home = get_project_home()
        set_project_home_path(home)
        try:
            for value, path_kind, expected in cases:
                with self.subTest(value=value, path_kind=path_kind):
                    if expected is None and value is not None:
                        with self.assertRaises(ConfigurationError):
                            stored_path_value(value, path_kind)
                    else:
                        self.assertEqual(
                            stored_path_value(value, path_kind), expected
                        )
        finally:
            set_project_home_path(previous_home)


if __name__ == "__main__":
    unittest.main()
