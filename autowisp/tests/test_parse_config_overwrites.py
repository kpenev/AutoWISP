"""Unit tests for ``user_interface.parse_config_overwrites``."""

import unittest

from autowisp.database.user_interface import parse_config_overwrites


class TestParseConfigOverwrites(unittest.TestCase):
    """Edge-case coverage for the configuration-line → overwrites parser."""

    def test_key_equals_value(self):
        """Plain ``key = value`` produces ``{key: [(None, 'value')]}``."""

        self.assertEqual(
            parse_config_overwrites(["foo = bar"]),
            {"foo": [(None, "bar")]},
        )

    def test_colon_separator(self):
        """``:`` is accepted as an alternative to ``=``."""

        self.assertEqual(
            parse_config_overwrites(["foo: bar"]),
            {"foo": [(None, "bar")]},
        )

    def test_blank_lines_skipped(self):
        """Blank / whitespace-only lines do not appear in the output."""

        self.assertEqual(
            parse_config_overwrites(["", "   ", "\t\n", "foo = bar"]),
            {"foo": [(None, "bar")]},
        )

    def test_comment_only_lines_skipped(self):
        """Lines starting with ``#`` are skipped entirely."""

        self.assertEqual(
            parse_config_overwrites(
                ["# a comment", "   # indented", "foo = bar"]
            ),
            {"foo": [(None, "bar")]},
        )

    def test_section_headers_skipped(self):
        """``[section]`` headers from INI-style files are ignored."""

        self.assertEqual(
            parse_config_overwrites(
                ["[general]", "foo = bar", "[calibrate]", "baz = qux"]
            ),
            {"foo": [(None, "bar")], "baz": [(None, "qux")]},
        )

    def test_inline_comment_stripped(self):
        """A trailing ``#`` comment is not part of the value."""

        self.assertEqual(
            parse_config_overwrites(["foo = bar  # trailing"]),
            {"foo": [(None, "bar")]},
        )

    def test_semicolon_preserved_in_value(self):
        """``;`` is data, not a comment marker -- preserve it verbatim."""

        self.assertEqual(
            parse_config_overwrites(["foo = a;b;c"]),
            {"foo": [(None, "a;b;c")]},
        )

    def test_single_quoted_value(self):
        """Single quotes are stripped; their inner content is kept verbatim."""

        self.assertEqual(
            parse_config_overwrites(["foo = 'a value with spaces'"]),
            {"foo": [(None, "a value with spaces")]},
        )

    def test_double_quoted_value(self):
        """Double quotes are stripped; their inner content is kept verbatim."""

        self.assertEqual(
            parse_config_overwrites(['foo = "value with = and # inside"']),
            {"foo": [(None, "value with = and # inside")]},
        )

    def test_key_without_value(self):
        """A bare key (no separator) is recorded with value ``None``."""

        self.assertEqual(
            parse_config_overwrites(["flag"]),
            {"flag": [(None, None)]},
        )

    def test_key_with_empty_value(self):
        """``key =`` (empty value) is recorded as ``None``."""

        self.assertEqual(
            parse_config_overwrites(["key ="]),
            {"key": [(None, None)]},
        )

    def test_dash_and_dot_in_key(self):
        """Parameter names with ``-`` and ``.`` are valid."""

        self.assertEqual(
            parse_config_overwrites(["num-parallel-processes = 4"]),
            {"num-parallel-processes": [(None, "4")]},
        )

    def test_later_value_wins(self):
        """A repeated key keeps the last value seen."""

        self.assertEqual(
            parse_config_overwrites(["foo = first", "foo = second"]),
            {"foo": [(None, "second")]},
        )

    def test_non_parameter_keys_excluded(self):
        """``project-home`` and ``split-channels`` are silently dropped.

        Mirrors what
        ``autowisp/browser_interface/home/static/home/js/create.project.js``
        strips before populating the BUI textarea; these are never valid
        DB parameter overrides.
        """

        self.assertEqual(
            parse_config_overwrites(
                [
                    "project-home = .",
                    "split-channels = 'R(0,1;0,1)'",
                    "verbose = debug",
                ]
            ),
            {"verbose": [(None, "debug")]},
        )

    def test_test_cfg_like_block(self):
        """Sanity check on a multi-section block resembling test.cfg."""

        lines = [
            "[general]",
            "project-home = .",
            "verbose = debug",
            "",
            "[calibrate]",
            "saturation-threshold = 15000",
            "fnum = 'int(RAWFNAME.split(\"-\")[1])'",
        ]
        self.assertEqual(
            parse_config_overwrites(lines),
            {
                "verbose": [(None, "debug")],
                "saturation-threshold": [(None, "15000")],
                "fnum": [(None, 'int(RAWFNAME.split("-")[1])')],
            },
        )


if __name__ == "__main__":
    unittest.main()
