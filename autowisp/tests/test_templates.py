"""Checks over the browser interface's templates, as plain text.

No Django, no database and no rendering: these read the template files
and look for mistakes that Django does not report. That is the whole
reason they exist -- a template error that raises is found the first time
the page is opened, but the ones below fail *silently*, by putting
something on the page that was never meant to be seen.
"""

import unittest
from pathlib import Path

#: Every template shipped with the browser interface.
_templates = sorted(
    (Path(__file__).resolve().parents[1] / "browser_interface").rglob("*.html")
)


class TestTemplateComments(unittest.TestCase):
    """Comments have to be written in a form Django actually strips."""

    def test_templates_were_found(self):
        """Guards the guard: an empty sweep would pass everything."""

        self.assertTrue(_templates, "no templates found to check")

    def test_no_multiline_hash_comments(self):
        """``{# ... #}`` must open and close on one line.

        Django's short comment syntax does not span lines: the parser
        looks for the closing ``#}`` in the same token, so a comment
        written across several lines is not a comment at all and is
        rendered to the page as ordinary text. It raises nothing, so it
        reaches the user rather than the developer -- which is exactly
        what happened, above a banner on the processing progress page.

        ``{% comment %} ... {% endcomment %}`` is the multi-line form.
        """

        leaked = [
            f"{path.relative_to(Path(__file__).resolve().parents[2])}:{number}"
            for path in _templates
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            )
            if "{#" in line and "#}" not in line
        ]

        self.assertEqual(
            leaked,
            [],
            "these {# #} comments do not close on their own line, so Django "
            "renders them as text; use {% comment %} ... {% endcomment %}",
        )


if __name__ == "__main__":
    unittest.main()
