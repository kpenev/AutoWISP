---
name: python-conventions
description: Conventions for python code. Load before writing or editing any Python in this repository.
---
# Python Conventions

- Use snake_case for variable and function names

- Use CamelCase for class names

- Avoid single letter variables except when really obvious (e.g. x, y, ...)

- Avoid variable names that differ only by a single letter e.g. (positon and
  positions)

- Start function names with a verb: `_get_table`, not `_table`;
  `_find_image_products`, not `_image_products`. It matters most where
  functions are passed around or kept in tables, which is where a noun reads
  as data rather than as something to call.

- Keep a value local to its only user; a module-level constant is for
  something shared.

- Before writing a helper, a checker or a test fixture, look for an existing
  one to reuse or generalize: a near-copy of something already there is two
  things to keep in step. E.g. a test needing cameras generalized the
  fixture's own `_add_camera` rather than adding its own, and the
  configuration page's rule warnings reuse the expression library's.

- A docstring's summary is one line, within the 80 columns, indentation and
  quotes included; anything more goes in a paragraph after a blank line.
  Test docstrings too: a test's sentence that wraps onto a second line is
  the usual offender. Count the columns rather than estimate them; Black
  does not rewrap docstrings, and pylint flags only a line over 80, not a
  summary wrapped to stay under it.
