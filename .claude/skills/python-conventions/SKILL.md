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

- Keep a value local to its only user; a module-level constant is for
  something shared.

- Before writing a helper, a checker or a test fixture, look for an existing
  one to reuse or generalize: a near-copy of something already there is two
  things to keep in step. E.g. a test needing cameras generalized the
  fixture's own `_add_camera` rather than adding its own, and the
  configuration page's rule warnings reuse the expression library's.
