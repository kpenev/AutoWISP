---
name: bui-conventions
description: Conventions for BUI code
---
# BUI Conventions

- All styling should be done in CSS, not inline in the HTML. This allows for
  better separation of concerns and easier maintenance.

- JavaScript should be in .js files, not inline in the HTML. This allows for
  better separation of concerns and easier maintenance.

- Maximize reusability of components. If you find yourself copying and pasting
  code, it's a sign that you should create a base component (e.g. template) and
  inherit from it.
