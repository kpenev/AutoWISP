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

- Do **not** style a full-width callout or banner with `lcars-element` /
  `lcars-u-*`. Those are LCARS layout *size units*, not "use full width": they
  pin a fixed ~7.5rem x 3rem right/bottom-aligned box, so a message overflows
  and clips. Render a plain block with the LCARS background colour only, e.g.
  `<div class="lcars-red-alert-bg" style="width:100%; box-sizing:border-box;
  padding:...; color:#000; font-weight:bold;">`. This is the pattern set by
  commit 542f50b5 ("Make error-page callouts full-width").

- A BUI change needs `pip install .` (never `-e` — it breaks BUI styling) plus a
  browser hard-refresh (Cmd-Shift-R) to take effect. Editing CSS/JS alone will
  silently no-op against a cached stylesheet.

- **Name another BUI app by its full path**, `from
  autowisp.browser_interface.diagnostics.quantities import ...`, never with a
  relative `..` and never by the bare app name. That holds for the strings too
  — `INSTALLED_APPS`, each `AppConfig.name`, `ROOT_URLCONF`,
  `WSGI_APPLICATION`, `DJANGO_SETTINGS_MODULE`, the context processors and
  every `include()`. Relative imports *within* one app (`from .log_views
  import ...`) are fine.

  `manage.py` is run by path, which still leaves `browser_interface` on
  `sys.path[0]`, so a bare `core.models` would import and run — under a second
  module name, giving Django two model classes for one app label and the
  documentation build an unimportable module. The app labels (`home`, `core`,
  …) come from the last component of the full path, so the tables and the
  migrations are unaffected by the spelling.

  Nothing in the suite catches a bare name on its own; `python -m django check`
  with only the repository on `PYTHONPATH` does, since `browser_interface` is
  then not on the path at all.
