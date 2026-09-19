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
