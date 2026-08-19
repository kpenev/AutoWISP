# Diagnostic Expressions in the BUI

## Context

The BUI diagnostics app lets users plot per-image diagnostics (`bg_center`,
`s_center`, `astrom_residual`, `pointing_offset`, `pixel_q999`, …) against time
or against each other. Those quantities are fixed: whatever a processing step
recorded into `ImageDiagnostics` is exactly what can be plotted. There is no way
to look at a *derived* quantity — a normalised residual
`astrom_residual / diagonal_fov`, a night-relative background
`bg_center - median(bg_center)`, or a contrast ratio between pixel quantiles —
without leaving the BUI and writing a script.

This adds user-defined **diagnostic expressions**: named mathematical
expressions over the recorded diagnostics, stored in the BUI database (the
Django DB that holds the project list, shared by all projects), which then
appear in the diagnostic selector dropdowns alongside the built-in diagnostics
and plot through exactly the same machinery. Expressions can be built out of
other expressions, and can be exported to and imported from a JSON file so
users can share useful ones.

Decisions taken with the user:

- **Variables** = image diagnostics only (`ImageDiagnostics` /
  `DiagnosticType`). Not photometry diagnostics, image metadata, or detrending
  statistics.
- **Scope** = global to all projects. An expression whose variables are not
  recorded in the currently-open project is simply hidden from that project's
  dropdown.
- **Edit UI** = a new page in the diagnostics app.

## Design

### Namespace

Expressions live in the **same flat name space** as `DiagnosticType` names. This
is what keeps the change small: the URL patterns
(`image/<slug:diagnostic_name>`, `image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>`),
`diagnostics_app.html`'s two `<select>`s, and `diagnostics_app.js`'s
`navigateDiagnostics()` all keep working untouched — an expression is just
another name in `available_diagnostics`.

Consequences to handle explicitly:

- Names are restricted to Django's slug charset (`SlugField`) so they survive
  the URL converters.
- Saving/importing rejects a name that collides with a built-in diagnostic. The
  reserved set is the static list seeded by
  `_init_diagnostic_types()` in `autowisp/database/initialize_database.py:448`,
  plus `quantiles` and anything matching `pixel_q*`, plus any `DiagnosticType`
  name present in the currently-open project DB.
- If a future release ships a `DiagnosticType` matching an existing expression
  name, the **real diagnostic wins** at resolution time and the management page
  flags the expression as shadowed. The no-collision rule carries extra weight
  because expression names are themselves variables inside other expressions
  (see *Composition* below) — a collision would be genuinely ambiguous there,
  not merely confusing in a dropdown.

### Evaluation

Reuse `autowisp.evaluator.Evaluator` (`autowisp/evaluator.py`) — the asteval
wrapper already used for every other user expression in the pipeline. It takes
a dict of `{name: numpy array}` and evaluates **vectorized** over all images at
once, which is what makes `bg_center - median(bg_center)` work for free
(asteval's default symtable carries the numpy/math names).

### Alignment — one canonical image list per session, NaN elsewhere

Every array is built against the **same canonical image list**, with `NaN`
wherever a value does not exist. Alignment is then structural: index *i* is the
same image in every array, for every diagnostic and every expression, with no
join anywhere.

The canonical list is simply:

```sql
SELECT id, jd FROM image
WHERE observing_session_id = ? AND jd IS NOT NULL
ORDER BY jd
```

No image-type filter, no requirement that any diagnostic be present, no join to
`image_diagnostics` — and therefore no `channel` in it either, so it is a
function of the **observing session alone** and is shared by every channel.
Images with nothing recorded for a given channel are just NaN across the board
there.

Requiring "at least one diagnostic in this channel" would buy nothing: those
rows are masked out before anything reads them, so the plots are identical
either way. It would only add a join to the query and a channel argument to the
signature. A session holds a manageable number of images (see *Scaling*), so
the extra padding costs nothing that matters.

This is what makes missing data a non-problem rather than a special case:

- `NaN` propagates through numpy arithmetic on its own, so an expression is
  automatically undefined exactly where its inputs are. A `bg_center`-based
  expression is NaN on every calibration frame, which is the right answer.
- Two axes of a scatter are aligned by construction. The old inner join
  survives only as a one-line finite mask.
- Each array is built independently against the canonical list, so nothing
  depends on evaluating intermediates over a matched image subset.

**Mask non-finite values inside `plot_image_diagnostic_series`, immediately
before `axes.scatter`.** `get_*_series_data` returns padded arrays and nothing
upstream filters them. Three lines at that one choke point:

```python
keep = numpy.isfinite(time_values) & numpy.isfinite(diag_values)
time_values, diag_values = time_values[keep], diag_values[keep]
image_ids = numpy.asarray(image_ids)[keep]
```

This is the simplest possible placement, and it is the only one that is
genuinely a single site: `plot_image_diagnostic_series` is shared by the
time-series and the diagnostic-vs-diagnostic paths, and the same
`isfinite(x) & isfinite(y)` covers both — for a time series the x values are
JDs, which the canonical list guarantees are non-null, so the mask reduces to
`isfinite(y)` on its own.

The one thing this placement must get right, and does: `image_ids` is indexed
with the *same* mask, so `collection.set_urls()` — the per-point click-through
to the calibrated frame — stays aligned with the drawn markers, without
depending on how matplotlib treats NaN offsets in `_iter_collection`.

Two knock-on effects are **accepted deliberately**, because `min_jd` and the
subplot grouping are heuristics and not worth complicating the data path for:

- `min_jd` (the x-axis offset) is now the first image of the observing
  session rather than the first image carrying the plotted quantity, so the
  `JD - …` label may differ from today.
- `group_series_by_jd_overlap` now sees every series of a session spanning that
  session's full time range. Series from one session therefore always share a
  subplot, and series from different sessions merge iff the sessions overlap in
  time. This is arguably steadier than today's behaviour, where the grouping
  shifts depending on which images happen to carry the chosen diagnostic.

One small guard does need to move with it. `create_image_diagnostics_figure`
currently skips a series via `if jd_values.size:`, which no longer detects an
empty series — a padded array is full-length even when every value is NaN, so
an empty selection would claim a blank subplot. Change that test to
`if numpy.any(numpy.isfinite(diag_values)):`. One line, and it keeps today's
skip-empty-series behaviour without pulling the mask upstream.

**The one real cost:** numpy's plain aggregates propagate NaN, so
`bg_center - median(bg_center)` goes all-NaN if a single image lacks
`bg_center`. Add `nanmedian`, `nanmean`, `nanstd`, `nanmin`, `nanmax`,
`nansum`, `nanpercentile` to the evaluator's symtable — exactly as
`LightCurveEvaluator` already does for `nanmean`/`nanmedian` in
`autowisp/evaluator.py` — and document that aggregates want the `nan*` forms.
Do **not** silently rebind `median` to `nanmedian`: an expression shared
through an export file has to mean what it says. The all-NaN trap is caught
instead by the two bounded checks under *Scaling*.

Note also that these aggregates are **per (session, channel)**, not global:
`nanmedian(bg_center)` is the median over the plotted session, not over the
whole archive. That is the useful meaning for night-relative quantities, and it
is the only one that stays affordable at scale. Say so in the documentation.

### Evaluation

Per (observing session, channel) series:

1. Determine what the target expression needs, by `ast.parse(text,
   mode="eval")` and collecting `ast.Name` ids. Ids matching another
   expression's name are dependencies to evaluate first; ids matching a
   `DiagnosticType` name are real diagnostics to fetch. Walking the dependency
   graph gives the **evaluation order** and the transitive set of real
   diagnostics needed (see *Composition*).
2. One query for `(Image.id, Image.jd, DiagnosticType.name,
   ImageDiagnostics.value)` restricted to that session/channel and
   `DiagnosticType.name.in_(needed)`, ordered by `Image.jd`.
3. Scatter the results into `{name: numpy array}`, each array pre-filled with
   `NaN` at the canonical list's length.
4. Evaluate in order into one `Evaluator`, then read off the target:

   ```python
   evaluate = Evaluator(padded)
   for name in evaluation_order:
       evaluate.symtable[name] = evaluate(expressions[name])
   values = evaluate.symtable[target]
   ```

   `numpy.atleast_1d` + broadcast to the image count so a constant-valued
   expression still plots.

The per-series count shown in the table is a **SQL aggregate only** — the
number of images having all the needed diagnostics, from the
`HAVING COUNT(DISTINCT diagnostic_type.id) == len(needed)` subquery. It is an
upper bound on the number of plotted points, since arithmetic can still produce
NaN, and the column should be labelled as "images with the required inputs"
rather than implying a point count. Do not evaluate expressions to build this
table — see *Scaling* — and so the all-NaN case is reported instead by the two
bounded checks described there.

### Scaling — nothing may be O(all images)

A single observing session holds a manageable number of images, but the
`image` table as a whole will not: one of the first intended applications of
AutoWISP runs to **millions of rows**. So the governing rule is that expression
evaluation is always **anchored to one (observing session, channel)** and
bounded by that session's size. Nothing in this feature may do work
proportional to the whole collection.

#### Prerequisite: index `image.observing_session_id` — delivered elsewhere

Every query in this feature is scoped by observing session, so that column has
to be indexed. It currently is not: `data_model/image.py:96` declares it as a
plain `ForeignKey` with no `index=True`, and `Image` has no `__table_args__`
covering it. SQLite does not index foreign keys, so
`WHERE observing_session_id = ?` full-scans `image` today — already the shape
of the existing `get_diagnostic_series_data`, which the canonical-image-list
query would inherit.

**This branch does not add the index.** It is the first real entry in the
project-database migration mechanism described in
`project_db_migrations_plan.md`, which lands first, on its own branch. This
work rebases onto it and simply assumes
`Index("image_observing_session", "observing_session_id", "jd")` exists.

If the sequencing ever changes, note that a model-only fix is not enough:
`create_all` skips existing tables wholesale and will not add an index to one,
so existing projects would silently keep the full scan.

#### Query discipline

- **Every query carries an `observing_session_id` filter**, so the planner
  drives from `image` on the new index and probes `image_diagnostics` on its
  `(image_id, channel, diagnostic_id)` unique index. Confirm with
  `EXPLAIN QUERY PLAN`; a plan leading with a scan of `image_diagnostics` is
  the failure mode to catch.
- `get_available_diagnostics()` (`image_diagnostics_views.py:493`) does a
  `GROUP BY` over the join of `diagnostic_type` and *all* of
  `image_diagnostics` just to learn which names are in use. No index fixes
  this — the query shape is the problem. A per-type `EXISTS` probe is ~25
  index seeks on the existing `(diagnostic_id, value)` index instead of
  walking every row. Small, and directly in code this feature touches.

#### Not fixable by indexing

These are architectural and must be respected regardless of schema:

- **Never evaluate an expression to populate the series table.** That table has
  a row per (session, channel), so evaluating per row is Python work
  proportional to the entire image collection. Availability and counts are SQL
  aggregates, full stop; evaluation happens only for series the user actually
  selected to plot.
- **The all-NaN check must be bounded**, since the finite count is no longer
  available for free. Two cheap places give the same protection: at plot time,
  when a selected series masks to empty (free — the mask already ran); and on
  the management page, spot-checking a *single* recent session.

Two pre-existing limits are worth naming but are **not** in scope: the
cross-session series table grows a row per (session, channel) with no
pagination, and `get_available_diagnostic_series()` aggregates across every
session for the chosen diagnostic — inherently proportional to the number of
images carrying that diagnostic, so no index helps. The requirement on this
work is that the expression path is no more expensive than the plain-diagnostic
path it sits beside, not that it fixes those.

### Composition — expressions referencing other expressions

An expression may reference another expression by name. These are resolved by
**evaluating in dependency order**, not by rewriting text: each expression is
computed once and its result array assigned back into the same `Evaluator`'s
symtable, so a reference to it is an ordinary variable lookup by the time the
dependent expression runs. Given

```
rel_astrom_residual        = astrom_residual / diagonal_fov
rel_astrom_residual_scaled = rel_astrom_residual / median(rel_astrom_residual)
```

`rel_astrom_residual` is evaluated once and both of its uses in the second
expression read the same array.

The order comes from rounds: expressions depending only on real diagnostics are
round 0, those depending only on diagnostics and round 0 are round 1, and so
on. This is Kahn's algorithm, and it carries three properties worth naming:

- **No recomputation.** A subexpression used twice — or shared by several
  dependents in a diamond — is computed once. Textual substitution would
  re-expand it at every use, which is quadratic-to-exponential in depth in the
  bad cases. At the array sizes here (hundreds to thousands of images per
  session/channel) this is not what makes or breaks the feature, but it is
  free to get right.
- **Cycle detection falls out.** If a round comes up empty while expressions
  remain unassigned, those remaining are exactly the ones in or downstream of
  a cycle — reported by name, globally, in one pass. No bespoke recursion
  guard, and it catches a cycle introduced from either end.
- **Intermediates need no alignment bookkeeping**, because every array is
  already on the canonical image list (see *Alignment*). An intermediate that
  is undefined for some images is simply NaN there, and that propagates to its
  dependents by itself.
- **Nothing is rewritten.** No `ast.unparse` roundtrip, no re-parenthesising,
  no stored-text-vs-resolved-text distinction to keep straight anywhere in the
  UI, export, or error messages. What the user typed is what gets evaluated.

The additions this requires, beyond what is already planned:

- `order_expressions(targets, expressions, known_names)` → `(evaluation_order,
  needed_diagnostics)`. The round-labelling above, restricted to the
  dependency subtree of *targets* so plotting one expression does not evaluate
  the whole library. Raises on a cycle or an unresolvable name.
- `validate_expression()` additionally accepts other expression names as valid
  variables, and reports a cycle as a validation error.
- Deleting an expression that others reference is **blocked**, with the error
  naming the dependents. Cheapest safe default, and easy to relax later.
- `import_expressions` validates the **whole incoming set together** after
  staging it, rather than per entry, so a file whose expressions reference each
  other imports regardless of the order they appear in.

Everything else — the model, the queries, the URL space, the plotting path, the
management page — is untouched by nesting.

### Safety

`ast.parse(..., mode="eval")` is the guard: it accepts a single expression and
structurally rejects statements, assignments, loops, and imports. This matters
because imported expressions come from *other people's files* — unlike the
already-existing free-text expression boxes in `tune_starfind_views.py`,
`lightcurve_views.py`, and `detrending_diagnostics_views.py`, which only ever
see what the local user typed. Applied in `validate_expression()`, shared by
the edit form and the importer.

## Implementation

### 1. Model + migration

`autowisp/browser_interface/diagnostics/models.py` (currently the empty stub) —
this app has no Django models yet, but the BUI DB is the Django `default`
database, so a model here lands in `bui_db.sqlite3` alongside `home.Project`:

```python
class DiagnosticExpression(models.Model):
    """A user-defined expression over per-image diagnostics."""

    name = models.SlugField(
        max_length=100, unique=True,
        help_text="Name shown in the diagnostics selectors",
    )
    expression = models.TextField(
        help_text="Python expression over per-image diagnostic names",
    )
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
```

Generate the migration once and check it in — the repo has **no**
`makemigrations` call anywhere; `start.py:migrate_and_serve()` only runs
`migrate` at startup, so hand-checked-in migrations are the convention (see
`home/migrations/0001_initial.py`):

```bash
DJANGO_SETTINGS_MODULE=django_project.settings \
  python autowisp/browser_interface/manage.py makemigrations diagnostics
```

Optionally register the model in `diagnostics/admin.py`.

### 2. `diagnostics/expression_data.py` (new)

The project-DB-facing half. No Django views here.

- `get_expressions()` → `{name: expression}` from the Django ORM.
- `get_expression_names(expression)` → the bare `ast.Name` ids referenced by
  one expression, via the `ast` walk above; the caller splits them into
  diagnostics, other expressions, and asteval builtins.

  Keep this a **pure function of the expression text, derived on demand** — do
  not persist the variable list alongside the expression. Parsing happens at
  save time anyway (it is the `mode="eval"` security guard), so caching would
  not remove this code, only add a column or dependency table plus
  invalidation on every write path, with silent-failure modes if it drifts.
  More importantly the useful half cannot be cached: whether a name is a real
  diagnostic depends on which project is open, and whether it resolves to
  another expression depends on what exists at the time — an import batch may
  legitimately contain forward references. Expressions number in the tens and
  parsing is microseconds; if that ever changes, `functools.lru_cache` on this
  function is keyed by the text and so needs no invalidation.
- `order_expressions(targets, expressions, known_names)` →
  `(evaluation_order, needed_diagnostics)`, the round-labelling described under
  *Composition*, restricted to the dependency subtree of *targets*. Raises on a
  cycle (naming the expressions involved) or an unresolvable name.
- `validate_expression(name, expression, db_session)` → raises
  `django.core.exceptions.ValidationError` on: non-slug name, reserved/colliding
  name, non-`mode="eval"` parse, a reference cycle, or names that are neither a
  known diagnostic, nor another expression, nor present in a fresh
  `Evaluator().symtable`. Shared by form and import.
- `get_expression_dependents(name)` → the expressions that reference *name*,
  for the delete guard.
- `get_canonical_images(session_id, db_session)` → `(image_ids, jd_values)`
  ordered by `jd`: the canonical list every array is padded to. No `channel`
  argument — one list per session, shared by every channel and by both the
  plain-diagnostic and expression paths.
- `get_expression_availability(name, db_session)` →
  `[(session_label, session_id, channel, count), …]`, the count being finite
  values.
- `get_expression_series_data(session_id, channel, name, db_session)` →
  `(jd_values, values, image_ids)`, the same 3-tuple contract as the existing
  `get_diagnostic_series_data`, NaN-padded to the canonical list and *not*
  masked — the single mask lives in `plot_image_diagnostic_series`.

### 3. `diagnostics/image_diagnostics_views.py` (modify)

- Factor the duplicated series-dict construction (identical ~18 lines in
  `get_available_diagnostic_series:111-130` and
  `get_available_series_for_pair:124-142`) into one
  `make_series(session_label, session_id, channel, count, quantile_name=None)`
  helper and use it from all three call sites.
- `get_available_diagnostics()` (line 493): append expression names, but only
  those whose referenced diagnostics all exist in this project's
  `DiagnosticType` rows.
- `get_available_diagnostic_series()` (line 34): if the name is an expression,
  build the list from `get_expression_availability()` instead of the
  type-based query.
- `get_diagnostic_series_data()` (line 143): if the name is an expression,
  delegate to `get_expression_series_data()`.

`create_image_diagnostics_figure`, `plot_image_diagnostic_series`,
`update_plot_view`, `download_plot_view`, and the whole clickable-point
`set_urls()` path (line 231) need **no changes** — points on an expression plot
still link to `preview_calibrated_image`, since they are still per-image.

### 4. `diagnostics/diag_vs_diag_views.py` (modify)

Replace `_get_series_query()` (lines 30–92, two `aliased(ImageDiagnostics)` +
two `aliased(DiagnosticType)` self-joins) and the parallel self-join in
`get_xy_series_data()` (lines 186–211) with a uniform composition. Because both
axes come back on the same canonical image list, there is **no join left to
write**:

- Availability = the union of the two axes' series keys
  `(session_id, channel[, quantile_name])`, with the count being the number of
  images where both are finite.
- Data = call the single-quantity getter for each axis and pair them
  positionally; `isfinite(x) & isfinite(y)` at the plotting boundary is what
  used to be the inner join.

This gives all four combinations (type×type, type×expr, expr×type, expr×expr)
from one code path and deletes ~90 lines of aliasing. **Regression risk to
verify by hand:** the `quantiles` pseudo-name expands to one series per
`pixel_q*` per (session, channel), and the current code special-cases which
axis's quantile name lands in the series id. Preserve that by keying on the
quantile name whenever either axis is `quantiles`, and check `quantiles` vs a
plain diagnostic still renders one row per quantile.

### 5. `diagnostics/expression_views.py` (new)

- `list_expressions` — table of expressions with a per-row status computed
  against the open project (OK / missing variables / shadowed by a real
  diagnostic / **inputs present but every value NaN**, the visible symptom of
  using `median` where `nanmedian` is meant), plus the create-or-edit form.
  The all-NaN check evaluates against **one** recent observing session, never
  all of them. Show each row's direct dependencies so a composed expression is
  traceable.
- `save_expression` — POST, `validate_expression()` then create/update;
  `django.contrib.messages.error` on failure, redirect back.
- `delete_expressions` — POST with checked ids (mirror
  `home/views.py:delete_projects`), refusing any id that other expressions
  reference and naming those dependents in the error.
- `export_expressions` — JSON download, following the exact pattern of
  `home/views.py:509 export_master_config`:
  `json.dump` into a `StringIO`, `HttpResponse` with
  `Content-Disposition: attachment; filename="diagnostic_expressions.json"`.
  A selected-subset export must pull in the expressions its selection depends
  on, or the file will not import cleanly elsewhere.
- `import_expressions` — POST, `json.load(request.FILES["expressions-import"])`,
  following `configuration/views.py:511 import_survey_info`. Stages the whole
  incoming set, validates it as a unit against the staged set plus what is
  already stored (so intra-file references resolve regardless of order),
  reports per-entry failures via `messages`, and upserts by name with an
  explicit overwrite-vs-skip choice for names that already exist.

Export format (versioned so it can evolve):

```json
{
  "autowisp_diagnostic_expressions": 1,
  "expressions": [
    {"name": "rel_astrom_residual",
     "expression": "astrom_residual / diagonal_fov",
     "description": "Astrometry residual as a fraction of the field of view"}
  ]
}
```

### 6. Templates, JS, URLs

- New `diagnostics/templates/diagnostics/diagnostic_expressions.html`
  extending `core/lcars_app.html`. Use the hidden-file-input +
  `onchange="form.submit()"` import idiom from
  `configuration/templates/configuration/edit_survey.html:13-40`, and a plain
  `<a>` for export. Note: do **not** wrap full-width status banners in
  `lcars-element`/`lcars-u-*` — those are fixed-width classes.
- New `diagnostics/static/diagnostics/js/diagnostic.expressions.js` for the
  edit/delete row interactions.
- `diagnostics_app.html`: wrap the built-in and expression options in two
  `<optgroup>`s in both `<select>`s (lines 20–40), and add a "Diagnostic
  Expressions" button to the `left_menu` block (line 45).
  `diagnostics_app.js` needs **no** change — dropdown values stay plain names.
- `diagnostics/urls.py`: add `expressions`, `expressions/save`,
  `expressions/delete`, `expressions/export`, `expressions/import`.
- `diagnostics/views.py`: re-export the new views through the hub, as it
  already does for the other modules.

### 7. meson.build (mandatory — sources are listed explicitly, no globs)

- `diagnostics/meson.build`: add `expression_data.py`, `expression_views.py`.
- `diagnostics/migrations/meson.build`: add `0001_initial.py`.
- `diagnostics/templates/diagnostics/meson.build`: add
  `diagnostic_expressions.html`.
- `diagnostics/static/diagnostics/js/meson.build`: add
  `diagnostic.expressions.js`.

### 8. Documentation

`documentation/source/diagnostics.rst` is a complete narrative of the
diagnostics UI and is rendered into `docs/`. Add a section covering: defining
an expression, which variables are available, the vectorized/aggregate
semantics, **why aggregates want the `nan*` forms**, building expressions out
of other expressions, project-availability, and import/export.

## Verification

1. **Unit tests** — add to `diagnostics/tests.py` (currently a stub) a Django
   `TestCase` covering `validate_expression()` accept/reject cases (bad slug,
   reserved name, statement instead of expression, unknown variable, reference
   cycle), `get_expression_names()`, `order_expressions()` — a multi-level
   chain, a diamond where one expression feeds two dependents, a cycle, and
   the restriction to the target's subtree — the delete-with-dependents
   refusal, and an export→import round trip of a composed set. Also cover
   NaN padding and masking on synthetic arrays: a diagnostic missing for some
   images lands at the right indices, NaN propagates through a composed
   expression, and the finite mask keeps `image_ids` aligned with the plotted
   values. These need only the Django DB, no project DB.
2. **Pipeline tests** unaffected, but run them since `image_diagnostics_views`
   sits next to shared code:
   `python -m autowisp.tests failed_test -v`
3. **Manual, end to end** — the BUI must run from the *installed* package, and
   not editable for BUI work:
   ```bash
   pip install .
   wisp-bui            # runs manage.py migrate, then runserver
   ```
   Hard-refresh the browser (cached JS/CSS), then, in a project that has been
   through `find_stars` / `solve_astrometry` / `fit_star_shape`:
   - Define `rel_astrom_residual = astrom_residual / diagonal_fov`; confirm it
     appears in both selectors and plots against time with the right
     per-series counts.
   - Define `rel_bg = bg_center - nanmedian(bg_center)` to confirm aggregate
     (vectorized) functions work, and that the median is taken per session
     rather than across sessions. Then define the same thing with plain
     `median` in a session where some image lacks `bg_center`, and confirm both
     the plot-time message and the management page report it as all-NaN rather
     than silently plotting nothing.
   - Confirm a diagnostic recorded for only some images of a session plots at
     the right positions in time (NaN padding must not shift points), and that
     clicking a point still opens the *correct* frame — the masking-vs-`set_urls`
     alignment is the thing most likely to break silently.
   - Confirm that selecting a series with no data for the chosen quantity is
     skipped rather than claiming a blank subplot (the
     `any(isfinite(...))` guard).
   - The x-axis offset and the subplot split/merge may legitimately differ from
     the previous release, since both now derive from the full session span.
     Check they are still *sensible*, not that they are unchanged.
   - Define `rel_astrom_residual_scaled = rel_astrom_residual /
     median(rel_astrom_residual)` and confirm the composed expression plots,
     that the management page lists its dependency, and that deleting
     `rel_astrom_residual` is refused while it is referenced.
   - Try to save a cycle (`a = b + 1`, then edit `b` to `a + 1`) and confirm it
     is rejected with a clear message.
   - Plot `rel_astrom_residual` vs `bg_center`, `bg_center` vs
     `rel_astrom_residual`, and `rel_astrom_residual` vs `rel_bg`; click a
     point and confirm it still opens the calibrated-frame preview.
   - Regression check: `quantiles` vs a plain diagnostic still yields one table
     row per `pixel_q*`, and plain-vs-plain is unchanged.
   - "Download Figure" produces a PDF for an expression plot.
   - Export, delete the expressions, re-import the file, confirm they return
     with composition intact; re-import over an existing name and confirm the
     overwrite/skip choice.
   - Define an expression using a diagnostic the project lacks and confirm it
     is hidden from the selectors but listed (flagged) on the management page,
     and that an expression composed from it is hidden too.
   - Confirm expressions persist across switching projects (they are global).
4. **Scaling** — the constraint in *Scaling* has to be checked, not assumed,
   and a small test project will not reveal a violation:
   - Run `EXPLAIN QUERY PLAN` on the canonical-image-list query, the
     availability subquery, and the per-series data query. Each must drive from
     `image` on `observing_session_id` and probe `image_diagnostics` by index;
     any plan that leads with a scan of `image_diagnostics` is a failure.
   - Confirm the `image_observing_session` index (from the migrations branch)
     is actually used by these queries, in a project database created before
     it existed and then migrated.
   - Synthesise a large `image` / `image_diagnostics` set (a few hundred
     thousand rows across many sessions is enough to show the shape) and
     confirm that opening the series table and the expression management page
     stay responsive, and that their cost does not grow with the number of
     sessions beyond the row count of the table itself.
   - Confirm no code path evaluates an expression outside a session the user
     selected for plotting, or the one session sampled for the all-NaN check.
5. **Lint/format**: `pylint autowisp/browser_interface/diagnostics/` and Black
   at 80 columns; keep any incidental reformatting as its own commit.

## Out of scope (noted for later)

- Expressions over `PhotometryDiagnostics` (recorded by `fit_magnitudes` but
  surfaced nowhere in the BUI) and over the file-based detrending statistics.
