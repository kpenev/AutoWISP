# Diagnostic Expressions in the BUI

## Context

The BUI diagnostics app lets users plot per-image diagnostics (`bg_center`,
`s_center`, `astrom_residual`, `pointing_offset`, `pixel_q999`, …) against time
or against each other. Those quantities are fixed: whatever a processing step
recorded into `ImageDiagnostics` is exactly what can be plotted. There is no way
to look at a *derived* quantity — a normalised residual
`astrom_residual / diagonal_fov`, a night-relative background
`bg_center - nanmedian(bg_center)`, or a contrast ratio between pixel
quantiles — without leaving the BUI and writing a script.

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

Expressions live in the **same flat name space** as `DiagnosticType` names —
and, after the merge in §4, as `jd` too. This is what keeps the change small:
`diagnostics_app.html`'s two `<select>`s and the single surviving URL pattern
`image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>` treat all three kinds
alike — an expression is just another name in `available_diagnostics`.

Consequences to handle explicitly:

- Names are restricted to Django's slug charset (`SlugField`) so they survive
  the URL converters.
- Saving/importing rejects a name that collides with a built-in diagnostic. The
  reserved set is the static list seeded by
  `_init_diagnostic_types()` in `autowisp/database/initialize_database.py:448`,
  plus `jd`, anything matching `pixel_q*`, and the family pseudo-name — no
  project input, per §1a. **Done**, as `is_reserved_name()` in
  `diagnostic_types`, which `check_expression` asks.

  The family name is the one entry that is reserved without being
  readable, and it was **renamed `quantiles` → `pixel_quantiles`** while
  being moved down to tier 1: it belongs beside the `pixel_q*` members it
  stands for, and the spelling without digits is precisely what keeps it
  from matching `is_quantile_diagnostic`. It had lived as
  `_quantiles_quantity` in the browser interface, where tier 1 could not
  see it — so nothing stopped an expression being named `quantiles` and
  then being silently swallowed by the fan-out that expands the family
  into one series per member. The rename touches the selector value, the
  one `{% url %}` in `processing/progress.html`, and the tests; a
  bookmark naming the old spelling now reports an unknown quantity, which
  is the same treatment any other unresolvable name gets.
- If a future release ships a `DiagnosticType` matching an existing expression
  name, the **real diagnostic wins** at resolution time and the management page
  flags the expression as shadowed. The no-collision rule carries extra weight
  because expression names are themselves variables inside other expressions
  (see *Composition* below) — a collision would be genuinely ambiguous there,
  not merely confusing in a dropdown.
- A bookmarked URL naming an expression that is unresolvable in the currently
  open project must render the page with an empty series table and a message,
  not a 500. The name survives in the URL after a project switch, so this is
  reachable by ordinary use, not just by hand-editing the address bar.

### Evaluator

Reuse `autowisp.evaluator.Evaluator` (`autowisp/evaluator.py`) — the asteval
wrapper already used for every other user expression in the pipeline. It takes
a dict of `{name: numpy array}` and evaluates **vectorized** over all images at
once, which is what makes `bg_center - nanmedian(bg_center)` work for free
(asteval's default symtable carries most of the numpy/math names).

### NaN-aware aggregates

Because every array is NaN-padded to the canonical image list, the `nan*`
aggregates are the ones users actually want, so they must be **uniformly
available** — the same names, in every evaluator, regardless of environment.
They are not today:

- asteval's default symtable supplies only `nanargmax`, `nanargmin`, `nanmax`,
  `nanmin`, `nansum` (plus the `nan` constant and `nan_to_num`). Verified
  against asteval 0.9.31; the set is an implementation detail of asteval and
  may shift between versions, which is precisely why it cannot be relied on.
- `nanmean`, `nanmedian`, `nanstd`, `nanvar`, `nanpercentile`, `nanquantile`,
  `nanprod`, `nancumsum`, and `nancumprod` are **absent**.
- `LightCurveEvaluator` patches over exactly two of those gaps
  (`nanmean`/`nanmedian`, `autowisp/evaluator.py:109-110`) while `Evaluator`
  patches none — so which aggregates work depends on which evaluator you happen
  to be in.

Fix it once, in a shared base class both evaluators derive from (see §1). Bind
the whole set unconditionally rather than only the currently-missing names —
rebinding `numpy.nanmax` over asteval's own `numpy.nanmax` is a no-op, and it
makes the available names a property of AutoWISP rather than of the installed
asteval.

List the names explicitly rather than deriving them from `dir(numpy)`: a
derived list would vary with the numpy version, which is the unpredictability
being removed. All 14 (`nanmean`, `nanmedian`, `nanstd`, `nanvar`, `nansum`,
`nanprod`, `nanmin`, `nanmax`, `nanpercentile`, `nanquantile`, `nanargmin`,
`nanargmax`, `nancumsum`, `nancumprod`) predate the numpy 1.21 floor and
survive numpy 2, so an explicit list fails loudly if that ever stops being
true.

This is a small, self-contained improvement to shared pipeline code that
happens to be a prerequisite here; it lands as its own commit.

### Alignment — one canonical image list per session and image type

Every array is built against the **same canonical image list**, with `NaN`
wherever a value does not exist. Alignment is then structural: index *i* is the
same image in every array, for every diagnostic and every expression, with no
join anywhere.

The canonical list is simply:

```sql
SELECT id, jd FROM image
WHERE observing_session_id = ? AND image_type_id = ? AND jd IS NOT NULL
ORDER BY jd
```

No requirement that any diagnostic be present and no join to
`image_diagnostics`, so the list is a function of the observing session and
the image type alone. Images with nothing recorded for a given channel are
just NaN across the board there.

**Type splits the list; channel does not**, and the asymmetry is not
arbitrary. Channels are simultaneous measurements of one exposure, so they
share an index space by construction — a row exists for the image either
way, and requiring "at least one diagnostic in this channel" would buy
nothing, since those rows are masked out before anything reads them. Image
types are *different images*, which were never meaningfully on a common
index.

That distinction is load-bearing rather than tidy, because **the canonical
list is what an aggregate spans**. `calibrate` runs on dark, flat and object
frames (`database/defaults.py:10-12`), and each produces the `pixel_q*`
diagnostics (`image_calibration/calibrator.py:820`, handed to `mark_end` at
`processing_steps/calibrate.py:342`). So wherever a user's
`observing_session_label` puts calibration frames in the same session as
objects — which `get_or_create_target` permits, tolerating `zero`/`dark`/
`flat` with null pointing — a session-wide list would make
`nanmedian(pixel_q999)` a median *across image types*. Silently, and
certainly not what was meant.

Splitting also costs almost nothing in the series table, because rows with
no data are dropped already: it separates rows only where a diagnostic
genuinely spans types, which is exactly where they should be separate. An
object-only diagnostic such as `bg_center` yields the same rows as before.

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
`bg_center`. The `nan*` forms are the answer, and they must be **uniformly
available** — see *NaN-aware aggregates* below. Do **not** silently rebind
`median` to `nanmedian`: an expression shared through an export file has to
mean what it says.

This bites more often than it first appears, because the canonical list has no
image-type filter and no processing-progress filter. `bg_center` is written by
`fit_star_shape` (step 5 of the pipeline), so a session that is partially
processed, that contains a failed or skipped image, or whose
`observing_session_label` groups calibration frames together with object frames
(`get_or_create_observing_session` in `database/provenance_resolver.py` keys the
session purely on that header expression, and `get_or_create_target` explicitly
tolerates `zero`/`dark`/`flat` with null pointing) will have at least one image
lacking the diagnostic — which is all it takes.

It is nonetheless treated as **ordinary user error**, in exactly the way
`bg_center / 0` is, and gets the same treatment: an empty plot. Determining
whether an expression *actually* evaluates to all-NaN requires per-session
evaluation, and that machinery is not worth maintaining (see *Series table
semantics* and *Scaling*). The one concession is a static, save-time warning
described under *Bare-aggregate warning*.

Note also that these aggregates are **per (session, image type, channel)**,
not global:
`nanmedian(bg_center)` is the median over the plotted session, not over the
whole archive. That is the useful meaning for night-relative quantities, and it
is the only one that stays affordable at scale. Say so in the documentation.

### Evaluation — per-series algorithm

Per (observing session, image type, channel) series:

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

### Series table semantics

The per-series count shown in the table is a **SQL aggregate only** — the
number of images having all the needed diagnostics, from the
`HAVING COUNT(DISTINCT diagnostic_type.id) == len(needed)` subquery. Label the
column as "images with the required inputs" rather than implying a point count.

This is exact for the case that carries meaning: an expression none of whose
inputs are recorded in a session yields count 0 and therefore no row, which is
how "not available here" gets communicated. It is an *upper* bound only when
arithmetic manufactures NaN from finite inputs — a bare aggregate over a padded
array, or a division by zero. Such a series does get a row and then plots
empty. That is **accepted**: the alternative is evaluating the expression for
every (session, channel) in the table, which is precisely the work forbidden
under *Scaling*.

### Bare-aggregate warning

The one concession to the `median`-vs-`nanmedian` trap is static and costs
nothing. At **save time only**, walk the already-parsed AST for calls to
`median`, `mean`, `std`, `min`, `max`, `sum`, or `percentile` without a `nan`
prefix, and emit a **non-blocking** `django.contrib.messages.warning`
suggesting the `nan*` form.

No DB query, no session or channel to choose, no evaluation, and no false
alarms from partial processing — it reports what the expression says, not what
some sampled session happens to contain. A deliberate bare `median` still
saves. The AST is already being parsed at this point as the security guard, so
this adds a walk and a name list, nothing more.

### Scaling — nothing may be O(all images)

A single observing session holds a manageable number of images, but the
`image` table as a whole will not: one of the first intended applications of
AutoWISP runs to **millions of rows**. So the governing rule is that expression
evaluation is always **anchored to one (observing session, image type,
channel)** and
bounded by that session's size. Nothing in this feature may do work
proportional to the whole collection.

#### Prerequisite: index `image.observing_session_id` — already delivered

Every query in this feature is scoped by observing session, so that column has
to be indexed. **This has landed and is no longer a dependency.**
`Index("image_observing_session", "observing_session_id", "jd")` is declared in
`Image.__table_args__` (`autowisp/database/data_model/image.py`), applied to
existing project databases by
`autowisp/database/migrations/versions/0002_image_session_index.py`, and
covered by `autowisp/tests/test_database_migration.py`.

The reason it needed a migration rather than a model edit is worth keeping on
record: SQLite does not index foreign keys, and `create_all` skips existing
tables wholesale and will not add an index to one, so a model-only fix would
have left every existing project silently full-scanning `image`.

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
  a row per (session, image type, channel), so evaluating per row is Python
  work
  proportional to the entire image collection. Availability and counts are SQL
  aggregates, full stop; evaluation happens only for series the user actually
  selected to plot.
- **There is no all-NaN check anywhere**, and deliberately so. Every way of
  getting one either violates the rule above or is a heuristic that lies: a
  management page that spot-checks one recent session reports on the session
  *most* likely to be mid-processing, so it would raise false alarms about
  expressions that are fine everywhere else, and it would still miss a
  per-channel failure in the channel it did not sample. The static save-time
  warning covers the realistic mistake at the moment it is made, and an empty
  plot covers the rest.

Two pre-existing limits are worth naming but are **not** in scope: the
cross-session series table grows a row per (session, image type, channel)
with no
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

The order comes from a depth-first walk from the targets that appends each
expression only once the expressions it references have been appended.
Appending on the way *out* is what makes it an order rather than a
traversal: on the way in, two expressions reached at the same depth are
indistinguishable even when one references the other, so reversing the order
of discovery is **not** a valid evaluation order. With `a = b + c` and
`c = b * 2`, both `b` and `c` are reached from `a` together, and reversing
can place `c` before the `b` it needs — the more so because the references
come back as a set, making discovery order non-deterministic.

Four properties worth naming:

- **No recomputation.** A subexpression used twice — or shared by several
  dependents in a diamond — is computed once. Textual substitution would
  re-expand it at every use, which is quadratic-to-exponential in depth in the
  bad cases. At the array sizes here (hundreds to thousands of images per
  session/channel) this is not what makes or breaks the feature, but it is
  free to get right.
- **Cycle detection falls out**, and names the loop precisely. Reaching an
  expression already on the current path closes a cycle, and that path *is*
  the loop, so the message reads `a -> b -> a` rather than listing every
  expression that happens to be stuck. An expression merely downstream of a
  cycle is not implicated, which matters because it is the innocent one.
  Catches a cycle introduced from either end, and self-reference with no
  special case.
- **Intermediates need no alignment bookkeeping**, because every array is
  already on the canonical image list (see *Alignment*). An intermediate that
  is undefined for some images is simply NaN there, and that propagates to its
  dependents by itself.
- **Nothing is rewritten.** No `ast.unparse` roundtrip, no re-parenthesising,
  no stored-text-vs-resolved-text distinction to keep straight anywhere in the
  UI, export, or error messages. What the user typed is what gets evaluated.

The additions this requires, beyond what is already planned:

- `order_expressions(targets, expressions)` → `(evaluation_order,
  needed_diagnostics)`. The depth-first post-order above, restricted to the
  dependency subtree of *targets* so plotting one expression does not evaluate
  the whole library. Raises on a cycle or an unresolvable name. (What a name
  may mean comes from `diagnostic_types.is_known_quantity()`, so no project
  input is needed — see §1a.)
- `check_expression()` additionally accepts other expression names as valid
  variables, and reports a cycle among its problems.
- Deleting an expression that others reference is **blocked**, with the error
  naming the dependents. Cheapest safe default, and easy to relax later.
- `import_expressions` validates the **whole incoming set together** after
  staging it, rather than per entry, so a file whose expressions reference each
  other imports regardless of the order they appear in.

Everything else — the model, the queries, the URL space, the plotting path, the
management page — is untouched by nesting.

### Safety

Three layers, and it is worth being clear which one does what, because the
obvious answer is wrong.

**`ast.parse(..., mode="eval")` is not the security boundary.** It rejects
statements, assignments, loops and imports, which is worth having and is
free — parsing is how the names get read anyway. But every interesting
attack is a valid *expression*: `__import__('os').listdir('.')`,
`().__class__.__bases__[0].__subclasses__()` and `open(f).read()` all parse
without complaint.

**asteval is the sandbox**, and it holds: `__import__`, `eval`, `exec` and
`getattr` are absent from its symbol table, dunder traversal raises, and its
`open` wrapper refuses every file mode but reading. Verified against asteval
0.9.31.

**Reading was still enough to matter**, which is why `EvaluatorBase` now
drops `open` and `print` (`removed_names`). Asteval leaves `open` available,
so an expression could read any file the user can — harmless in a box the
local user typed into, but expressions now travel between installations in
export files, and a shared "expression" that reads `~/.ssh/` is not one.
`print` goes with it as a side effect rather than a value. Removing them in
the evaluator rather than in the validator protects the ~26 modules that
build an `Evaluator`, not only this feature, and means `check_expression`
needs no denylist of its own: it admits the symbol table as it finds it.

`check_expression` closes the loop by restricting every referenced name to
that symbol table, the project's diagnostics, or another expression — so the
mathematical subset is the whole of what an expression can reach.

## Implementation

### 1. `autowisp/evaluator.py` — uniform NaN-aware aggregates (modify)

> **Done.** `EvaluatorBase` carries all 14 NaN-ignoring aggregates and both
> evaluators derive from it. `__call__` was hoisted onto the base as well,
> so `LightCurveEvaluator` raises rather than returning `None`; the four
> call sites in `get_from_lc.py` that had opted into `raise_errors=True` by
> hand are correspondingly simpler. Two dead-code fixes came with it: an
> assertion against an undefined `times`, and `old_main()`.

Landed first, as its own commit; it is shared pipeline code and stands on
its own merits independently of this feature.

`Evaluator` and `LightCurveEvaluator` are **siblings**, both deriving straight
from `asteval.Interpreter` — neither inherits from the other. That is why the
`nanmean`/`nanmedian` lines had to be repeated in `LightCurveEvaluator` in the
first place, and it is why a shared base class is the right shape here rather
than an extra module-level helper: it is the only place the two can meet.

Add `EvaluatorBase(asteval.Interpreter)` carrying the `nan_aggregates` tuple as
a class attribute and binding it in `__init__`, then derive **both** evaluators
from it. Neither subclass needs an explicit call — both already open their
`__init__` with `super().__init__()`, so the binding happens there, which is
also exactly the right moment:

- In `Evaluator`, before the `for data_entry in data` loop, so a data column
  named like an aggregate shadows the function rather than the reverse. Later
  assignment wins in a symtable, and the data is what the user is asking about.
- In `LightCurveEvaluator`, before the closing `self._reset()` that binds the
  dataset names, giving datasets the same precedence.

Hoist `__call__` (with its `raise_errors=True` default) onto the base as well,
so both evaluators raise instead of returning `None`. Asteval's default prints
the error and returns `None`, which does not buy tolerance — it relocates the
crash away from the expression that caused it, after the message naming that
expression has already scrolled past. Checked against the eight call sites in
`autowisp/diagnostics/get_from_lc.py`:

- Four (`:80`, `:145`, `:178`, `:185`) already passed `raise_errors=True` by
  hand. Having opted in four separate times is itself a statement about which
  behaviour was wanted. Now that it is the default these become redundant and
  are **removed**, which is most of what makes this a net simplification
  rather than an addition.
- The other four (`:25`, `:29`, `:34`, `:76`) relied on the default and never
  checked the result: the `None` lands in `transit_model`'s arguments, in
  `len(None)`, and in `None - ndarray`. All three fail later and less legibly
  than raising would.

`get_from_lc.py` is the only consumer of `LightCurveEvaluator`, so the blast
radius is one module. A caller wanting the old behaviour can still pass
`raise_errors=False` explicitly.

Two dead-code findings in the same module were cleaned up while confirming the
above, since they were what made the call-site audit hard to read:

- `get_from_lc.py:35` asserted against an undefined `times` (only ever a
  parameter of `transit_model`), so the `shift_to` branch raised `NameError`
  whenever it ran. Corrected to `len(model_values)`, which is what the
  following subtraction actually requires.
- `old_main()` was obsolete scratch code — never called from anywhere, a
  hardcoded personal data path, a docstring naming one target while saving
  files named for another, most of the body commented out, and two undefined
  names (`combined_figure_id`, `individual_figures_id`) whose definitions were
  themselves commented out. Deleted, along with the `pyplot` and
  `DataReductionFile` imports that existed only to serve it. The console
  script `wisp-get-from-lc` points at `main`, and the module's only external
  importer takes `get_plot_data`/`calculate_combined`, so nothing referenced
  it. Takes the module from pylint 8.95/10 to 10.00/10.

Delete the now-redundant `self.symtable["nanmean"]` / `["nanmedian"]` lines
from `LightCurveEvaluator.__init__`.

Note `Evaluator.__init__` recurses through `self.__init__(...)` for the FITS
and HDF5 branches, so the binding runs more than once on those paths. Harmless,
since it is idempotent.

### 1a. `autowisp/diagnostics/diagnostic_types.py` — the vocabulary (new)

> **Done.** The catalogue moved, and the design got simpler than this
> section first proposed: rather than tier 1 taking *known_names* and
> *known_patterns* as arguments, `diagnostic_types` answers the question
> itself through `is_known_quantity()`. See *One predicate, not two
> parameters* below for why that turned out to be sound rather than merely
> shorter.
>
> Three things beyond what was planned came with it. `time_quantity` moved
> here too, since tier 1 needs `jd` and cannot import tier 2 — which also
> ends the duplicate definition §3 noted. `image_processing.py` now asks
> `is_quantile_diagnostic()` instead of testing `startswith("pixel_q")`,
> so the code creating quantile rows and the code validating against them
> share one definition. And `get_known_names()` was deleted from tier 2,
> having no callers left.
>
> One defect was fixed in passing: the continuation line of the
> `*_map_residual` description was not an f-string, so every project ever
> created seeded them reading "and smoothed `{param.upper()}` map"
> literally. `TestCatalogue.test_descriptions_are_fully_interpolated`
> guards it, since a missing `f` prefix is otherwise silent.

Numbered `1a` rather than renumbering §2–§9, which are cross-referenced
throughout. Like §1 it is shared pipeline code that happens to be a
prerequisite here, and lands as its own commit.

**The problem it removes.** Validation currently needs an open project,
because `check_expression` learns what a name may mean only from
`get_known_names(db_session)`. That is a false dependency: the diagnostic
catalogue is a *static list* seeded by `_init_diagnostic_types()`
(`autowisp/database/initialize_database.py:448`), identical in every
project. The only genuinely runtime-created names are `pixel_q*`, made by
`calibrate` rather than seeded — and those are a *pattern*, not a list, so
no amount of enumeration would have captured them anyway.

So the vocabulary is knowable without a database, and the database module
should be its consumer rather than its owner.

**Extract the catalogue.** `_init_diagnostic_types()` is a ~90-line literal
of `(name, description)` pairs wrapped around a single
`db_session.add(DiagnosticType(...))`. Move the literal into this module;
the function shrinks to a loop over it and keeps its only real job, which
is writing rows.

**The description travels with the name.** It has to: leaving descriptions
behind in `initialize_database.py` would split the catalogue across two
files that must be edited together, which is the drift the extraction
exists to prevent. Validation ignores them, but the seeder does not.

**A mapping, not a sequence of pairs.** `DiagnosticType.name` is
`unique=True` and `description` is non-null, so the catalogue *is* a
name → description mapping; a list of pairs would silently admit duplicate
names that the table then rejects at insert time.

**Read-only accessors, not module constants.** A module-level dict is
mutable, and a caller that mutates the shared catalogue corrupts it for
every other caller in the process. Expose functions returning immutable
values instead, following `_evaluator_names()` (`expressions.py:28`), which
already does exactly this:

```python
@functools.lru_cache(maxsize=1)
def standard_diagnostic_types():
    """Every diagnostic seeded into a new project, name -> description."""
    return MappingProxyType({...})


@functools.lru_cache(maxsize=1)
def standard_diagnostic_names():
    """Just the names, which is all validation needs."""
    return frozenset(standard_diagnostic_types())
```

`types.MappingProxyType` is what makes the mapping genuinely read-only
rather than read-only by convention. Caching costs nothing and the values
never change within a process. The quantile pattern —
`re.compile(r"pixel_q\d+\Z")` — is a private module constant behind
`is_quantile_diagnostic()` rather than exposed, since every caller wants
the question answered rather than the pattern itself.

**Names stay lower case.** `.pylintrc` sets
`const-rgx=[a-z_][a-z0-9_]{2,30}$`, so the ALL_CAPS spelling would fail
lint; the existing `_image_order` in `expression_series.py:141` is the
house style.

**One predicate, not two parameters.** The first draft of this section had
`check_expression` take *known_names* and a new *known_patterns*, on the
grounds that a project might carry types outside the catalogue. **It
cannot**, and the code says so:

```python
if diag_type_id is None:
    if is_quantile_diagnostic(diag_name):   # was startswith("pixel_q")
        ...create the row...
    else:
        raise PipelineError(f"Unknown diagnostic type {diag_name!r}")
```

That branch in `_save_image_diagnostics` is the only creator of
`DiagnosticType` rows besides the seeder, and it refuses every name that
is not a quantile. So a project's diagnostic types are *provably* a subset
of catalogue ∪ `pixel_q*`, `known_names` was a parameter that could never
carry anything the catalogue does not already imply, and threading it
around was ceremony.

So the vocabulary answers for itself:

```python
is_quantile_diagnostic(name)   # the pattern, shared with the creator
is_diagnostic(name)            # catalogue ∪ quantiles
is_known_quantity(name)        # the above, plus jd -- what may be *read*
is_reserved_name(name)         # the above, plus the family pseudo-name
```

The last two differ by exactly one entry, and deliberately:
`pixel_quantiles` stands for a family rather than for values, so an
expression may not read it — but neither may it take the name. Reading
and shadowing are separate questions and only here do they disagree.

`check_expression(name, expression, expressions)`, `order_expressions(targets,
expressions)` and `_needed_diagnostics` all lose their name arguments.
Tier 1 gains no database dependency, because the new module is pure data —
the property §3 separates the tiers for is preserved.

Two consequences worth recording:

- **A name matching the quantile pattern is now reserved as well as
  resolvable**, closing a hole the old code had: nothing stopped an
  expression being named `pixel_q999` and shadowing a real quantile.
- **`order_expressions` no longer rejects a diagnostic that this project
  has not recorded.** It resolves, and comes back all-NaN from the padding.
  That is what §Namespace already asks for — an unresolvable bookmark
  should render an empty table, not a 500 — and `evaluate_expressions`
  still reports anything for which no values were supplied, with a better
  message than the old "unresolvable".

**`autowisp/diagnostics/meson.build`** gains `diagnostic_types.py` (§7).

### 2. Model + migration

> **Done.** `bui_database_plan.md` is complete, and this model derives from
> the `core.models.BuiModelBase` it introduced, so `created`, `modified` and
> the trigger maintaining `modified` come for free — the trigger machinery
> selects models by the column rather than from a list, so nothing had to be
> told this table exists.
>
> Two additions beyond what was sketched here: `Meta.ordering = ["name"]`,
> since the management page lists them alphabetically, and admin
> registration, as a way to inspect or repair an expression when the
> management page is itself what needs fixing.

The model regardless of ORM: a slug `name` unique across the installation, the
`expression` text, an optional `description`, and the created/modified
timestamps — which come from the shared base rather than being declared here,
so they are not repeated per model.

```python
class DiagnosticExpression(BuiModelBase):
    """A user-defined expression over per-image diagnostics."""

    name = models.SlugField(
        max_length=100, unique=True,
        help_text="Name shown in the diagnostics selectors",
    )
    expression = models.TextField(
        help_text="Python expression over per-image diagnostic names",
    )
    description = models.TextField(blank=True)
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

### 3. The expression layer — three tiers, only one of which knows Django

> **Done**, and rewritten before it was built. The first draft put
> everything in one `browser_interface/diagnostics/expression_data.py` and
> had several functions take a bare *name*, which forced them to look the
> expression up and so hid a browser-interface dependency inside what should
> be pipeline code. `validate_expression` additionally raised
> `django.core.exceptions.ValidationError`, coupling a project-database
> check to Django. Both are fixed by the tiering below.
>
> All three tiers now exist: `diagnostics/expressions.py` (no database),
> `diagnostics/expression_series.py` (project database) and
> `browser_interface/diagnostics/expression_data.py` (the browser-interface
> one). The property they were separated for holds and is tested: an
> expression is defined, resolved, counted, evaluated and plotted, and only
> the last tier imports Django. The library reaches the plotting code as an
> argument from `views.py` and is never looked up beneath it.
>
> Reading a library from a file, for `run_pipeline`, is the one part of
> tier 3 not built — it waits on §5's export format, and is noted under
> *Out of scope* rather than deferred silently.

#### Why this is not only a browser-interface feature

Expressions exist to **identify images to exclude from magnitude fitting,
EPD and TFA** — not merely to draw plots. That makes them pipeline
machinery that the browser interface happens to also edit and visualise,
and it has to hold across AutoWISP's usage modes:

- **`run_pipeline`, project database only.** Must be able to use
  expressions, with the definitions arriving as configuration rather than
  from the browser-interface database.
- **Individual steps.** Deliberately unaffected: the user deals with
  diagnostics themselves.

There is a precedent for the shape. `fit_magnitudes` already takes
`--fit-source-condition`, "an expression involving catalog, reference
and/or photometry variables which evaluates to zero if a source should be
excluded". The image-level equivalent is the same idea one level up, and
should read the same way.

**A limit worth stating rather than working around:** image diagnostics
live *only* in the project database. `fit_star_shape` passes them to
`mark_end`, and `image_processing._save_image_diagnostics` writes the
`ImageDiagnostics` rows; they are not in the data-reduction file structure.
So a mode with no database at all has no recorded diagnostics, and
expressions over them do not apply there. Nothing below tries to fix that:
there is no facility for supplying diagnostic values from outside, and none
is proposed. What has to survive the browser interface is the *definitions*
arriving as configuration instead of from its database — which is tier 3.

#### The tiers

> **Signatures superseded by §1a.** The *known_names* argument described
> throughout this section is gone: `check_expression(name, expression,
> expressions)` and `order_expressions(targets, expressions)` now ask
> `diagnostic_types.is_known_quantity()` instead, and tier 2's
> `get_known_names()` was deleted. The reasoning below still holds --
> tier 1 has no database -- but it reaches that through a pure-data module
> rather than by taking the names from its caller.

**Tier 1 — `autowisp/diagnostics/expressions.py`. Done. No database of any
kind.** Pure functions over an expression library and already-fetched
values. This is where every rule about expressions lives, so it is the only
place any of it has to be tested.

- `get_expression_names(expression)` → the bare `ast.Name` ids referenced
  by one expression; the caller splits them into diagnostics, other
  expressions, and asteval builtins.

  Keep this a **pure function of the expression text, derived on demand** —
  do not persist the variable list alongside the expression. Parsing happens
  at save time anyway (it is the `mode="eval"` security guard), so caching
  would not remove this code, only add a column or dependency table plus
  invalidation on every write path, with silent-failure modes if it drifts.
  More importantly the useful half cannot be cached: whether a name is a
  real diagnostic depends on which project is open, and whether it resolves
  to another expression depends on what exists at the time — an import batch
  may legitimately contain forward references. Expressions number in the
  tens and parsing is microseconds; if that ever changes,
  `functools.lru_cache` on this function is keyed by the text and so needs
  no invalidation.
- `get_bare_aggregates(expression)` → the names of aggregate calls lacking a
  `nan` prefix, for the save-time warning. The same `ast` walk, looking at
  `ast.Call` funcs rather than bare `ast.Name` ids.
- `order_expressions(targets, expressions, known_names)` →
  `(evaluation_order, needed_diagnostics)` by the depth-first post-order
  under *Composition*, restricted to the dependency subtree of *targets* so
  that plotting one expression does not evaluate the whole library. Raises
  on a cycle, naming the loop, or on an unresolvable name.
- `get_expression_dependents(name, expressions)` → those referencing
  *name*, for the delete guard. **Takes the library**, rather than fetching
  it.
- `rename_references(expression, old_name, new_name)` → the text with each
  reference renamed at its exact source offsets, for the rename cascade in
  §5. Added after the fact; see *Renaming* there for why a rename may not
  simply be refused.
- `evaluate_expressions(targets, expressions, values)` → `{name: array}`.
  *values* is `{diagnostic_name: numpy array}` on a common index, which in
  practice tier 2 built from the canonical image list. Passing it in rather
  than fetching it is what keeps this tier free of a database and cheap to
  test, not a facility for anyone to supply values by hand. The single point
  where an `Evaluator` is built and the ordered evaluation is run.
- `check_expression(name, expression, expressions, known_names)` → a list
  of problems, as plain strings: non-slug name, reserved or colliding name,
  a body that is not a single `mode="eval"` expression, a reference cycle,
  or names that are neither a known diagnostic, nor another expression, nor
  in a fresh `Evaluator().symtable`. **Returns problems rather than
  raising**, so the tier stays free of Django; §5 turns them into a
  `ValidationError`.

  The reserved set is the static list seeded by `_init_diagnostic_types()`,
  plus `quantiles`, `jd` and `pixel_q*`, plus the `DiagnosticType` names of
  the open project. That last group is why *known_names* is an argument:
  this tier has no database to query for them, and which names are taken
  depends on which project is open.

**Tier 2 — `autowisp/diagnostics/expression_series.py` (new). Project
database only. Done.** Turns a series into the values tier 1 needs.
`get_canonical_images` and `count_images_with_all` both moved here out of
`image_diagnostics_views`, where §4 had put them, and `SeriesKey` came with
them — every function below wanted three of its four fields, which is the
same argument-order hazard the key was introduced to remove. The browser
interface now imports it, along with `time_quantity`, which had been defined
in both places.

Signatures differ from the draft above in two ways, both settled by the code
they meet: the image type is its **name** rather than its id, since that is
what `SeriesKey` carries and what the queries and tests already key on; and
they take that key rather than its fields spelled out.

- `get_canonical_images(series_key, db_session)` → `(image_ids, jd_values)`,
  the list every array is padded to. The channel is deliberately unused —
  channels share an index space where types do not — as is `quantile_name`,
  which identifies a plotted series rather than a set of images.
- `get_diagnostic_values(series_key, names, db_session)` →
  `({name: array}, image_ids)`, NaN-padded to that list. The ids come back
  because the same query already carries them; the dates do not, since `jd`
  is asked for by name like anything else and arrives in the dictionary.
- `get_series_values(series_key, quantities, expressions, db_session)` →
  `({quantity: array}, image_ids)`, *not* masked; the single mask lives in
  `plot_image_diagnostic_series`. **Takes the library explicitly** — the
  signature the first draft got wrong — and **takes all the quantities
  wanted at once**, which is the point below.
- `get_known_names(db_session)` → every `diagnostic_type` name plus `jd`.
  Not in the draft, but tier 1 takes *known_names* as an argument precisely
  because it cannot look them up, so something had to.
- `get_expression_availability(name, expressions, db_session)` →
  `[(session_label, session_id, image_type, channel, count), …]`, the count
  coming from the SQL aggregate under *Series table semantics* rather than
  from evaluating anything.

**The padding is the database's, not ours.** One query does it: a cross join
pairs every wanted diagnostic with every image of the series, and an outer
join attaches the values, leaving `NULL` where nothing was recorded. The
result is therefore a rectangle — one row per image per name — which the
unique index on `(image_id, channel, diagnostic_id)` guarantees, so the
value column is read out whole and reshaped into one row per name rather
than being matched back up image by image. `EXPLAIN QUERY PLAN` confirms it
still drives from `image` on `image_observing_session` and probes
`image_diagnostics` by that unique index, as *Scaling* requires.

Two details that arrangement depends on, both of them silent when wrong:

- **The image order must be total**, so everything is ordered by `jd` *and*
  `id`. Two frames of a session can share a `jd`, and an order that leaves
  the tie to the database lets two queries return the same images
  differently and pair a value with the wrong image.
- **Each block's name is read from its first row**, never from a sorted copy
  of the names asked for. SQL's ordering of strings is the collation's to
  decide — MySQL's default is case-insensitive where Python's is not — and
  nothing here should depend on the two agreeing.

**Both axes together, not one at a time.** *quantities* is a sequence
because §4's figure path wants x and y for the same series, and resolving
them separately would waste the one property that needs deliberate plumbing
to be real. Tier 1's `evaluate_expressions` already takes a *set* of
targets, so one call gives:

- one query for the **union** of the diagnostics both axes need, rather
  than two overlapping ones;
- one `Evaluator`, so a subexpression the two axes share is computed once —
  which is exactly the *No recomputation* claim under *Composition*. Per
  axis it would be computed twice, and the claim would hold only within an
  axis rather than across the plot.

Note this is the only place that property needs arranging. Everything else
called lazy in this plan is **scoping, not memoization**, and needs no state
whatsoever: the series table never evaluates because its counts are SQL
aggregates, and `order_expressions` walks only the targets' subtree. Nothing
is cached between requests, and nothing needs to be — a session/channel is
hundreds to thousands of images, so evaluation is microseconds against an
indexed query that dominates it. Should that ever stop being true, cache
against the project database's `timestamp` columns rather than inventing a
store; but measure first, because a value depends on the expression text,
every transitive dependency's text, the diagnostic rows for that session and
channel, and which project is open — and a silently stale plot is worse than
a slow one in a tool for deciding which images to discard.

**Tier 3 — where the library comes from. Done for the browser interface.**
Two sources, both producing the same `{name: expression}` dictionary,
neither knowing what it is for:

- `browser_interface/diagnostics/expression_data.py`: **Done.**
  `get_expressions()` from the Django model, returning the whole library
  rather than the part that resolves — filtering needs a project's names,
  which this tier has not got. **The only Django-aware code in the whole
  feature** besides the views.

  Where it is *called* took one correction worth recording. Importing it
  into `image_diagnostics_views` pulls in the Django models, and so the app
  registry, which broke `test_diagnostics_views` — a module that
  deliberately exercises the plotting path against a project database with
  no Django at all. That property is worth more than the convenience, so
  the library is fetched in `views.py`, already the app's Django side, and
  handed to `display_diagnostics`, `create_diagnostics_figure` and
  `get_available_series` as an argument. The two plot routes gained named
  handlers there in place of the `functools.partial` in `urls.py`, which had
  nowhere per-request to fetch anything.
- A JSON or `key=value` file for `run_pipeline`. The §5 export format is
  the obvious vehicle, which is why it is versioned and why a subset export
  pulls in what it depends on — see *Out of scope*.

#### Consequences worth stating

- The browser interface becomes one caller among several rather than the
  owner. Tier 1 and tier 2 have no Django import, so they are testable
  without standing Django up, and usable from `run_pipeline`.
- Actually *excluding* images in magfit/EPD/TFA — an
  `--fit-image-condition` alongside the existing `--fit-source-condition`,
  and the plumbing to honour it — is **not in this plan**. This tiering is
  what makes it a small addition later rather than a rewrite. Noted under
  *Out of scope*.

### 4. Merge the two view modules into one

> **Done.** `diag_vs_diag_views.py` is gone; both axes resolve through one
> path against the canonical image list, with `jd` an ordinary quantity. Net
> deletion despite adding the canonical-list machinery. The characterization
> tests below were written and green against the previous code first, and
> pass unchanged in intent against the merged one.
>
> One thing the work settled that this section left open: `quantiles`
> resolves to its concrete `pixel_q*` member **once per series**, in
> `resolve_quantity`, so `jd` ends up the only quantity needing a branch
> anywhere. The default `aspect_ratio` is also conditional — 3.0 against
> time, 1.0 otherwise — preserving the wide time-series and square scatter
> looks the two modules had separately.
>
> **Deviation — now closed.** The last paragraph below says
> `get_available_diagnostics()` "gains `jd` plus the expression names whose
> referenced diagnostics all exist". The `EXISTS` rework and `jd` landed
> with the merge; the expression names have landed since, as **three**
> functions rather than one, because §6 wants the two kinds in separate
> `<optgroup>`s and a single list would have to be split again to render:
> `get_recorded_diagnostics` (the database half, the raw in-use names),
> `get_available_diagnostics` (a pure function of those, collapsing the
> quantiles into the family) and `get_available_expressions`.
> `display_diagnostics` probes once and hands the result to both.
>
> **The test it needed was availability, not validity** — this block was
> written in the §1a commit itself and its first draft said otherwise. §1a
> made `check_expression` project-independent, so filtering the dropdown by
> it would filter *nothing*: every stored expression is valid in every
> project, and all of them would be offered everywhere, which is the
> opposite of what §Namespace asks for. Instead take the transitive
> `needed` set `order_expressions([name], expressions)` already returns and
> offer the name iff `needed - {jd}` is a subset of the names this project
> actually records — which the `EXISTS` probes have already computed by
> that point. That filter must consult the **raw** in-use set, before
> `pixel_q*` collapses into `quantiles`, since an expression may reference
> a concrete `pixel_q999`; and it must treat a `PipelineError` as
> unavailable, so a stored cycle cannot 500 the plot page.

`image_diagnostics_views.py` and `diag_vs_diag_views.py` were one feature
wearing two URLs. They already share the template (`diagnostics_app.html`) and
the JavaScript, and the x-selector already lists `time` as **one option among
the diagnostics** — the user-facing model is already "x is a quantity, one of
which happens to be time". The whole split is a single branch in
`navigateDiagnostics()`: `if (xDiag === 'time')` choose URL shape A, else
shape B.

Under the canonical-image-list design the case is stronger still, because
`get_canonical_images()` already returns `(image_ids, jd_values)`. **`jd` is
the one quantity that arrives with every series at zero query cost.** It is the
natural x, not a special case. Making it an ordinary variable also buys
something new: expressions can *use* it — `jd - 2460000`, night-relative time,
phase folds — none of which is possible today.

So: one code path, `x` and `y` both resolved through the same single-quantity
getter, with `jd` joining diagnostics and expressions as a third kind of
resolvable name.

- **`jd`, not `bjd`.** `Image.jd` is plain mid-exposure JD: `JD-OBS` is built
  in `provenance_resolver.py` as `exposure_start.jd + exposure/2` with no
  `location`, and stored verbatim by `add_images_to_db.py`. BJD exists, but
  only in lightcurves (`lc_data_io.py` re-wraps the same `JD-OBS` with an
  observatory location and per-source coordinates). Barycentric correction
  depends on where on the sky you point, so **BJD is per-source while
  diagnostics are per-image** — it is not merely absent here, it is undefined.
  Naming the variable `bjd` would assert a correction that has not been
  applied.
- **`jd` is the only never-NaN quantity**, since the canonical list is defined
  by `jd IS NOT NULL`. It is therefore the sole legitimate exception to the
  "prefer `nan*`" guidance — say so in the docs, since it will otherwise look
  inconsistent.
- Availability = the union of the two axes' series keys
  `(session_id, image_type, channel[, quantile_name])`, count from the SQL
  aggregate. An axis of `jd` constrains nothing, so a `jd` vs *y* table is
  today's time-series table, except split by image type.

  **The image type is new** — see *Alignment* for why it belongs in the key
  rather than only in the canonical list. Practically this means a `Type`
  column in the table, `image_type_id` in the `GROUP BY`, and a join to
  `image_type` for the name. It changes what is displayed only where a
  diagnostic actually spans types, which today means `quantiles`: `pixel_q*`
  is recorded for dark, flat and object frames alike, so a shared session
  currently shows them as one undifferentiated row and one undifferentiated
  series.
- Data = ask for both axes at once and pair positionally;
  `isfinite(x) & isfinite(y)` at the plotting boundary is what used to be the
  inner join. As implemented for plain diagnostics this is
  `get_quantity_values` once per axis, which is free — each is a separate
  column. Once expressions arrive it becomes a single
  `get_series_values(…, quantities=(x, y), …)` call, so that two axes
  sharing a subexpression evaluate it once rather than twice; see §3.

This deletes `_get_series_query()` (lines 30–92, two `aliased(ImageDiagnostics)`
+ two `aliased(DiagnosticType)` self-joins), the parallel self-join in
`get_xy_series_data()` (lines 186–211), one of the two figure factories, the
`navigateDiagnostics()` branch, and the duplicated ~18-line series-dict
construction — which is now deduplicated as a side effect rather than as the
separate `make_series` commit previously planned.

Two behaviours must be carried across deliberately:

- **The shared x-offset stays presentation logic.**
  `create_image_diagnostics_figure` computes `min_jd` across *all* selected
  series (lines 369–379) and applies
  that one value to every one of them, so nights keep their relative spacing
  and the axis reads `JD - 2460…`. Keep exactly that, conditioned on `x ==
  "jd"`. It never touches the data, so it stays out of the expression system.
  Deliberately **not** replaced by a seeded `time = jd - min(jd)` expression:
  expression aggregates are per (session, image type, channel), so that would
  zero each
  night independently, overlay them at a common origin, and — because all
  x-ranges would then overlap — collapse today's per-night subplots into one.
  That is a useful view, but a different one; a user who wants it can define it.
  **No expressions are seeded**; the library starts empty.
- **Subplot grouping generalizes.** `group_series_by_jd_overlap` is really
  "group by x-range overlap". For a non-time x the ranges typically overlap, so
  it collapses to the single axes the xy path uses today — today's behaviour
  falls out of the general rule rather than needing a branch. Assert this in a
  test rather than assuming it.

`plot_image_diagnostic_series`, `update_plot_view`, `download_plot_view`, and
the clickable-point `set_urls()` path need no behavioural change — points are
still per-image, so they still link to `preview_calibrated_image`.

Also in this pass: `get_available_diagnostics()` (line 493) switches its
`GROUP BY` over all of `image_diagnostics` to per-type `EXISTS` probes (see
*Query discipline*), and gains `jd` plus the expression names whose referenced
diagnostics all exist in this project's `DiagnosticType` rows.

**Regression risk — pin it with a test before refactoring, not by eye.** The
`quantiles` pseudo-name expands to one series per `pixel_q*` per
(session, channel), and the current code special-cases which axis's quantile
name lands in the series id (`_get_series_query` lines 76–87). Write a test
asserting one series per `pixel_q*` with the quantile name in the series id,
**for both axis orders**, and get it green against the current code first. With
the merge this is the only remaining path, so the test is load-bearing rather
than precautionary.

**The series id gets a new encoding, not a fourth positional field.** It is
packed today as `f"{session_id}_{channel}"` or
`f"{session_id}_{channel}_{quantile_name}"` and unpacked by `split("_")`,
which works only because channels carry no underscore while quantile names
do — hence the `"_".join(parts[2:])` in `split_series_id`. Adding image type
to that is how the encoding finally breaks.

It stays a string, because it has to be: the id is the table row's HTML
`id`, and four more element ids are built from it — `marker-button:`,
`plot-color:`, `scale:` and `label:` (`diagnostics_app.html:82`,
`diagnostics_app.js:15-38`) — and it is a key of the `datasets` object
posted back to `update_plot_view`. What changes is the encoding:

```python
series["id"] = "|".join(
    (str(session_id), image_type, channel, quantile_name or "")
)
```

- **A fixed four fields**, so `split("|")` needs no cleverness and an
  underscore anywhere is harmless. An absent quantile is the empty string
  rather than a missing field, keeping the count constant.
- **`|` cannot occur** in a session id, an `ImageType.name`, a channel or a
  `pixel_q*` name. `make_series` should assert that rather than trust it, so
  a future channel naming scheme fails loudly instead of producing an
  ambiguous id.
- **Not `:`**, which already separates the prefixes above; not JSON, which
  would need escaping to sit in an HTML attribute. `|` is valid in an HTML5
  `id`, and the code reaches these elements with `getElementById` rather
  than CSS selectors, so nothing needs escaping.

`make_series` and `split_series_id` remain the only two places that know the
format, so the change is contained to them and their tests. Keeping the
quantile last also means the existing characterization assertion — that a
quantile series id *ends with* its `pixel_q*` name — holds under the new
encoding as it did under the old, so that test needs no rewriting for this.

### 5. `diagnostics/forms.py` + `diagnostics/expression_views.py` (new)

> **Done**, together with §6, and **with no JavaScript of its own** — the
> largest departure from what this section and §6 first described. Three
> things were handed back to Django once it was asked what actually needed
> a client:
>
> - **Editing is a URL**, `expressions/edit/<slug:name>`, rendering the
>   same page with the form bound to that row, rather than a click that
>   fills the form from `data-` attributes. Linkable, survives a refresh,
>   works with JavaScript off, and it stops the form's field ids being
>   hard-coded in a second place. Under `edit/` rather than
>   `expressions/<name>` so an expression named `save` or `delete` cannot
>   collide with a literal route. A stale link 404s, which the error
>   middleware deliberately leaves to Django.
> - **Delete and Export are two submit buttons on one form**, using
>   HTML5's `form=` to sit in the left menu and `formaction=` to go to
>   their own views. One set of checkboxes serves both with no query
>   string built by hand. The consequence is that **export is a POST**;
>   nothing is lost, since the URL of a download whose content depends on
>   what is ticked is not worth bookmarking.
> - **The empty-selection case is the view's**, which says so, rather than
>   a client-side guard that swallowed the click.
>
> What stays client-side is table sorting only, for §9's reason: it moves
> the existing rows rather than rebuilding them.
>
> `_render_list` takes the library from `form.expressions` rather than
> fetching it again: the form was built with the library this request is
> about, and on a failed save the table must show what is *stored* rather
> than what was typed. Descriptions are the one stored column no rule is
> derived from, so they come from a separate `values_list` and are merged
> into each row.
>
> **No automated tests for the views**, deliberately. Standing Django up
> to drive them is more gymnastics than the coverage is worth, and the
> logic they call is tested where it lives and needs neither Django nor a
> project: `rename_references`, `check_expression`, `order_expressions`
> and the availability filter. What is left in the views is plumbing, and
> the manual pass under *Verification* covers it.

#### Validity is global; availability is per-project

Worth stating plainly, because the two are easily confused and only one of
them depends on which project is open.

**Validity does not.** After §1a, `check_expression(name, expression,
expressions)` takes no project input at all: the vocabulary comes from
`is_known_quantity()`, and no project can contain a diagnostic outside it,
because the only two creators are the static seeder and the quantile
branch that refuses everything else. An expression therefore means the
same thing in every project, which is what makes one shared library
coherent.

**Availability does.** *Is anything actually recorded here?* is answered by
`get_available_diagnostics()` (EXISTS probes, drives the dropdowns) and
`get_expression_availability()` (SQL counts, drives the series table). So
"hidden where its variables are not recorded" is availability, never
validity.

Three consequences:

- **The page works with no project open**, which is what a global library
  ought to do. Views still open `start_db_session()` when they need
  availability, but validity no longer depends on it, so there is nothing
  to degrade and nothing to 500 on.
- **The status column changes meaning, deliberately.** Validity is now
  project-independent, so "missing variables" fires only for genuinely
  unknown names — a typo, or a diagnostic no AutoWISP version defines.
  Whether an expression is usable *here* is a separate question, answered
  by `get_available_diagnostics()`'s bounded `EXISTS` probes, and shown as
  availability rather than as brokenness. Today's single status conflates
  "meaningless" with "no data recorded yet"; these are not the same
  complaint and should not read as one.
- Nothing is ever rewritten or removed for being unresolvable or
  unavailable somewhere. An expression naming a project-specific type still
  validates wherever that type is seeded and is merely unavailable
  elsewhere.

#### Outsource to Django wherever Django already knows

The BUI has no `forms.py`, no `ModelForm` and no class-based views today —
every view is a function doing its own POST handling. A `ModelForm` is
therefore a new pattern here, adopted deliberately: it deletes code that
would otherwise have to be maintained and kept correct by hand.

`diagnostics/forms.py` — `DiagnosticExpressionForm(ModelForm)` over
`DiagnosticExpression`, fields `name`, `expression`, `description`. Django
then owns:

| Would be hand-rolled | Django built-in |
| --- | --- |
| slug charset check | the model's `SlugField` validator |
| name already taken | `unique=True` → `validate_unique()` |
| create-vs-update branching | `ModelForm(instance=…)` + `form.save()` |
| per-field error plumbing | `form.errors`, rendered by the template |

What Django cannot know is whether the expression *means* anything, so
`clean()` calls `check_expression()` and raises its returned problem
strings as a `ValidationError`. **This is the one place that adaptation
happens** — tier 1 returns strings rather than raising precisely so it
stays usable without Django. `clean()` also records
`get_bare_aggregates()` on the form for the view to warn about; it is not
an error, so it must not fail validation.

`expressions` is a constructor keyword argument -- the library the
expression would join, which the view has and the form does not. No project
input is needed beyond it, since §1a made validity project-independent.

The name rules are thus checked twice — once by the model field, once by
`check_expression`. That overlap is deliberate: `check_expression` is the
authority for every path into the library, including import and, later,
the command line, none of which pass through this form.

Not outsourceable, and staying custom: the delete-dependents guard, and
the JSON import/export — `django.core.serializers` emits pk/model-label
records, which is the wrong shape for a portable file the CLI is meant to
read (see *Out of scope*).

#### The views

- `list_expressions` — table of expressions with a per-row status computed
  from `check_expression` alone (OK / missing variables / shadowed by a
  real diagnostic) — project-independent since §1a, with availability shown
  separately per *Validity is global; availability is per-project* above —
  plus a blank `DiagnosticExpressionForm`. Status comes from
  the same `check_expression()` the form uses, so the page and the save
  path can never disagree. Show each row's direct dependencies so a
  composed expression is traceable. **No all-NaN status** — every status
  here is derived from names alone, so the page costs no evaluation and no
  per-session query.
- `save_expression` — POST. Build the form with `instance=` the existing
  row when editing and `expressions=` the library; on `is_valid()`,
  `form.save()` then a non-blocking `messages.warning` per
  `form.bare_aggregates`. On failure re-render with `form.errors` rather
  than flattening them into `messages`, so problems land against the field
  that caused them. Editing is keyed by an `edit_name` POST field rather
  than by primary key, so the page is driven entirely by the names it
  shows and **a rename is an edit** — see below.
- `delete_expressions` — POST with checked names (mirror
  `home/views.py:delete_projects`), refusing any that other expressions
  reference and naming those dependents in the error. Dependents are judged
  against *what will remain*, so deleting a whole chain together is allowed
  while deleting only the bottom of it is not.
- `export_expressions` — JSON download, following the exact pattern of
  `home/views.py:509 export_master_config`:
  `json.dump` into a `StringIO`, `HttpResponse` with
  `Content-Disposition: attachment; filename="diagnostic_expressions.json"`.
  A selected-subset export must pull in the expressions its selection depends
  on, or the file will not import cleanly elsewhere — a transitive closure
  over `get_expression_names`, not just the direct references.
- `import_expressions` — POST, `json.load(request.FILES["expressions-import"])`,
  following `configuration/views.py:511 import_survey_info`. Refuses a file
  whose version key it does not recognise. Stages the whole incoming set and
  validates it as a unit against the staged set laid over what is already
  stored (so intra-file references resolve regardless of order), reports
  per-entry failures via `messages`, and upserts by name with an explicit
  overwrite-vs-skip choice for names that already exist.

#### Renaming — the cascade the delete guard does not cover

Not in the first draft of this plan, and the gap only shows up once
editing is keyed by name: **renaming an expression others reference
orphans them**, which is the delete guard's hazard reached from the other
side. Nothing rewrites expression text at resolution time, so a dependent
goes on naming the old spelling and is simply broken by it.

Blocking it the way deletion is blocked was the obvious first answer and
is the wrong one, because unlike a delete there is no gesture that makes
it legal: the dependent's *reference* is what has to change, and pointing
it at the new name will not validate while that name does not yet exist.
The only route left is copy under the new name → repoint each dependent →
delete the original, which is three steps and discoverable by nobody.

So the rename **succeeds and carries its dependents with it**, in one
transaction, and the page reports which expressions it updated. The delete
guard stays the only refusal.

- `rename_references(expression, old_name, new_name)` joins tier 1. Splices
  the new identifier in at the exact source offsets `ast` reports for each
  matching `ast.Name`, right to left, leaving every other byte alone.
- **Not `str.replace`**, which would corrupt `rel` inside `rel_bg`, inside
  a string literal, or a keyword argument's name. Only `ast.Name` nodes
  move, so all three are untouched by construction.
- **Not `ast.unparse`** either, which would reformat and re-parenthesise
  the user's text — the thing *Composition* rules out. This is a source
  edit the user asked for, not a resolution step, and what comes back is
  what they will edit next.
- Offsets from `ast` count **utf-8 bytes**, not characters, so the splice
  is done on the encoded text; otherwise a non-ASCII character earlier in
  the line shifts every offset after it.
- The dependents are written with `QuerySet.update()`, which the
  `modified` trigger covers where `auto_now` would not — see
  `core/models.py`.

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

> **Partly done**, split by which change they belonged to. Everything the
> axis merge needed has landed: `navigateDiagnostics()` lost its branch and
> `data-image-url`, the hard-coded `<option value="time">` is gone since
> `jd` is now an ordinary entry in `available_diagnostics`, `urls.py`
> collapsed onto one display route with `image/<slug:diagnostic_name>`
> redirecting to `x=jd`, and `views.py` dropped the `diag_vs_diag_views`
> re-exports. **Remaining** is everything belonging to the expressions page
> itself: the new template and JS, the two `<optgroup>`s, the left-menu
> button, the `expressions/*` routes and their re-exports.
>
> Keeping the redirect turned out to be load-bearing rather than courteous:
> `processing/progress.html` reverses `display_image_diagnostics` six times,
> so the URL *name* had to survive, not just the path.
>
> **Now done, with two changes to what is below.**
>
> - **No `<optgroup>`s, and no second list.** Expressions join
>   `available_diagnostics` itself. Splitting them would contradict what
>   §Namespace rests on: an axis reads *a name*, and a recorded diagnostic
>   is an expression of itself as far as anything downstream is concerned.
>   That is the same flatness that makes an expression shadowing a
>   diagnostic ambiguous, so the selector should not imply two kinds.
>   `get_available_diagnostics(recorded, expressions)` therefore returns
>   one list and the template loop is untouched.
> - **No `diagnostic.expressions.js`.** See §5 — the edit/delete row
>   interactions it was to carry are a URL and two submit buttons.
>
> The left-menu "Diagnostic Expressions" button and the `expressions/*`
> routes landed as described, plus `expressions/edit/<slug:name>`.
>
> One thing outside this section came with it: `core/lcars_app.html` now
> colours a message by level — red for error, atomic tangerine for
> warning, anakiwa for success, golden tanoi otherwise — all from the
> LCARS palette rather than invented. Warnings previously rendered
> identically to information, which this feature makes untenable: the
> bare-aggregate advice and the "renamed, and updated …" confirmation are
> not the same kind of thing.

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
  Expressions" button to the `left_menu` block (line 45). The x-selector's
  hard-coded `<option value="time">time</option>` becomes `jd`, which is now
  an ordinary entry in `available_diagnostics` rather than a special case, so
  the literal option disappears entirely.
- `diagnostics_app.js`: `navigateDiagnostics()` **loses its branch**. With one
  URL shape it reduces to a single `replace` pair, and `data-image-url` is no
  longer needed on `#diag-selector-bar`.
- `diagnostics/urls.py`: collapse the two display routes into
  `image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>` plus its `update_plot` and
  `download_plot` siblings, and **keep `image/<slug:diagnostic_name>` as a
  redirect** to `x=jd` so existing bookmarks survive. Add `expressions`,
  `expressions/save`, `expressions/delete`, `expressions/export`,
  `expressions/import`.
- `diagnostics/views.py`: re-export the new views through the hub, as it
  already does for the other modules, and drop the re-exports of the deleted
  `diag_vs_diag_views` names.

### 7. meson.build (mandatory — sources are listed explicitly, no globs)

> **Done.** Every file this plan adds is listed. Two entries below were
> never needed: `diagnostic.expressions.js`, since §5/§6 need no
> JavaScript, and therefore nothing in
> `diagnostics/static/diagnostics/js/meson.build`. Note this section
> covers more than §6 anticipated, since §3's tiers brought
> `autowisp/diagnostics/` and `autowisp/tests/` entries of their own.

- `diagnostics/meson.build`: `expression_data.py` — **done**; add
  `expression_views.py`.
- `diagnostics/migrations/meson.build`: `0001_initial.py` — **done**.
- `autowisp/diagnostics/meson.build`: `expression_series.py` — **done**
  (`expressions.py` landed with tier 1); add `diagnostic_types.py` (§1a).
- `diagnostics/meson.build`: add `forms.py` (§5).
- `autowisp/tests/meson.build`: `test_expression_series.py` — **done**.
- `browser_interface/static/js/meson.build`: `sortable.min.js` — **done**,
  with §9.
- `diagnostics/templates/diagnostics/meson.build`: add
  `diagnostic_expressions.html`.
- `diagnostics/static/diagnostics/js/meson.build`: add
  `diagnostic.expressions.js`.

### 8. Documentation

> **Done**, as a "Quantities of your own" section between *Choosing what
> to draw* and *Every point is a link* — expressions are a way of choosing
> what to draw, so that is where a reader meets them. Covers defining one,
> composition, the per-(session, type, channel) span of an aggregate, the
> `nan*` warning with `jd` named as the one exception, availability versus
> validity, import/export, and the `pixel_quantiles`-on-both-axes note.

`documentation/source/diagnostics.rst` is a complete narrative of the
diagnostics UI and is rendered into `docs/`. Add a section covering: defining
an expression, which variables are available, the vectorized/aggregate
semantics, **why aggregates want the `nan*` forms** — and that a bare one
usually yields an empty plot rather than an error, because a session normally
contains images without the diagnostic — building expressions out of other
expressions, project-availability, and import/export.

Two things the rest of the page needs to say once expressions exist:

- **`jd` is the one quantity that is never NaN**, so it is the sole
  legitimate exception to the "prefer `nan*`" guidance. It will otherwise
  look inconsistent.
- **`pixel_quantiles` is one quantile per series**, so selecting it for
  *both* axes draws each quantile against itself — the identity line.
  Pre-existing behaviour, deliberately left alone, and the answer is an
  expression: `pixel_q999 / pixel_q99` compares two of them properly.

### 9. Sorting the series table

> **Done.** Not part of the expression feature — an affordance of the same
> table §4 and §6 rework, added while that table was open.

The series table is ordered server-side by session, then type, then channel,
then quantile (`get_available_series`). Anything that cuts across that order
is then scattered down the table: every channel of one session reads well,
every session of one channel does not. Clicking a column heading re-keys it.

**Client-side, and by a vendored library rather than by our own code.**

*Client-side* because a row carries state that exists nowhere but in the DOM:
the `.active` class marking it selected, three live `<input>`s whose current
values `getSelectedDatasets()` reads straight back out, and the marker
`<svg>` that `selectSymbol()` swapped in. Re-sorting server-side behind a URL
parameter would reload the page and discard every one of them, which is why
the view, its URLs and `rows.sort(...)` are all left exactly as they are —
they set the order the table *arrives* in, and nothing more is asked of them.

*A library* because sorting a table by its headings is not a problem worth
solving again: [tofsjonas/sortable](https://github.com/tofsjonas/sortable)
4.1.7 is 1760 bytes, has no dependencies, needs no init call, and is in the
public domain (Unlicense), so the vendored copy carries no obligations. The
BUI already vendors third-party JS in `static/js/` — jQuery and
`jquery.orgchart` — so this is the established pattern rather than a new one.

The decisive property is *how* it sorts, not its size. It shallow-clones the
`<tbody>` and appends the **existing `<tr>` nodes** into the clone: rows are
moved, never rebuilt. Every piece of DOM-only state above therefore survives
a sort, as do the per-row click listeners `initImageDiagnostics()` attaches.
The libraries that rebuild rows from parsed cell data — the DataTables
family, `simple-datatables` — would silently drop all of it. **One thing the
shallow clone does not carry across is a listener bound to the `<tbody>`
itself**, so the per-row listeners must stay per-row and not be "simplified"
into one delegated to the body.

The rest the table already satisfies, or gets for nothing:

- Activation is `class="sortable"` on the `<table>`; the four control columns
  opt out with `class="no-sort"`. The generated headings stay inside their
  generic `{% for field in diagnostics_fields %}` loop — all of them sort,
  so the template still need not know what the fields are.
- **`class="asc"` as well**, which the library does not default to: without
  it the *first* click sorts descending, which reads as the table having been
  shuffled rather than ordered. With it the first click ascends and the
  second reverses, as everywhere else.
- `<thead>` and `<tbody>` are required, and both are already there.
- The comparator is `+a - +b` falling back to `localeCompare`, so Count and
  Quantile sort numerically without a `data-sort` attribute anywhere.
- `Array.sort` is stable, so successive clicks compose: session, then
  channel, gives channel-major with the sessions ordered inside each. This is
  why no tiebreaker is configured; `data-sort-tbr` on a heading is the escape
  hatch if one is ever wanted, at the cost of teaching the template which
  column index each field landed at.

Files:

- `static/js/sortable.min.js` — vendored **pinned to 4.1.7**, not `@latest`,
  with a provenance comment naming project, version and license, as
  `jquery.min.js` does. In the shared `static/js/` rather than under
  `diagnostics/`: it is generic, and the detrending and configuration tables
  are the obvious next users.
- `static/js/meson.build` — `'sortable.min.js'`, per §7's rule that sources
  are listed explicitly.
- `diagnostics_app.html` — `sortable` on the table, `no-sort` on the Color,
  Style, Scale and Label headings, and the `<script>` tag. Load order is
  free: the library registers one delegated listener on `document` and reads
  the DOM at click time.
- `diagnostics_app.css` — the arrow indicators only, with the transparent
  resting arrow kept so sorting does not shift the heading row sideways.
  **Its stylesheet is deliberately not vendored**: it is a theme — zebra
  stripes, a grey header background, its own `border-spacing` and `padding`
  — and every part of that fights LCARS and `table.standard-header`. The
  selectors are the upstream ones unchanged, so an upgrade stays a diff of
  one file, and they need no further scoping: `.sortable` is itself the
  opt-in, so a table without it is left alone.
- `documentation/source/diagnostics.rst` — a "Choosing what to draw"
  subsection introducing the table, its four editable columns, and sorting.

Deliberately **not** included: bulk selection. Sorting groups the rows one
wants together but still leaves a click per row, and the obvious completion
is a shift-click range select — perhaps ten lines of our own JS, since no
library is needed for it. Deferred rather than dismissed.

## Verification

1. **NaN-aware aggregates** — assert every name in `nan_aggregates` resolves in
   a bare `Evaluator()` *and* in a `LightCurveEvaluator`, and that each is the
   corresponding `numpy` function. This is the check that stops the two
   evaluators drifting apart again, and it is what makes a future asteval
   upgrade that drops a name fail loudly rather than silently. Also assert a
   data key wins over an aggregate of the same name, that a bad expression
   raises by default in **both** evaluators, and that an explicit
   `raise_errors=False` still returns `None`.
1. **The vocabulary (§1a) — done**, in `test_diagnostic_types.py`.
   `TestSeeding` asserts the catalogue equals what `_init_diagnostic_types()`
   actually writes, descriptions included, by seeding a project database and
   comparing the rows: that equality is the whole point of the extraction and
   the one thing that can silently drift once the literal lives elsewhere.
   `TestCatalogue` covers immutability (a caller cannot corrupt the shared
   mapping) and fully-interpolated descriptions; `TestRuntimePatterns` that
   `pixel_q999` matches while `pixel_quality` does not -- the `\Z`-anchored
   digits are what stop a plausible diagnostic name being swallowed.
1. **The predicate (§1a) — done**, in `test_expressions.py`.
   `TestQuantileNames` covers a quantile resolving as a variable *and* being
   refused as an expression name, since both directions come from one
   predicate and only the first is obvious, plus that ordering agrees with
   the direct check. `TestNoProjectNeeded` runs the project-free path end to
   end: `bg_center / diagonal_fov` accepted, `bg_centre / diagonal_fov`
   rejected, `jd` usable as a variable, with no database opened.
1. **Unit tests** covering `check_expression()` accept/reject cases (bad
   slug, reserved name, statement instead of expression, unknown variable,
   reference cycle) — which need **no database and no Django**, since tier 1
   takes its library and known names as arguments, and are correspondingly
   cheap to write exhaustively. Plus
   `get_expression_names()`, `get_bare_aggregates()` (flags
   `median`, ignores `nanmedian`), `order_expressions()` — a chain, a diamond
   where one expression feeds two dependents, a cycle, and the restriction to
   the target's subtree — the delete-with-dependents refusal, and an
   export→import round trip of a composed set. Plus
   `rename_references()`: a name used twice in one expression, a name that
   is a *substring* of another (`rel` inside `rel_bg` — the case that kills
   a textual replace), one inside a string literal, and the rename carried
   through the view so the dependents really are updated in the database. Also cover NaN padding and
   masking: a diagnostic missing for some images lands at the right indices,
   NaN propagates through a composed expression, and the finite mask keeps
   `image_ids` aligned with the plotted values.

   **All of it goes in `autowisp/tests/test_*.py`** — *not* in
   `diagnostics/tests.py`. The six BUI app `tests.py` files are untouched
   stubs, and nothing runs them: CI invokes only
   `python -m autowisp.tests …` plus one direct
   `python -m unittest autowisp.tests.test_database_migration`, and
   `start.py` runs `migrate` but never `manage.py test`. A test placed in the
   app would therefore never execute, in CI or via the documented command.
   New classes must additionally be imported by name into
   `autowisp/tests/__main__.py`, which lists test classes explicitly rather
   than discovering them.

   Four things the fixtures must account for, found while writing the
   characterization tests:

   - A lazily-initialized project DB creates the schema but seeds **no**
     `diagnostic_type` rows — `_init_diagnostic_types()` runs only on real
     project creation — so tests create every type they use. This is also why
     no `pixel_q*` type exists there: those are made at runtime.
   - SQLite does not enforce foreign keys, so an `ObservingSession` can carry
     dummy provenance ids rather than a full
     `Observer`/`Camera`/`Telescope`/`Mount`/`Observatory`/`Target` chain.
     `test_error_render.py:68` already relies on this.
   - Mocking `plot_image_diagnostic_series` keeps `reverse()` — and therefore
     Django settings — out of figure-level tests entirely, while capturing
     exactly the values that reach plotting.
   - **The user data directory is redirected centrally**, so no test has to
     remember to do it. `autowisp/tests/__init__.py` replaces
     `platformdirs.user_data_dir` with a session-scoped `TemporaryDirectory`
     before importing anything else. Django's default database would
     otherwise be the developer's real
     `~/.local/share/autowisp/bui_db.sqlite3` — the file holding their
     project list — and the same lookup also feeds `bui.log`, the default
     project home in `database/interface.py`, and `run_pipeline.out` in two
     places, so one redirect covers all five. It lives in the package
     `__init__` rather than `__main__` because CI also runs
     `python -m unittest autowisp.tests.…` directly, which never loads
     `__main__`; and it must precede the other imports because
     `settings.py` resolves the directory at *import* time.
   - Configuring Django from `autowisp/tests/` additionally requires
     `autowisp/browser_interface` on `sys.path`, since
     `django_project.settings` names its apps bare (`home`, `core`,
     `diagnostics`, …). A `DiagnosticExpression` test then needs
     `call_command("migrate", run_syncdb=True, verbosity=0)` against the
     redirected database.
1. **The `quantiles` regression test from §4, written and green before the
   merge**: one series per `pixel_q*` per (session, channel), quantile name
   in the series id, for both axis orders.
1. **Merge equivalence, also captured before the merge.** The two behaviours
   §4 must carry across are exactly the ones no small test project will reveal
   by eye, so pin them:
   - `x="jd"` reproduces the old time-series table and figure: one shared
     offset across all selected series (not per-series), axis label
     `JD - <min>`, and one subplot per non-overlapping night.
   - A non-time x collapses to a single axes, because the generalized
     x-range-overlap grouping finds every range overlapping.
   - `image/<name>` still resolves, via redirect, to `x=jd`.
1. **The image-type split**, which needs a fixture no existing test has: one
   session holding frames of more than one type, with a diagnostic recorded
   for several of them — `pixel_q*` is the real instance, since `calibrate`
   runs on dark, flat and object alike. Assert that the series table shows
   one row per type rather than one lumped row, that a diagnostic recorded
   for a single type yields exactly the rows it did before, and — the point
   of the whole change — that `nanmedian` over such a diagnostic is taken
   within a type rather than across them. The last of those is the only
   assertion that would catch a regression silently, since the first two
   merely look untidy when wrong.
2. **Pipeline tests** unaffected, but run them since the evaluator change in §1
   is shared pipeline code:
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
     appears in both selectors and plots against `jd` with the right per-series
     counts.
   - Confirm `jd` itself appears in both selectors, that `jd` vs a diagnostic
     is indistinguishable from the old time-series view, and that an expression
     using `jd` (e.g. `jd - nanmin(jd)`) validates, saves, and plots — the
     capability the merge unlocks.
   - Define `rel_bg = bg_center - nanmedian(bg_center)` to confirm aggregate
     (vectorized) functions work, and that the median is taken per session
     rather than across sessions. Then define the same thing with plain
     `median`: confirm the save-time warning appears, that it still saves, and
     that in a session where some image lacks `bg_center` it occupies a table
     row and plots empty. The management page must **not** flag it.
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
   - Rename `rel_astrom_residual` and confirm the composed expression that
     references it is rewritten to match, still plots, and that the page
     says which expressions it updated. Then rename an expression nothing
     references, which must simply work and report nothing.
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
   - **Sorting (§9)**, on a table with several sessions and channels: the
     Channel heading regroups the rows ascending, and clicking it again
     reverses them; Session then Channel gives channel-major with the
     sessions still ordered inside each; Count sorts numerically (9 before
     100, not after); the four control headings neither sort nor show an
     arrow. Then the check the library was chosen for: **select two rows,
     change one's colour and type a label, and sort.** The selection, the
     colour and the label must all survive, and the next row click must still
     update the plot — that is what proves the rows were moved rather than
     re-rendered.

     All of that except the appearance was checked headlessly first, by
     driving the vendored library against the really-rendered table in
     `jsdom` and asserting on node identity, so the browser pass is about how
     it looks rather than whether it works. Two things that check caught:
     the first click sorts *descending* without `class="asc"`, and the
     composition really does rest on `Array.sort` stability rather than on
     luck.
4. **Scaling** — the constraint in *Scaling* has to be checked, not assumed,
   and a small test project will not reveal a violation:
   - Run `EXPLAIN QUERY PLAN` on the canonical-image-list query, the
     availability subquery, and the per-series data query. Each must drive from
     `image` on `observing_session_id` and probe `image_diagnostics` by index;
     any plan that leads with a scan of `image_diagnostics` is a failure.
   - Confirm the `image_observing_session` index is actually used by these
     queries in a project database created *before* it existed and then brought
     forward with `autowisp/database/migrate_cli.py` — not only in a freshly
     created one, where `create_all` would have supplied it regardless.
   - Synthesise a large `image` / `image_diagnostics` set (a few hundred
     thousand rows across many sessions is enough to show the shape) and
     confirm that opening the series table and the expression management page
     stay responsive, and that their cost does not grow with the number of
     sessions beyond the row count of the table itself.
   - Confirm no code path evaluates an expression outside a session the user
     explicitly selected for plotting. The series table and the management page
     must both be reachable without a single call into the evaluator.
5. **Lint/format**: `pylint autowisp/browser_interface/diagnostics/` and Black
   at 80 columns; keep any incidental reformatting as its own commit.

## Out of scope (noted for later)

- Expressions over `PhotometryDiagnostics` (recorded by `fit_magnitudes` but
  surfaced nowhere in the BUI) and over the file-based detrending statistics.
- **Excluding images from magnitude fitting, EPD and TFA** — which is what
  expressions are ultimately *for*. The natural form is an
  `--fit-image-condition` beside the existing `--fit-source-condition`,
  evaluating to zero for images to drop, with the same wording. The tiering
  in §3 is what keeps this a small addition: tier 1 already evaluates a
  library against values handed to it, so a step needs only to fetch its
  diagnostics through tier 2 and apply the result. Deliberately not in this
  plan, which stops at defining, storing and visualising expressions.
- **Reaching expressions from the command line.** Command-line processing
  cannot assume a browser-interface database exists — that database is a
  browser-interface concept, and the CLI is handed a project home rather
  than discovering one — so expressions will need to arrive as
  configuration: a command-line option, or a file named by one.

  This is foreseen rather than solved here, but it shapes one decision that
  *is* in scope. The export format in §5 should be treated as the eventual
  configuration format, not merely as an interchange file, since a user
  exporting their library and pointing the CLI at it is the obvious path.
  Its version key (`autowisp_diagnostic_expressions: 1`) exists for exactly
  that reason. Keep it self-contained — a selected-subset export already has
  to pull in the expressions its selection depends on — so that a file taken
  from the BUI is directly usable without further resolution.
