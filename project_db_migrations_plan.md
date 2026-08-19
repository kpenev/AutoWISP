# Project Database Migrations via Alembic

*Branch: `project_db_migrations`. The `diagnostic_expressions_plan.md` work
depends on this and follows after.*

## Context

Project databases (`autowisp.db`, or a centralised MySQL/MariaDB pointed at by
`autowisp_db.url`) have no migration framework. Their stand-in is
`apply_additive_migrations()` in `autowisp/database/interface.py:42` —
`create_all` to pick up new tables, plus a hardcoded list of nullable columns to
`ALTER TABLE ... ADD`:

```python
additive_columns = [
    ("pipeline_run", "code_version", "VARCHAR(1000)"),
    ("error", "resolved", "TIMESTAMP"),
]
```

It has run out of room. The immediate trigger is an index:
`image.observing_session_id` is unindexed (`data_model/image.py:96` — a plain
`ForeignKey`, and SQLite does not index foreign keys), so every session-scoped
query full-scans `image`. `create_all` will not add an index to an existing
table, and the helper has no concept of indexes, so there is nowhere to put the
fix. The same will be true of the next non-column change.

An earlier draft of this plan hand-rolled a registry of migration classes with
its own ledger table and version ordering. **Alembic is purpose-built for
exactly this** and is adopted instead: it supplies the revision graph, the
version table, stamping of pre-existing databases, SQLite batch mode, and
model-vs-migration drift detection, all of which that draft would have
reimplemented more poorly.

What Alembic does *not* decide, and what this plan is therefore mostly about,
is **when and where migrations run** — because `apply_additive_migrations()` is
currently called from `set_project_home()`, which every pipeline worker process
executes.

## Design

### Layout

```
autowisp/database/
├── migrate.py                    # our policy layer: when/where/locking
└── migrations/
    ├── env.py                    # uses the connection we hand it
    ├── script.py.mako            # revision template
    └── versions/
        ├── 0001_baseline.py      # empty; represents the 1.8.1 schema
        └── 0002_image_observing_session_index.py
```

**No `alembic.ini`.** The `Config` is built in code, with `script_location`
resolved from `os.path.dirname(__file__)`, so an installed package has no
relative-path problem and there is no config file to ship or keep in sync:

```python
def _alembic_config():
    cfg = Config()
    cfg.set_main_option(
        "script_location", os.path.join(os.path.dirname(__file__), "migrations")
    )
    return cfg
```

`env.py` takes the caller's connection rather than building its own engine —
the documented pattern for applications that already own an `Engine`, which
`autowisp/database/interface.py` does:

```python
connectable = config.attributes.get("connection", None)
context.configure(
    connection=connectable,
    target_metadata=DataModelBase.metadata,
    render_as_batch=True,          # SQLite ALTER TABLE limitations
)
```

### Baseline: adopting databases that predate Alembic

Revision `0001_baseline` is empty (`def upgrade(): pass`). It represents "the
schema `apply_additive_migrations()` produces", i.e. 1.8.1. Every real revision
descends from it.

`apply_additive_migrations()` is **kept**, demoted to the step that reaches the
baseline. It is already exactly the code that brings an older project up to
1.8.1, and it is idempotent. Three cases, and they are the only ones:

| database state | action |
|---|---|
| has `alembic_version` | `command.upgrade(cfg, "head")` |
| AutoWISP tables, no `alembic_version` | `apply_additive_migrations()` → `command.stamp(cfg, "0001_baseline")` → `upgrade(cfg, "head")` |
| fresh project | `create_all` (existing `initialize_cmdline_database()`) → `command.stamp(cfg, "head")` |

This handles a pre-ledger project of **any** age with no user action and no
minimum supported version. Note the installed version cannot be used to infer a
project's schema — `importlib.metadata.version("autowisp")` describes the code,
so a 1.9 install opening a 1.5 project reports 1.9. Hence: bring it to a known
state, then stamp that.

`apply_additive_migrations()` is called only from this path, never from
`set_project_home()`, and stops being where new schema changes go — its
additive-column list is frozen at its 1.8.1 contents. Its docstring currently
describes it as the general mechanism and must be updated to say otherwise.

#### Why the helper is not simply converted into revisions

The obvious tidy-up is to express `additive_columns` as Alembic revisions and
delete `apply_additive_migrations()` entirely. It does not work, and the reason
is worth recording so the question is not re-opened without it.

The two **columns** would convert cleanly. Written idempotently — inspector
check, then `op.add_column` — a legacy database stamped at a pre-column
revision runs them as no-ops where the columns already exist and applies them
where they do not, which is correct for any legacy state.

The **`create_all` half cannot**. It is a catch-all for "any table added at any
point since this project was initialised" (the `error` table being the
precedent), and expressing that as revisions requires one of two unacceptable
things:

- an `op.create_table()` per table in the revision that introduced it — but
  that history is not recorded anywhere, so it would have to be reconstructed
  by git archaeology, to cover databases that may predate any of those tables;
- a revision calling `DataModelBase.metadata.create_all(connection)` — exactly
  the drift anti-pattern forbidden under *Adding a migration*. It would create
  *today's* table set rather than the one contemporary with the revision, its
  meaning would change every time a model is added, and later table-adding
  revisions would become no-ops on some databases but not others.

So `create_all` has to remain an adoption step outside the revision chain. And
once it does, converting only the columns leaves **both** mechanisms in place
plus two extra revisions — worse than either endpoint. Keeping the helper
whole, demoted to the baseline step, is the coherent choice.

There is a clean endpoint, and it is a separate piece of work: autogenerate a
revision `0001` that creates the entire schema, and have fresh projects
`upgrade head` instead of `create_all`. That gives one mechanism everywhere,
removes `create_all` from project initialisation as well, and genuinely
retires the helper. It costs a large generated revision to review once and a
re-test of project creation, so it belongs **after** Alembic is bedded in and
the index migration has shipped — not in this change.

### Where migrations run — and where they must not

`apply_additive_migrations()` runs today from `set_project_home()`
(`interface.py:275`), which every process opening a project calls, **including
every pipeline worker**. That is survivable for check-then-add column logic and
not survivable for general DDL — dozens of processes racing to `CREATE INDEX`.
Split the two roles:

- **`check_project_schema(engine)`** — read-only, and what `set_project_home()`
  calls. Compares `MigrationContext.get_current_heads()` against
  `ScriptDirectory.get_heads()`; raises an actionable error naming the pending
  revisions and pointing at `wisp-migrate` if behind. A worker therefore never
  issues DDL, and a stale project fails loudly instead of silently
  misbehaving.

  Cache the script-directory head at module level. It is constant for a given
  install, and re-walking `versions/` in every worker on every open is pure
  waste; with it cached the check is one `SELECT` from `alembic_version`.

- **`migrate_project(engine)`** — the only thing that mutates. Takes a lock,
  applies the three-case logic above, returns what it did.

Callers of `migrate_project()`:

- **BUI project selection** — `select_project()` in
  `browser_interface/home/views.py`, reporting results via `messages`. The
  landing page can also flag projects needing migration, alongside the existing
  `find_missing_databases()` column.
- **`wisp-migrate <project_home>`** — new console script in `pyproject.toml`,
  for non-BUI use and scripting.
- **`run_pipeline`** — once, in the main process, before any worker spawns. So
  CLI users do not meet the `check_project_schema()` error in normal use.

### What is still ours despite Alembic

- **Locking.** Alembic has none — it reads `alembic_version`, runs the
  revisions and writes the new head, with no advisory lock anywhere, so two
  concurrent `upgrade` calls both read the same revision and both run it.
  Reachable here: a centralised MySQL project database serves several users at
  once, Django's `runserver` is threaded so two clicks on project selection are
  two concurrent `select_project()` calls, and starting `run_pipeline` while
  the BUI is open is ordinary. `migrate_project()` therefore takes a lock,
  branching on `connection.dialect.name`:

  - **SQLite** — `BEGIN IMMEDIATE`, which takes the write lock for the
    transaction. SQLite has transactional DDL, so this holds for the whole
    migration.
  - **MySQL/MariaDB** — `GET_LOCK('autowisp_migrate', timeout)` up front and
    `RELEASE_LOCK('autowisp_migrate')` at the end. It must be this rather than
    a transaction-scoped lock such as `SELECT ... FOR UPDATE` on a lock row:
    **DDL on MySQL causes an implicit commit**, which would release a
    transaction-scoped lock partway through the first revision — precisely when
    it is needed. `GET_LOCK` is session-scoped and survives those commits.
    Acquire it on the same connection that is then handed to Alembic, and
    release it in a `finally`.

  What the lock is really protecting is data migrations and the
  `alembic_version` update itself; the idempotency required below already makes
  a doubly-applied pure-DDL revision a no-op.
- **MySQL has no transactional DDL.** DDL forces an implicit commit, so a
  failed revision cannot be rolled back there the way it can on SQLite. Keep
  each revision to a **single DDL statement** so there is no partial state, and
  rely on idempotent operations for the crash window between the DDL and the
  `alembic_version` update. A change needing several DDL steps becomes several
  revisions.
- **Backups.** For SQLite, copy the file with a `.pre-<revision>` suffix before
  mutating. For a server database there is no file to copy and no rollback —
  require an explicit acknowledgement flag and say so in the message.

### Adding a migration — worked example

The index `diagnostic_expressions` needs. Two edits plus a generated file.

**1. Declare it on the model**, so fresh projects get it from `create_all`
(`autowisp/database/data_model/image.py`):

```python
class Image(DataModelBase):
    """The table describing the image specified"""

    __tablename__ = "image"
    __table_args__ = (
        Index("image_observing_session", "observing_session_id", "jd"),
    )
```

**2. Generate the revision** and edit it down to the intended change:

```bash
wisp-migrate --autogenerate -m "index image.observing_session_id" <project_home>
```

What that command does: Alembic reflects the schema of the project database it
is pointed at, diffs it against `DataModelBase.metadata` (the ORM models,
supplied as `target_metadata` in `env.py`), and writes a new file into
`versions/` whose `upgrade()` and `downgrade()` bodies are prefilled with the
differences it found. `-m` becomes the revision docstring and part of the
filename, and `down_revision` is wired to the current head automatically. So
having done step 1, the diff is exactly "the models declare an index the
database does not have", and the generated body is the `op.create_index()` call
shown below.

Three things about this step:

- **It is a developer action, not a user one.** It needs a project database
  that is already at `head` to diff against — a database behind head produces
  a revision containing the *pending* changes as well.
- **It goes through `wisp-migrate` rather than the `alembic` CLI** because
  there is no `alembic.ini`: the bare `alembic` command would have no
  `script_location` to find `versions/`, and no way to resolve a project home
  to a database URL. The wrapper builds the `Config` and the engine.
- **The output is a draft.** Autogenerate misses some things and invents
  others — it cannot see server defaults or certain constraint changes, and it
  reports spurious diffs where SQLAlchemy renders a type differently from the
  way the backend reflects it. Always read the generated file, delete
  everything that was not intended, and replace the one-line `-m` message with
  a docstring saying *why* (as below). Committing autogenerate output unread is
  how unrelated schema drift gets shipped inside an unrelated migration.

```python
"""index image.observing_session_id

Every diagnostics query is scoped to one observing session, and SQLite does
not index foreign keys, so without this each one full-scans `image` --
unusable on collections running to millions of rows.  `jd` is included
because those queries also order by it.
"""

revision = "0002_image_observing_session_index"
down_revision = "0001_baseline"


def upgrade():
    op.create_index(
        "image_observing_session", "image", ["observing_session_id", "jd"]
    )


def downgrade():
    op.drop_index("image_observing_session", table_name="image")
```

**3. Add the file to `versions/meson.build`** — sources are listed explicitly,
so a revision missing from that list is absent from an installed build while
still working from a checkout. This is the step most likely to be forgotten;
the parity test below catches it.

Two rules for writing revisions:

- **Never import the ORM models.** A revision describes the schema as of its
  own release and must never change meaning afterwards; referencing the live
  model would silently re-point it every time that model is edited. Use string
  table/column names, or reflect from the connection.
- **Which means the definition is written twice** — once on the model, once in
  the revision — and the two must agree. Do not fix this by sharing the object;
  `alembic check` (below) fixes it by detecting the divergence.

### Revision ids: hand-assigned, and enforced

Revision ids are hand-assigned as `NNNN_slug` rather than left as Alembic's
random hex, so the chain reads at a glance and `versions/` sorts correctly.
Pass them with `--rev-id`; `wisp-migrate` should default the number to
head + 1.

This costs nothing in collision terms. Two branches adding a revision fork the
chain whichever scheme is used — both set `down_revision` to the same parent,
producing two heads, and Alembic then refuses `upgrade head` as ambiguous.
Random ids do not prevent that; they only make it less legible.

The convention is therefore:

- **Resolve a fork by re-pointing `down_revision`** at the other branch's
  revision, linearising the chain. Not by `alembic merge`, which creates a
  revision with two parents — this project wants a linear history.
- **A shipped revision id is permanent.** Databases in the field are stamped
  with that exact string, so renaming one strands them. Renumber freely before
  a release, never after. Which means a fork resolved post-release can leave a
  `0003` sitting after a `0004` in chain order — cosmetically odd, entirely
  correct, and preferable to breaking stamped databases.

Both failure modes are caught mechanically rather than by discipline — see
*Verification*: one test asserts a single head and a strictly increasing
numeric prefix from base to head, which detects an unresolved fork, a duplicate
number, and a revision that does not follow the naming pattern.

## Implementation

1. Add `alembic` to `dependencies` in `pyproject.toml`.
2. Scaffold `autowisp/database/migrations/` — `env.py`, `script.py.mako`,
   `versions/`, each with `meson.build` entries. `env.py` uses the passed-in
   connection and sets `render_as_batch=True`.
3. `autowisp/database/migrate.py` — `_alembic_config()`,
   `check_project_schema()`, `migrate_project()`, the three-case baseline
   logic, the dialect-specific lock, and the SQLite backup.
4. `versions/0001_baseline.py` — empty revision.
5. `versions/0002_image_observing_session_index.py` + the matching
   `Index(...)` in `Image.__table_args__`.
6. Replace the `apply_additive_migrations()` call at `interface.py:275` with
   `check_project_schema()`; keep the function, called only from the baseline
   path, and update its docstring.
7. `wisp-migrate` console script — subcommands for `upgrade` (default),
   `--autogenerate -m`, and `current`.
8. `select_project()` in the BUI calls `migrate_project()` and reports results.
9. `run_pipeline` calls `migrate_project()` once in the main process.
10. Extend `autowisp/tests/test_database_migration.py`; its existing coverage of
    `apply_additive_migrations` stays valid since the function survives.

## Verification

1. **Drift detection**: `alembic check` in CI — it compares the ORM metadata
   against the revision chain and fails on divergence. This is what keeps each
   revision honest against the model declaration it duplicates, and it covers
   every future revision, including one missing from `versions/meson.build`
   when run against an installed build.
2. **Chain shape**, as a plain unit test so it runs locally as well as in CI.
   Walk `ScriptDirectory.from_config(...)` and assert:
   - `len(script.get_heads()) == 1` — a fork left unresolved by a merge or
     rebase is the failure that actually breaks `upgrade head`, and this is
     where it gets caught rather than at a user's next project open;
   - every `revision` matches `NNNN_slug`;
   - numeric prefixes are **strictly increasing** from base to head, which
     catches a duplicate number and a revision merged out of order;
   - no revision has more than one parent, keeping the history linear.

   Roughly:

   ```python
   script = ScriptDirectory.from_config(_alembic_config())
   assert len(script.get_heads()) == 1, "unresolved fork: re-point down_revision"
   numbers = [int(rev.revision[:4]) for rev in script.walk_revisions()]
   assert numbers == sorted(set(numbers), reverse=True)
   ```
3. **Unit tests** in `autowisp/tests/test_database_migration.py`:
   - a fresh database is stamped at `head` with no revision executing;
   - a 1.8.1 database with no `alembic_version` is brought to baseline,
     stamped, upgraded, and is a no-op on a second run;
   - an **older** database — one lacking `error.resolved` — reaches the same
     end state in the same single pass, with no user action;
   - the **MySQL crash window**: DDL applied but `alembic_version` not
     advanced, simulated by applying a revision and resetting the table. The
     re-run must be a clean no-op, not an error;
   - `check_project_schema()` raises on a behind database, passes on a current
     one, and does not mutate either.
4. **Concurrency**: two `migrate_project()` calls racing on one database — one
   applies, the other waits and finds nothing to do. Exercise on both backends,
   since the lock path differs. Then confirm a worker calling
   `set_project_home()` against a behind database raises rather than attempting
   DDL.
5. **Dialects**: the suite runs against SQLite **and** MySQL/MariaDB — both are
   in real use, and they differ on exactly the points this rests on (batch
   mode, transactional DDL, locking). Add a MySQL service to CI, or gate those
   tests on a connection URL in the environment.
6. **End to end**: take a project database created by the current release, open
   it in the BUI, confirm the migration is reported, the index exists
   afterwards, and `EXPLAIN QUERY PLAN` shows a session-scoped query using it.
7. `python -m autowisp.tests failed_test -v` — the pipeline suite opens project
   databases throughout and is the real regression net.

## Relationship to `diagnostic_expressions`

This lands first, on `project_db_migrations` (already branched off `master`).
The `image_observing_session` index is delivered here.
`diagnostic_expressions_plan.md` already assumes it and rebases onto this work.
