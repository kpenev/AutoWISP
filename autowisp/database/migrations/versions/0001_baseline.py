"""Baseline: the schema as of AutoWISP 1.8.1.

Deliberately empty. This revision is never executed -- it exists only as a
point to stamp, marking "this database has the 1.8.1 schema".

Project databases predate Alembic, so an existing one carries no revision of
its own and its true age is unknowable (the installed AutoWISP version
describes the code, not the database). Rather than trying to identify what a
project is, ``migrate_project()`` brings it to a known state with
``apply_additive_migrations()`` and stamps it here, then applies everything
that follows. See ``project_db_migrations_plan.md``.
"""

# revision identifiers, used by Alembic.
revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Nothing to do: this revision only marks the starting schema."""


def downgrade():
    """Nothing to do: this revision only marks the starting schema."""
