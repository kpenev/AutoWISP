"""Drop the unique constraint on ``step.description``.

``VARCHAR(1000)`` is 4000 bytes under utf8mb4, over InnoDB's 3072-byte
index limit, so the table could not be created on MySQL or MariaDB.

The constraint is dropped rather than the column narrowed because it was
only ever a guard against a copy-pasted description. ``step.name`` carries
the uniqueness that matters, and keeps its constraint.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0006_drop_step_desc_unique"
down_revision = "0005_shorten_condition_expr"
branch_labels = None
depends_on = None

TABLE = "step"
COLUMN = "description"


def _unique_on_description(connection):
    """Reflected unique constraints and indexes covering just *COLUMN*.

    Backends disagree on which of the two a ``unique=True`` column becomes,
    so both are considered.
    """

    inspector = sqlalchemy.inspect(connection)
    found = [
        (constraint["name"], "constraint")
        for constraint in inspector.get_unique_constraints(TABLE)
        if constraint["column_names"] == [COLUMN]
    ]
    found += [
        (index["name"], "index")
        for index in inspector.get_indexes(TABLE)
        if index.get("unique") and index["column_names"] == [COLUMN]
    ]
    return found


def upgrade():
    """Remove the uniqueness, however this backend recorded it."""

    connection = alembic.op.get_bind()
    existing = _unique_on_description(connection)
    if not existing:
        return

    if connection.dialect.name == "sqlite":
        # Declared inline and therefore unnamed, so it cannot be dropped by
        # name; rebuild the table from a reflected copy without it.
        reflected = sqlalchemy.Table(
            TABLE, sqlalchemy.MetaData(), autoload_with=connection
        )
        for constraint in list(reflected.constraints):
            if isinstance(constraint, sqlalchemy.UniqueConstraint) and [
                column.name for column in constraint.columns
            ] == [COLUMN]:
                reflected.constraints.discard(constraint)
        with alembic.op.batch_alter_table(
            TABLE, copy_from=reflected, recreate="always"
        ):
            pass
        return

    for name, kind in existing:
        if kind == "index":
            alembic.op.drop_index(name, table_name=TABLE)
        else:
            alembic.op.drop_constraint(name, TABLE, type_="unique")


def downgrade():
    """Restore the uniqueness."""

    connection = alembic.op.get_bind()
    if _unique_on_description(connection):
        return
    alembic.op.create_unique_constraint(f"uq_{TABLE}_{COLUMN}", TABLE, [COLUMN])
