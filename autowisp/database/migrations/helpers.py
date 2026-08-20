"""Mechanical operations shared between migration revisions.

Revisions may import from here, unlike from the ORM models. The difference
is that a model *describes a schema* and moves over time, so a revision
referring to one silently changes meaning; these functions describe only
*how* to perform an operation, and take the schema as arguments.

That distinction only holds while the functions stay mechanical, so:

- **Never encode a table, column or type here.** Those belong in the
  revision, which is frozen; this module is not.
- **Never change what an existing function does.** Past revisions call it,
  and altering its behaviour rewrites history for every database that has
  not yet upgraded. Add a new function instead.

Everything here is idempotent by inspection rather than by ``IF EXISTS``,
which MySQL 8 does not accept for every object type, and because DDL
commits implicitly on MySQL -- a crash between the change and the
``alembic_version`` update leaves the change applied but unrecorded, so the
re-run has to succeed.
"""

import sqlalchemy as sa
from alembic import op


def _column_length(connection, table, column):
    """Return the declared length of *column*, or None if it is absent."""

    for described in sa.inspect(connection).get_columns(table):
        if described["name"] == column:
            return getattr(described["type"], "length", None)
    return None


def resize_varchar_column(
    table, column, *, new_length, old_length, nullable=False
):
    """Change a VARCHAR column's declared length.

    Does nothing if the column is already *new_length*, or is missing.

    SQLite cannot alter a column in place, so this goes through
    ``batch_alter_table``, which rebuilds the table; on other backends it
    is a plain ``ALTER``.

    Args:
        table(str):    Name of the table holding the column.

        column(str):    Name of the column to resize.

        new_length(int):    Length to change it to.

        old_length(int):    Length it is expected to have, which the
            backends that need it use to render the existing type.

        nullable(bool):    The column's existing nullability, preserved
            across the change.
    """

    connection = op.get_bind()
    current = _column_length(connection, table, column)
    if current is None or current == new_length:
        return

    with op.batch_alter_table(table) as batch:
        batch.alter_column(
            column,
            existing_type=sa.String(old_length),
            type_=sa.String(new_length),
            existing_nullable=nullable,
        )


def drop_foreign_keys_to(table, referred_table):
    """Drop every foreign key on *table* that points at *referred_table*.

    Does nothing if there are none, so re-running is safe.

    The constraint is found by reflection rather than by name because the
    two backends name it differently -- MySQL generates something like
    ``configuration_ibfk_2`` while SQLite, which declares foreign keys
    inline, has no name at all. The nameless case cannot be dropped by
    ``drop_constraint``, so there the table is rebuilt from a reflected
    definition with the constraint removed.

    Args:
        table(str):    Table whose foreign keys should be dropped.

        referred_table(str):    Only keys pointing here are dropped.
    """

    connection = op.get_bind()
    matching = [
        key
        for key in sa.inspect(connection).get_foreign_keys(table)
        if key["referred_table"] == referred_table
    ]
    if not matching:
        return

    if connection.dialect.name == "sqlite":
        reflected = sa.Table(table, sa.MetaData(), autoload_with=connection)
        for constraint in list(reflected.constraints):
            if isinstance(constraint, sa.ForeignKeyConstraint) and (
                constraint.referred_table.name == referred_table
            ):
                reflected.constraints.discard(constraint)
        with op.batch_alter_table(
            table, copy_from=reflected, recreate="always"
        ):
            pass
        return

    for key in matching:
        op.drop_constraint(key["name"], table, type_="foreignkey")
