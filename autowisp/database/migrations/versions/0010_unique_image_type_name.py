"""Make ``image_type.name`` unique.

Every lookup of an image type goes through its name --
``add_master_dependencies`` resolves one to an id, and the processing and
lightcurve layers filter on ``ImageType.name == ...`` -- so the code has
always assumed the name identifies the row. This states that assumption
where it can be enforced.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0010_unique_image_type_name"
down_revision = "0009_float_columns_to_double"
branch_labels = None
depends_on = None

CONSTRAINT_NAME = "uq_image_type_name"


def _is_unique(connection):
    """Whether the column already carries a uniqueness guarantee.

    Both spellings count: the backends reflect an inline UNIQUE
    differently, MySQL as a unique index and SQLite as a constraint.
    """

    inspector = sqlalchemy.inspect(connection)
    constrained = [
        constraint["column_names"]
        for constraint in inspector.get_unique_constraints("image_type")
    ] + [
        index["column_names"]
        for index in inspector.get_indexes("image_type")
        if index["unique"]
    ]
    return ["name"] in constrained


def upgrade():
    """Add the constraint, unless the column already has one."""

    if _is_unique(alembic.op.get_bind()):
        return

    # Rebuilds the table on SQLite, which cannot add a constraint in
    # place. That drops the table's timestamp trigger; migrate_project
    # reinstates it once the revisions are done.
    with alembic.op.batch_alter_table("image_type") as batch:
        batch.create_unique_constraint(CONSTRAINT_NAME, ["name"])


def downgrade():
    """Drop the constraint, if it is there."""

    if not _is_unique(alembic.op.get_bind()):
        return

    with alembic.op.batch_alter_table("image_type") as batch:
        batch.drop_constraint(CONSTRAINT_NAME, type_="unique")
