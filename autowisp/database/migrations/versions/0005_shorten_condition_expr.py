"""Narrow ``condition_expression.expression`` to 768 characters.

Same cause as ``0003``: the column carries a unique index, and at 4 bytes
per character ``VARCHAR(1000)`` exceeds InnoDB's 3072-byte index limit, so
the table could not be created on MySQL or MariaDB under utf8mb4.

768 characters is far more than an expression over header keywords needs.
"""

from autowisp.database.migrations.helpers import resize_varchar_column

# revision identifiers, used by Alembic.
revision = "0005_shorten_condition_expr"
down_revision = "0004_shorten_image_raw_fname"
branch_labels = None
depends_on = None


def upgrade():
    """Narrow the column to 768."""

    resize_varchar_column(
        "condition_expression", "expression", new_length=768, old_length=1000
    )


def downgrade():
    """Widen the column back to 1000."""

    resize_varchar_column(
        "condition_expression", "expression", new_length=1000, old_length=768
    )
