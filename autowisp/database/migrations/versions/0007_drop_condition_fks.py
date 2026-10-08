"""Drop the foreign keys pointing at ``condition.id``.

A condition is a *set* of expressions: ``condition`` holds one row per
member, all sharing an ``id``, so ``condition.id`` identifies a group
rather than a row and is not unique. A foreign key asserts the opposite.

InnoDB accepted that as a documented non-standard extension until MySQL
8.4, which began rejecting it (``ER_FK_NO_UNIQUE_INDEX_PARENT``) and so
could not create the schema at all. MariaDB still allows it, which is why
existing deployments were unaffected.

The columns and their values are untouched; only the constraint goes. On
SQLite nothing changes in practice, since foreign keys are not enforced
unless ``PRAGMA foreign_keys`` is on, and AutoWISP never turns it on.
"""

from autowisp.database.migrations.helpers import drop_foreign_keys_to

# revision identifiers, used by Alembic.
revision = "0007_drop_condition_fks"
down_revision = "0006_drop_step_desc_unique"
branch_labels = None
depends_on = None

# master_type carries two of them: condition_id and
# maker_image_split_condition_id.
REFERRING_TABLES = ("configuration", "master_type")


def upgrade():
    """Drop every foreign key referencing the condition group id."""

    for table in REFERRING_TABLES:
        drop_foreign_keys_to(table, "condition")


def downgrade():
    """Deliberately not reinstated.

    Recreating these would fail on MySQL 8.4 for the reason they were
    dropped, so a downgrade leaves the columns as plain integers.
    """
