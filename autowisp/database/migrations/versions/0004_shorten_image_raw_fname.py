"""Narrow ``image.raw_fname`` to 768 characters.

At 4 bytes per character a ``VARCHAR(1000)`` unique key is 4000 bytes, over
InnoDB's 3072-byte index limit -- so with a utf8mb4 charset the table could
not be created on MySQL or MariaDB at all. 768 is the widest that fits.

A longer path is now rejected on insert. That is deliberate: the
alternatives were a prefix index, which would report a spurious duplicate
for two files sharing a deep directory tree, and a hash column, which
cannot be maintained database-side because SQLite has no hash function.
"""

from autowisp.database.migrations.helpers import resize_varchar_column

# revision identifiers, used by Alembic.
revision = "0004_shorten_image_raw_fname"
down_revision = "0003_repair_timestamp_triggers"
branch_labels = None
depends_on = None


def upgrade():
    """Narrow the column to 768."""

    resize_varchar_column("image", "raw_fname", new_length=768, old_length=1000)


def downgrade():
    """Widen the column back to 1000."""

    resize_varchar_column("image", "raw_fname", new_length=1000, old_length=768)
