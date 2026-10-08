"""Index ``image.observing_session_id``, covering ``ORDER BY jd``.

Diagnostics and photometric-reference queries are all scoped to a single
observing session, but ``observing_session_id`` is a plain foreign key and
SQLite does not index foreign keys, so each of those queries scans the whole
``image`` table. That is tolerable for a test project and not for the
collections AutoWISP is aimed at, which run to millions of images.

``jd`` is part of the index because those same queries order by it, letting
the index serve the sort as well as the filter.
"""

import sqlalchemy
import alembic

# revision identifiers, used by Alembic.
revision = "0002_image_session_index"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

INDEX_NAME = "image_observing_session"


def _index_exists(connection):
    """Whether the index is already on the table.

    Checked by inspection rather than with ``IF NOT EXISTS``, which SQLite
    and MariaDB accept for indexes but MySQL 8 does not.
    """

    return INDEX_NAME in {
        index["name"]
        for index in sqlalchemy.inspect(connection).get_indexes("image")
    }


def upgrade():
    """Create the index, if it is not already there.

    Idempotent on purpose. MySQL commits DDL implicitly, so a crash between
    this statement and the ``alembic_version`` update leaves the index
    present but the revision unrecorded; the re-run has to succeed rather
    than fail on an index that already exists.
    """

    connection = alembic.op.get_bind()
    if not _index_exists(connection):
        alembic.op.create_index(
            INDEX_NAME, "image", ["observing_session_id", "jd"]
        )


def downgrade():
    """Drop the index, if it is there."""

    connection = alembic.op.get_bind()
    if _index_exists(connection):
        alembic.op.drop_index(INDEX_NAME, table_name="image")
