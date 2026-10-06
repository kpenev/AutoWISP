"""Correct the spelling in the description of ``master-mask``.

The help ``calibrate`` gives ``--master-mask`` misspelled "preceded", and
project creation stores that help as the parameter's description, which the
configuration page shows. New projects get the corrected text; this
corrects existing ones.

This revision changes rows, not the schema. The texts below are a copy of
what project creation writes, made so that this revision keeps doing what it
did if that changes.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0017_master_mask_description"
down_revision = "0016_diagnostic_definitions"
branch_labels = None
depends_on = None

PARAMETER = "master-mask"

#: The description before the correction and after it.
DESCRIPTIONS = tuple(
    "Mask(s) to apply, indicating pixel quality. All pixels are considered "
    '"good" if no mask is specified. If multiple channel images are being '
    f"processed each master filename should be {spelling} by "
    "``<channel name>:`` identifying which channel it applies to. Unlike "
    "other masters, channels without mask are allowed and multiple masks may "
    "be used for each channel."
    for spelling in ("preceeded", "preceded")
)


def _set_description(which):
    """Give the parameter its old (0) or new (1) description.

    Only where it still has the other one: a project without the parameter,
    or with a description of its own, is left as it is.
    """

    connection = alembic.op.get_bind()
    parameters = sqlalchemy.Table(
        "parameter", sqlalchemy.MetaData(), autoload_with=connection
    )
    connection.execute(
        parameters.update()
        .where(
            parameters.c.name == PARAMETER,
            parameters.c.description == DESCRIPTIONS[1 - which],
        )
        .values(description=DESCRIPTIONS[which])
    )


def upgrade():
    """Give the parameter the corrected description."""

    _set_description(1)


def downgrade():
    """Restore the misspelled description."""

    _set_description(0)
