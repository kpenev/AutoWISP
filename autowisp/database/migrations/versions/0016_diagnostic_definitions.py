"""Bring existing projects' diagnostic definitions up to a new project's.

Two changes, both to what project creation writes about diagnostics:

* The photometric reference selection page ranks the candidates by an
  expression from the project's library, ``photref_merit`` unless another is
  chosen there, rather than by a merit function typed into the page and kept
  in the browser session. New projects get ``photref_merit`` when they are
  created; this adds it to existing ones. A project that already has an
  expression of that name keeps its own.

* The descriptions of ``s_map_residual``, ``d_map_residual`` and
  ``k_map_residual`` were written with ``{param.upper()}`` left in them
  literally, a placeholder missing the ``f`` of its f-string. New projects get
  the parameter's name there instead; this corrects existing ones.

This revision changes rows, not the schema. The texts below are a copy of
what project creation writes, made so that this revision keeps doing what it
did if that changes.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0016_diagnostic_definitions"
down_revision = "0015_engine_set_masters"
branch_labels = None
depends_on = None

MERIT_NAME = "photref_merit"

MERIT_EXPRESSION = (
    "1.0 / ((1.0 - nanrank(s_center[0]))**2 + nanrank(bg_center[0])**2)"
)

MERIT_DESCRIPTION = (
    "How good a single photometric reference an image makes: higher for a "
    "larger source extraction S (sharper stars) and a lower background, each "
    "ranked among the images a reference is being chosen for."
)

#: Each map residual diagnostic: its description before and after the fix.
MAP_RESIDUAL_DESCRIPTIONS = {
    f"{param}_map_residual": (
        f"RMS difference between source extraction {param.upper()} "
        "and smoothed {param.upper()} map",
        f"RMS difference between source extraction {param.upper()} "
        f"and smoothed {param.upper()} map",
    )
    for param in ("s", "d", "k")
}


def _reflect(connection, table):
    """Return the named table as it is in the database."""

    return sqlalchemy.Table(
        table, sqlalchemy.MetaData(), autoload_with=connection
    )


def _set_descriptions(connection, which):
    """Give the map residuals their old (0) or new (1) descriptions.

    Only those still holding the other one: a project never holding the
    diagnostic, or holding a description of its own, is left as it is.
    """

    diagnostic_types = _reflect(connection, "diagnostic_type")
    for name, descriptions in MAP_RESIDUAL_DESCRIPTIONS.items():
        connection.execute(
            diagnostic_types.update()
            .where(
                diagnostic_types.c.name == name,
                diagnostic_types.c.description == descriptions[1 - which],
            )
            .values(description=descriptions[which])
        )


def upgrade():
    """Add the merit unless its name is taken, and fix the descriptions."""

    connection = alembic.op.get_bind()
    library = _reflect(connection, "diagnostic_expression")
    if (
        connection.execute(
            sqlalchemy.select(library.c.id).where(library.c.name == MERIT_NAME)
        ).first()
        is None
    ):
        connection.execute(
            library.insert().values(
                name=MERIT_NAME,
                expression=MERIT_EXPRESSION,
                description=MERIT_DESCRIPTION,
            )
        )
    _set_descriptions(connection, 1)


def downgrade():
    """Remove the merit unless changed, and restore the old descriptions.

    A changed merit is the user's own by then, and is kept. An unchanged one
    is removed even if other expressions read it, which then no longer
    resolve.
    """

    connection = alembic.op.get_bind()
    library = _reflect(connection, "diagnostic_expression")
    connection.execute(
        library.delete().where(
            library.c.name == MERIT_NAME,
            library.c.expression == MERIT_EXPRESSION,
        )
    )
    _set_descriptions(connection, 0)
