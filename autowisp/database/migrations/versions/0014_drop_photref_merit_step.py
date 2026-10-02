"""Remove the ``calculate_photref_merit`` step from existing projects.

The step was registered when a project was created but never part of the
processing sequence, so the pipeline never ran it, and its module has since
been deleted. New projects no longer get it. Projects created earlier kept
its row, with its dependencies and parameters, which nothing looks up: it
did no harm, but it left them holding definitions a new project does not
have. This brings them in line.

``merit-function`` goes with it, including whatever value a project set for
it: no other step takes that parameter, and nothing reads it any more.

This revision changes rows, not the schema. What the downgrade restores is
a copy of what project creation used to store, since the code that produced
it is gone.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0014_drop_photref_merit_step"
down_revision = "0013_exclusion_rule_parameters"
branch_labels = None
depends_on = None

STEP_NAME = "calculate_photref_merit"

STEP_DESCRIPTION = (
    "Sort DR files by their single photometric reference merit function."
)

#: The image type the step applied to, and its prerequisites were for.
IMAGE_TYPE = "object"

#: The steps that had to complete, on all images, before the step could run.
PREREQUISITES = (
    "calibrate",
    "find_stars",
    "solve_astrometry",
    "fit_star_shape",
    "fit_source_extracted_psf_map",
)

#: The only parameter no other step takes.
MERIT_PARAMETER = "merit-function"

MERIT_DESCRIPTION = (
    "The merit function to use. High values should indicate a better "
    "candidate to serve as photometric reference. The function may use any "
    "PSF parameter as well as the following frame properties:\n"
    "\tz: the zenith distance of the frame center\n"
    "\tbg: the background level at the center of the frame\n"
    "In addition, for each property the standard deviation over all frames "
    "can also be used as ``std_<param>`` (e.g. ``std_z``) as well as its "
    "quantile as ``qnt_<param>`` (e.g. ``qnt_bg`` for the background "
    "quantile)."
)

MERIT_DEFAULT = "1.0 / ((1.0 - qnt_s)**2 + qnt_bg**2)"

#: The id project creation gives the condition that always applies.
DEFAULT_CONDITION_ID = 1

#: The parameters the step shared with other steps.
SHARED_PARAMETERS = (
    "bg-map-error-avg",
    "bg-map-fit-terms-expression",
    "bg-map-max-rej-iter",
    "bg-map-rej-level",
    "fname-datetime-format",
    "iers-auto-max-age",
    "logging-datetime-format",
    "logging-fname",
    "logging-message-format",
    "observatory-location",
    "std-out-err-fname",
    "verbose",
)


def _reflect(connection, table):
    """Return the named table as it is in the database."""

    return sqlalchemy.Table(
        table, sqlalchemy.MetaData(), autoload_with=connection
    )


def _get_id(connection, table, name):
    """Return the id of the row of the given name, or None."""

    return connection.scalar(
        sqlalchemy.select(table.c.id).where(table.c.name == name)
    )


def upgrade():
    """Remove the step and what belongs only to it, if the step is there."""

    connection = alembic.op.get_bind()
    steps = _reflect(connection, "step")
    step_id = _get_id(connection, steps, STEP_NAME)
    if step_id is None:
        return

    parameters = _reflect(connection, "parameter")
    links = _reflect(connection, "step_parameters")
    configuration = _reflect(connection, "configuration")
    dependencies = _reflect(connection, "step_dependencies")

    connection.execute(
        dependencies.delete().where(
            sqlalchemy.or_(
                dependencies.c.blocked_step_id == step_id,
                dependencies.c.blocking_step_id == step_id,
            )
        )
    )
    connection.execute(links.delete().where(links.c.step_id == step_id))

    merit_id = _get_id(connection, parameters, MERIT_PARAMETER)
    if (
        merit_id is not None
        and connection.execute(
            sqlalchemy.select(links.c.step_id).where(
                links.c.param_id == merit_id
            )
        ).first()
        is None
    ):
        connection.execute(
            configuration.delete().where(
                configuration.c.parameter_id == merit_id
            )
        )
        connection.execute(
            parameters.delete().where(parameters.c.id == merit_id)
        )

    connection.execute(steps.delete().where(steps.c.id == step_id))


def downgrade():
    """Put the step back as project creation used to define it.

    The code this returns to still expects it. ``merit-function`` comes
    back with its default: a value a project had set was removed with it.
    A database without the pipeline's steps has nothing to put it back
    into.
    """

    connection = alembic.op.get_bind()
    steps = _reflect(connection, "step")
    if (
        _get_id(connection, steps, STEP_NAME) is not None
        or _get_id(connection, steps, PREREQUISITES[0]) is None
    ):
        return

    parameters = _reflect(connection, "parameter")
    links = _reflect(connection, "step_parameters")
    configuration = _reflect(connection, "configuration")
    dependencies = _reflect(connection, "step_dependencies")

    step_id = connection.execute(
        steps.insert().values(name=STEP_NAME, description=STEP_DESCRIPTION)
    ).inserted_primary_key[0]

    merit_id = _get_id(connection, parameters, MERIT_PARAMETER)
    if merit_id is None:
        merit_id = connection.execute(
            parameters.insert().values(
                name=MERIT_PARAMETER, description=MERIT_DESCRIPTION
            )
        ).inserted_primary_key[0]
        connection.execute(
            configuration.insert().values(
                parameter_id=merit_id,
                version=0,
                condition_id=DEFAULT_CONDITION_ID,
                value=MERIT_DEFAULT,
            )
        )

    for parameter_id in [merit_id] + [
        _get_id(connection, parameters, name) for name in SHARED_PARAMETERS
    ]:
        if parameter_id is not None:
            connection.execute(
                links.insert().values(step_id=step_id, param_id=parameter_id)
            )

    image_type_id = _get_id(
        connection, _reflect(connection, "image_type"), IMAGE_TYPE
    )
    for prerequisite in PREREQUISITES:
        prerequisite_id = _get_id(connection, steps, prerequisite)
        if prerequisite_id is not None and image_type_id is not None:
            connection.execute(
                dependencies.insert().values(
                    blocked_step_id=step_id,
                    blocked_image_type_id=image_type_id,
                    blocking_step_id=prerequisite_id,
                    blocking_image_type_id=image_type_id,
                    allow_pending=False,
                )
            )
