"""Add the exclusion rules to the configuration, and share the observation id.

``fit_magnitudes``, ``epd`` and ``tfa`` each have an exclusion rule: a
configuration parameter saying which images or observations to leave out of
the fit. Projects created before the rules existed get them here, unset, so
that nothing is excluded until a rule is deliberately set, and so that the
rules can be set on the configuration page and vary through conditions like
any other parameter.

EPD also starts identifying observations the way TFA does, by
``tfa-observation-id``. The existing parameter is linked to ``epd`` rather
than a second one being added, so that both steps read the one value the
project already stores; otherwise EPD would use the command-line default
while TFA used the stored value, and the two would disagree about which
observation is which. Its description, which the configuration page shows,
is brought up to date with that; the value the project stores is not
touched.

This revision changes rows, not the schema. The names and help texts below
are a copy of what the steps' command-line parsers give a new project, made
so that this revision keeps doing what it did if they change.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0013_exclusion_rule_parameters"
down_revision = "0012_qc_included_datasets"
branch_labels = None
depends_on = None

#: The id project creation gives the condition that always applies.
DEFAULT_CONDITION_ID = 1

OBSERVATION_ID_PARAMETER = "tfa-observation-id"

#: What the observation id is for now that EPD and exclusion lists use it.
OBSERVATION_ID_DESCRIPTION = (
    "The datasets whose values identify an observation, used to match "
    "observations across light curves and to list them in --qc-exclude-file. "
    "Shared by EPD and TFA. The default suits a single camera; for example, "
    "the following works for HAT: fitsheader.cfg.stid fitsheader.cfg.cmpos "
    "fitsheader.fnum."
)

#: What it was described as while only TFA used it.
OLD_OBSERVATION_ID_DESCRIPTION = (
    "The datasets to use for matching observations across light curves. For "
    "example, the following works for HAT: fitseader.cfg.stid "
    "fitsheader.cfg.cmpos fitsheader.fnum."
)

#: For each step, the name and description of its exclusion rule.
EXCLUSION_RULES = {
    "fit_magnitudes": (
        "magfit-exclusion-rule",
        "A boolean expression over the image diagnostics and the project's "
        "diagnostic expressions, true for the images to leave out of the "
        "master photometric reference, e.g. ``(cloud[0] > 0.3) | "
        "(srcextract_mag_zeropt['G0'] < 19.5)``. A slot subscript stands for "
        "the channel being decided for, a quoted channel name for that "
        "channel. The pipeline evaluates it to produce the exclusion list. "
        "Excluded images are still fit. If unset, nothing is excluded.",
    ),
    "epd": (
        "epd-exclusion-rule",
        "A boolean expression over the image diagnostics and the project's "
        "diagnostic expressions, true for the observations to leave out of "
        "the EPD fit, e.g. ``(cloud[0] > 0.3) | (srcextract_mag_zeropt['G0'] "
        "< 19.5)``. A slot subscript stands for the channel being decided "
        "for, a quoted channel name for that channel. The pipeline evaluates "
        "it to produce the exclusion list. Excluded observations are still "
        "corrected. If unset, nothing is excluded.",
    ),
    "tfa": (
        "tfa-exclusion-rule",
        "A boolean expression over the image diagnostics and the project's "
        "diagnostic expressions, true for the observations to leave out of "
        "the TFA fit, e.g. ``(cloud[0] > 0.3) | (srcextract_mag_zeropt['G0'] "
        "< 19.5)``. A slot subscript stands for the channel being decided "
        "for, a quoted channel name for that channel. The pipeline evaluates "
        "it to produce the exclusion list. Excluded observations are still "
        "corrected. If unset, nothing is excluded.",
    ),
}


def _reflect(connection, table):
    """Return the named table as it is in the database."""

    return sqlalchemy.Table(
        table, sqlalchemy.MetaData(), autoload_with=connection
    )


def _get_id(connection, table, name):
    """Return the id of the step or parameter of the given name, or None."""

    return connection.scalar(
        sqlalchemy.select(table.c.id).where(table.c.name == name)
    )


def _is_linked(connection, links, step_id, parameter_id):
    """Whether the step already takes the parameter."""

    return (
        connection.execute(
            sqlalchemy.select(links.c.step_id).where(
                links.c.step_id == step_id, links.c.param_id == parameter_id
            )
        ).first()
        is not None
    )


def upgrade():
    """Add whichever of the parameters, links and defaults are missing.

    Any of them can be there without this revision having run: after an
    interrupted upgrade, the next attempt finds what the first one added.
    A database without the steps holds no pipeline configuration to extend.
    """

    connection = alembic.op.get_bind()
    steps = _reflect(connection, "step")
    parameters = _reflect(connection, "parameter")
    links = _reflect(connection, "step_parameters")
    configuration = _reflect(connection, "configuration")

    for step_name, (rule_name, description) in EXCLUSION_RULES.items():
        step_id = _get_id(connection, steps, step_name)
        if step_id is None:
            continue

        rule_id = _get_id(connection, parameters, rule_name)
        if rule_id is None:
            rule_id = connection.execute(
                parameters.insert().values(
                    name=rule_name, description=description
                )
            ).inserted_primary_key[0]
        if not _is_linked(connection, links, step_id, rule_id):
            connection.execute(
                links.insert().values(step_id=step_id, param_id=rule_id)
            )
        if (
            connection.execute(
                sqlalchemy.select(configuration.c.id).where(
                    configuration.c.parameter_id == rule_id
                )
            ).first()
            is None
        ):
            connection.execute(
                configuration.insert().values(
                    parameter_id=rule_id,
                    version=0,
                    condition_id=DEFAULT_CONDITION_ID,
                    value=None,
                )
            )

    observation_id = _get_id(connection, parameters, OBSERVATION_ID_PARAMETER)
    if observation_id is None:
        return
    connection.execute(
        parameters.update()
        .where(parameters.c.id == observation_id)
        .values(description=OBSERVATION_ID_DESCRIPTION)
    )
    epd_id = _get_id(connection, steps, "epd")
    if epd_id is not None and not _is_linked(
        connection, links, epd_id, observation_id
    ):
        connection.execute(
            links.insert().values(step_id=epd_id, param_id=observation_id)
        )


def downgrade():
    """Remove the rules, with every value set for them, and EPD's link.

    That includes rules a user set and the conditions under which each
    applies: without the parameters there is nothing for them to belong to.
    """

    connection = alembic.op.get_bind()
    steps = _reflect(connection, "step")
    parameters = _reflect(connection, "parameter")
    links = _reflect(connection, "step_parameters")
    configuration = _reflect(connection, "configuration")

    for rule_name, _ in EXCLUSION_RULES.values():
        rule_id = _get_id(connection, parameters, rule_name)
        if rule_id is None:
            continue
        connection.execute(
            configuration.delete().where(
                configuration.c.parameter_id == rule_id
            )
        )
        connection.execute(links.delete().where(links.c.param_id == rule_id))
        connection.execute(
            parameters.delete().where(parameters.c.id == rule_id)
        )

    observation_id = _get_id(connection, parameters, OBSERVATION_ID_PARAMETER)
    if observation_id is None:
        return
    connection.execute(
        parameters.update()
        .where(parameters.c.id == observation_id)
        .values(description=OLD_OBSERVATION_ID_DESCRIPTION)
    )
    epd_id = _get_id(connection, steps, "epd")
    if epd_id is not None:
        connection.execute(
            links.delete().where(
                links.c.step_id == epd_id, links.c.param_id == observation_id
            )
        )
