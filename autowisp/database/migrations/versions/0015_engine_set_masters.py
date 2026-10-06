"""Stop storing the masters the engine sets, and selecting the master photref.

Project creation used to list ``master_photref`` as an optional input master
of ``fit_magnitudes`` on object images, under ``master-photref-dr-fname``, an
option the step does not have. The master the engine selected was therefore
never handed to the step, which rebuilt the master every time. Worse, being
an input, it was selected for every object image by ``TARGETID``,
``CLRCHNL`` and ``EXPTIME`` alone, so two masters of one target, channel and
exposure time made every later evaluation of a matching image fail: masters
of this type have no expression saying which of several to prefer. The
engine now hands the step the master built from the batch's own single
photometric reference, found through the progress that built it, and new
projects no longer list the input. This removes it from existing ones.

Project creation also stored the options through which the engine hands each
step its masters, as parameters anyone could configure: ``master-bias``,
``master-dark``, ``master-flat``, ``single-photref-dr-fname`` and
``master-photref-fname``. A configured value never had any effect: the engine
replaces it with the master it selects, and leaves out images it has no
master for. New projects no longer store them, so they cannot be configured
in the first place. This removes them, with any values set for them, from
existing projects.

This revision changes rows, not the schema. The downgrade restores a copy of
the rows project creation used to write.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0015_engine_set_masters"
down_revision = "0014_drop_photref_merit_step"
branch_labels = None
depends_on = None

#: The input master no longer selected: step, image type, master type.
INPUT = ("fit_magnitudes", "object", "master_photref")

#: The option name the input carried, which the step never had.
INPUT_CONFIG_NAME = "master-photref-dr-fname"

#: The id project creation gives the condition that always applies.
DEFAULT_CONDITION_ID = 1

#: Each removed parameter: its description, default and the steps taking it.
PARAMETERS = {
    **{
        f"master-{master}": (
            f"The master {master} to apply. No {master} correction is applied "
            "of not specified. Each master filename should be preceeded by "
            "``<channel name>:`` identifying which channel it applies to. All "
            "channels must have a masters specified and no channel should "
            "have multpiple.",
            None,
            ("calibrate",),
        )
        for master in ("bias", "dark", "flat")
    },
    "single-photref-dr-fname": (
        "The name of the data reduction file of the single photometric "
        "reference to use or used to start the magnitude fitting iterations.",
        "single_photref.hdf5.0",
        (
            "fit_magnitudes",
            "create_lightcurves",
            "epd",
            "generate_epd_statistics",
            "tfa",
            "generate_tfa_statistics",
        ),
    ),
    "master-photref-fname": (
        "The name of a master photometric reference to use. If specified, the "
        "sintgle reference is ignored and magnitude fitting proceeds without "
        "any iterations.",
        None,
        ("fit_magnitudes",),
    ),
}


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


def _get_input_ids(connection):
    """Return the step, image type and master type ids, any of them None."""

    return tuple(
        _get_id(connection, _reflect(connection, table), name)
        for table, name in zip(("step", "image_type", "master_type"), INPUT)
    )


def _where_input(inputs, step_id, image_type_id, master_type_id):
    """Return the condition selecting the input's row."""

    return sqlalchemy.and_(
        inputs.c.step_id == step_id,
        inputs.c.image_type_id == image_type_id,
        inputs.c.master_type_id == master_type_id,
    )


def _remove_input(connection):
    """Delete the input, if it is there."""

    ids = _get_input_ids(connection)
    if None in ids:
        return
    inputs = _reflect(connection, "input_master_types")
    connection.execute(inputs.delete().where(_where_input(inputs, *ids)))


def _remove_parameters(connection):
    """Delete the parameters that are there, and every row referring to them."""

    parameters = _reflect(connection, "parameter")
    for name in PARAMETERS:
        parameter_id = _get_id(connection, parameters, name)
        if parameter_id is None:
            continue
        for table in (
            "configuration",
            "step_parameters",
            "alternate_parameter_names",
        ):
            referring = _reflect(connection, table)
            column = (
                referring.c.parameter_id
                if table == "configuration"
                else referring.c.param_id
            )
            connection.execute(referring.delete().where(column == parameter_id))
        connection.execute(
            parameters.delete().where(parameters.c.id == parameter_id)
        )


def _restore_input(connection):
    """Put the input back, unless it is there or has nothing to refer to."""

    ids = _get_input_ids(connection)
    if None in ids:
        return
    inputs = _reflect(connection, "input_master_types")
    if (
        connection.execute(
            sqlalchemy.select(inputs.c.id).where(_where_input(inputs, *ids))
        ).first()
        is not None
    ):
        return
    step_id, image_type_id, master_type_id = ids
    connection.execute(
        inputs.insert().values(
            step_id=step_id,
            image_type_id=image_type_id,
            master_type_id=master_type_id,
            optional=True,
            config_name=INPUT_CONFIG_NAME,
        )
    )


def _restore_parameters(connection):
    """Put back the parameters that are missing, with their defaults.

    Each is linked to those of its steps the database has; a database with
    none of them has nothing to put it back into.
    """

    steps = _reflect(connection, "step")
    parameters = _reflect(connection, "parameter")
    configuration = _reflect(connection, "configuration")
    links = _reflect(connection, "step_parameters")
    for name, (description, default, step_names) in PARAMETERS.items():
        step_ids = [
            step_id
            for step_id in (
                _get_id(connection, steps, step_name)
                for step_name in step_names
            )
            if step_id is not None
        ]
        if not step_ids or _get_id(connection, parameters, name) is not None:
            continue
        parameter_id = connection.execute(
            parameters.insert().values(name=name, description=description)
        ).inserted_primary_key[0]
        connection.execute(
            configuration.insert().values(
                parameter_id=parameter_id,
                version=0,
                condition_id=DEFAULT_CONDITION_ID,
                value=default,
            )
        )
        for step_id in step_ids:
            connection.execute(
                links.insert().values(step_id=step_id, param_id=parameter_id)
            )


def upgrade():
    """Delete the input and the parameters, whichever are there."""

    connection = alembic.op.get_bind()
    _remove_input(connection)
    _remove_parameters(connection)


def downgrade():
    """Put the input and the parameters back as project creation defined them.

    The parameters come back with their defaults: values a project had set
    were removed with them.
    """

    connection = alembic.op.get_bind()
    _restore_input(connection)
    _restore_parameters(connection)
