"""Store paths inside a project relative to its home, as ``{PROJHOME}/...``.

A project now holds no absolute path to anything inside it, so that moving or
copying its directory is all it takes to relocate it. This revision converts
what the database holds:

* the masters and raw images it records, where they lie inside the project;

* configuration values of path parameters: a relative value meant the
  directory the browser interface was started from, and is taken to be
  inside the project, as the defaults it replaces are (``MASTERS/...``); the
  log names named the home ``{project_home}``;

* the description of ``subpixmap``, whose default changed.

The files the project has produced hold such paths too. Rewriting them takes
far longer than opening a project should, so it is left to the product
migration, which this revision records as pending, one kind of product at a
time so that an interrupted migration resumes with the kind it was at.

The project home is given by ``migrate_project`` in
``config.attributes["project_home"]``. The parameter names and texts below
are a copy, made so that this revision keeps doing what it did if those
change.

There is no going back: the products are not converted back, so a database
downgraded past this revision would not match them.
"""

import logging
from pathlib import Path, PurePath

import alembic
import sqlalchemy

from autowisp.exceptions import DatabaseError

# revision identifiers, used by Alembic.
revision = "0018_projhome_paths"
down_revision = "0017_master_mask_description"
branch_labels = None
depends_on = None

_logger = logging.getLogger(__name__)

PROJHOME = "{PROJHOME}"

PENDING_TABLE = "pending_migration"

#: The kinds of product the product migration converts, each recorded as
#: pending until it is done.
PENDING_PRODUCTS = (
    "calibrated_images",
    "masters",
    "data_reduction",
    "lightcurves",
)

#: Parameters whose values are paths, by how their values are converted.
PATH_PARAMETERS = {
    **{
        name: "project"
        for name in (
            "calibrated-fname",
            "stacked-master-fname",
            "high-flat-master-fname",
            "low-flat-master-fname",
            "data-reduction-fname",
            "astrometry-catalog",
            "astrometry-catalog-source-list",
            "photometry-catalog",
            "photometry-catalog-source-list",
            "magfit-catalog",
            "magfit-catalog-source-list",
            "master-photref-fname-format",
            "magfit-stat-fname-format",
            "lc-catalog",
            "lc-catalog-source-list",
            "lightcurve-catalog-fname",
            "lc-fname",
            "detrending-catalog",
            "epd-statistics-fname",
            "tfa-statistics-fname",
        )
    },
    "logging-fname": "log",
    "std-out-err-fname": "log",
    "master-mask": "channel",
    "anet-indices": "external",
    "subpixmap": "external",
    "tempstore-dir": "external",
}

SUBPIXMAP_DESCRIPTION = {
    "old": (
        "The sub-pixel sensitivity map to assume. If not specified uniform "
        "sensitivy is assumed. This is especially important if processing "
        "images from color cameras. For a standard Bayer array, the built-in "
        "map called ``dslr_subpixmap.fits`` can be used."
    ),
    "new": (
        "The sub-pixel sensitivity map to assume. This is especially "
        "important if processing images from color cameras. If not "
        "specified, images that are one channel of a 2x2 pattern (a "
        "standard Bayer array) use the built-in map "
        "``dslr_subpixmap.fits``, and other images assume uniform "
        "sensitivity."
    ),
}


def _project_path(fname, project_home, *, must_exist):
    """
    Return the path in ``{PROJHOME}`` form, or None if it is not inside.

    The part after ``{PROJHOME}`` uses ``/`` on every system, as the stored
    form does everywhere else: stored paths are compared as strings.

    Args:
        fname(str):    The path as stored, possibly a template.

        project_home(str):    The absolute path of the project home.

        must_exist(bool):    Should a relative path be taken to be inside the
            project only if it names a file there?

    Returns:
        str or None:    ``{PROJHOME}/<relative>``, or None for a path outside
        the project or a relative one naming no file there.
    """

    if fname.startswith(PROJHOME):
        return fname
    path = PurePath(fname)
    # On Windows, a path is absolute only with a drive: without one it is
    # relative to the current drive.
    if not path.is_absolute():
        if must_exist and not (Path(project_home) / path).exists():
            return None
        relative = path
    elif path.is_relative_to(project_home):
        relative = path.relative_to(project_home)
    else:
        return None
    return PROJHOME + "/" + relative.as_posix()


def _convert_files(connection, table_name, column, project_home):
    """Convert the paths to files inside the project held by a column."""

    table = sqlalchemy.Table(
        table_name, sqlalchemy.MetaData(), autoload_with=connection
    )
    for row_id, fname in connection.execute(
        sqlalchemy.select(table.c.id, table.c[column])
    ).all():
        converted = _project_path(fname, project_home, must_exist=True)
        if converted is None:
            if not PurePath(fname).is_absolute():
                _logger.warning(
                    "%s.%s of row %d is the relative path %r, which names "
                    "no file in the project home %r; left as it is.",
                    table_name,
                    column,
                    row_id,
                    fname,
                    project_home,
                )
        elif converted != fname:
            connection.execute(
                table.update()
                .where(table.c.id == row_id)
                .values({column: converted})
            )


def _convert_path_entry(entry, kind, project_home):
    """Return one path of a configured value in ``{PROJHOME}`` form."""

    if kind == "channel":
        if ":" not in entry:
            _logger.warning(
                "%r is not of the form <channel>:<path>; left as it is.",
                entry,
            )
            return entry
        channel, entry = entry.split(":", 1)
        return (
            channel + ":" + _convert_path_entry(entry, "project", project_home)
        )
    if kind == "external":
        if not PurePath(entry).is_absolute():
            _logger.warning(
                "%r is a relative path, which must now be absolute; left as "
                "it is.",
                entry,
            )
        return entry
    if kind == "log":
        entry = entry.replace("{project_home}", PROJHOME)
    return _project_path(entry, project_home, must_exist=False) or entry


def _convert_value(value, kind, project_home):
    """Return a configured value of a path parameter in ``{PROJHOME}`` form."""

    if value.startswith("[") and value.endswith("]"):
        return (
            "["
            + ", ".join(
                _convert_path_entry(entry.strip(), kind, project_home)
                for entry in value[1:-1].split(",")
            )
            + "]"
        )
    return _convert_path_entry(value, kind, project_home)


def _convert_configuration(connection, project_home):
    """Convert the configured values of path parameters."""

    metadata = sqlalchemy.MetaData()
    parameter = sqlalchemy.Table(
        "parameter", metadata, autoload_with=connection
    )
    configuration = sqlalchemy.Table(
        "configuration", metadata, autoload_with=connection
    )
    for row_id, name, value in connection.execute(
        sqlalchemy.select(
            configuration.c.id, parameter.c.name, configuration.c.value
        )
        .join(parameter, configuration.c.parameter_id == parameter.c.id)
        .where(parameter.c.name.in_(PATH_PARAMETERS))
    ).all():
        if not value:
            continue
        converted = _convert_value(value, PATH_PARAMETERS[name], project_home)
        if converted != value:
            connection.execute(
                configuration.update()
                .where(configuration.c.id == row_id)
                .values(value=converted)
            )


def _record_pending_products(connection):
    """Create the table of pending migrations if needed, and add products."""

    if not sqlalchemy.inspect(connection).has_table(PENDING_TABLE):
        # Its timestamp trigger is installed by migrate_project, once the
        # revisions are done.
        alembic.op.create_table(
            PENDING_TABLE,
            sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column(
                "timestamp",
                sqlalchemy.TIMESTAMP,
                nullable=False,
                server_default=sqlalchemy.text("CURRENT_TIMESTAMP"),
            ),
            sqlalchemy.Column(
                "name", sqlalchemy.String(100), nullable=False, unique=True
            ),
        )
    pending = sqlalchemy.Table(
        PENDING_TABLE, sqlalchemy.MetaData(), autoload_with=connection
    )
    recorded = set(
        connection.execute(sqlalchemy.select(pending.c.name)).scalars()
    )
    for name in PENDING_PRODUCTS:
        if name not in recorded:
            connection.execute(pending.insert().values(name=name))


def upgrade():
    """Convert the paths the database holds, and record the products."""

    connection = alembic.op.get_bind()
    project_home = alembic.context.config.attributes["project_home"]

    _record_pending_products(connection)
    _convert_files(connection, "master_file", "filename", project_home)
    _convert_files(connection, "image", "raw_fname", project_home)
    _convert_configuration(connection, project_home)

    parameter = sqlalchemy.Table(
        "parameter", sqlalchemy.MetaData(), autoload_with=connection
    )
    connection.execute(
        parameter.update()
        .where(
            parameter.c.name == "subpixmap",
            parameter.c.description == SUBPIXMAP_DESCRIPTION["old"],
        )
        .values(description=SUBPIXMAP_DESCRIPTION["new"])
    )


def downgrade():
    """Refuse: the products are not converted back."""

    raise DatabaseError(
        "A project cannot be taken back past AutoWISP 3.0: its products store "
        "paths as {PROJHOME}/..., and are not converted back, so the database "
        "would no longer match them. Restore the copy of the database taken "
        "before migrating, and the products from a backup, instead."
    )
