"""Copy a project's database into another, on either backend.

What creating an experiment, resetting its processing and merging it do to
the database, and what adding an existing server project does when it clones
the database (SUP-572). Either side may be SQLite or a server: an experiment
is always SQLite, whatever its original's backend.
"""

from sqlalchemy import MetaData, String, Table, delete, inspect as sa_inspect

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.migrate import (
    check_project_schema,
    create_project_schema,
)
from autowisp.database.project_rows import read_project
from autowisp.exceptions import DatabaseError

# The original's record of its experiments, which no copy carries, and the
# revision, which the target records when its schema is created.
_UNCOPIED_TABLES = {  # pylint: disable=invalid-name
    "experiment",
    "alembic_version",
}


# A table consumer of read_project: called with each table, and nothing more.
# pylint: disable-next=too-few-public-methods
class _TargetWriter:
    """Insert the rows of each table copied into the target, as they are read.

    Rows are inserted by column name, ids and timestamps included, in the
    order :func:`read_project` reads the tables, which puts every row after
    those it refers to.
    """

    def __init__(self, connection):
        """
        Write into the given connection, inside its transaction.

        Args:
            connection:    A connection to the target database, whose
                transaction commits or rolls back the whole copy.
        """

        self._connection = connection
        # SQLite does not enforce the length of a string column; a server
        # would refuse the value, or silently cut it short.
        self._check_lengths = connection.dialect.name != "sqlite"

    def __call__(self, table):
        """Start copying the table; return what to give its rows to."""

        if table.name in _UNCOPIED_TABLES:
            return None
        target_table = DataModelBase.metadata.tables[table.name]
        column_lengths = (
            {
                column.name: column.type.length
                for column in target_table.columns
                if isinstance(column.type, String) and column.type.length
            }
            if self._check_lengths
            else {}
        )
        key_columns = [
            column.name for column in target_table.primary_key.columns
        ]

        def write_rows(batch):
            rows = [dict(row) for row in batch]
            for row in rows:
                for column, length in column_lengths.items():
                    if row[column] is not None and len(row[column]) > length:
                        key = {name: row[name] for name in key_columns}
                        raise DatabaseError(
                            f"The {column!r} of {table.name} row {key} is "
                            f"{len(row[column])} characters long, more than "
                            f"the {length} the target database takes. "
                            "Shorten it in the project being copied, then "
                            "copy again.",
                            details={
                                "table": table.name,
                                "key": key,
                                "column": column,
                                "length": len(row[column]),
                                "limit": length,
                            },
                        )
            self._connection.execute(target_table.insert(), rows)

        return write_rows


def _drop_project_schema(engine):
    """Drop every table a project's schema has, revision included."""

    DataModelBase.metadata.drop_all(engine)
    Table("alembic_version", MetaData()).drop(engine, checkfirst=True)


def copy_project_database(
    source_engine, target_engine, *, replace=False, table_consumers=()
):
    """
    Copy every row of a project's database into another.

    The source is read once, within one transaction, by
    :func:`~autowisp.database.project_rows.read_project`; the target is
    written within one transaction, so a copy that fails leaves it as it was.
    Neither copies the ``experiment`` table: it is the original's record of
    its experiments.

    Both engines should come from
    :func:`~autowisp.database.interface.create_project_engine`, which makes a
    server present timestamps in UTC as SQLite does.

    Args:
        source_engine:    The engine of the project database to copy.

        target_engine:    The engine of the database to copy into.

        replace(bool):    Whether to replace the contents of a project
            database, rather than fill an empty one. When replacing, the
            target's rows are deleted (with ``DELETE``, which a server does
            not commit implicitly) and the source's inserted, its
            ``experiment`` table left as it is. Otherwise the target must
            hold no AutoWISP tables; it gets every one, ``experiment``
            included, empty.

        table_consumers:    Further consumers of the source's rows (see
            :func:`~autowisp.database.project_rows.read_project`), given them
            in the same pass, e.g. one fingerprinting the source.

    Returns:
        None

    Raises:
        DatabaseError:    If either side is not fully migrated, a new target
            already holds AutoWISP tables, or a string is too long for its
            column in a server target.
    """

    check_project_schema(source_engine)
    if replace:
        check_project_schema(target_engine)
    else:
        present = set(DataModelBase.metadata.tables) & set(
            sa_inspect(target_engine).get_table_names()
        )
        if present:
            raise DatabaseError(
                "Refusing to copy a project into "
                f"{target_engine.url.render_as_string(hide_password=True)}: "
                f"it already holds {len(present)} AutoWISP table(s), "
                f"including {', '.join(sorted(present)[:5])}, which copying "
                "would mix with the project's. Copy into an empty database."
            )
        create_project_schema(target_engine)

    try:
        with source_engine.connect() as source, target_engine.begin() as target:
            if replace:
                for table in reversed(DataModelBase.metadata.sorted_tables):
                    if table.name not in _UNCOPIED_TABLES:
                        target.execute(delete(table))
            read_project(source, [_TargetWriter(target), *table_consumers])
    except BaseException:
        if not replace:
            # Its rows rolled back, but its schema was committed before the
            # copy started: leave it empty, as it was found.
            _drop_project_schema(target_engine)
        raise
