"""Copy a project's database into another, on either backend.

What creating an experiment, resetting its processing and merging it do to
the database, and what adding an existing server project does when it clones
the database (SUP-572). Either side may be SQLite or a server: an experiment
is always SQLite, whatever its original's backend.
"""

from sqlalchemy import (
    MetaData,
    String,
    Table,
    delete,
    inspect as sa_inspect,
    select,
)

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.migrate import (
    check_project_schema,
    create_project_schema,
)
from autowisp.database.project_rows import begin_transaction, read_project
from autowisp.exceptions import DatabaseError


def _get_kept_row_test(table, is_kept, kept_keys):
    """
    Return a test for the rows of a table that the target keeps, or None.

    Replacing a project's rows can keep some of the target's, e.g. a merge
    keeps the original's runs of operations on experiments. Each such row
    stays in the target, and the source's copy of it, if any, is skipped. So
    one test serves both databases: applied to the target's rows it tells
    which to keep, applied to the source's which to skip.

    A row is kept if ``is_kept`` says so, or if it refers through a foreign
    key to a kept row of another table, so that e.g. the errors recorded
    against a kept run are kept with it.

    Args:
        table:    The table whose rows are tested.

        is_kept:    Callable given a row, as a mapping from column name to
            value, returning whether it is kept; or None, if the table's rows
            are kept only for referring to kept rows.

        kept_keys(dict):    By table name, the primary keys of the rows found
            kept so far, in the same database. Tables are tested in
            dependency order, so those this one refers to are complete.

    Returns:
        callable or None:    Given a row, whether it is kept; None if no row
        of the table can be.
    """

    references = [
        (foreign_key.parent.name, kept_keys[foreign_key.column.table.name])
        for foreign_key in table.foreign_keys
        if kept_keys.get(foreign_key.column.table.name)
    ]
    if is_kept is None and not references:
        return None

    def test_row(row):
        return (is_kept is not None and is_kept(row)) or any(
            row[column] in keys for column, keys in references
        )

    return test_row


def _get_key_column(table):
    """Return the name of a table's primary key, by which kept rows are told."""

    key_columns = list(table.primary_key.columns)
    if len(key_columns) != 1:
        raise ValueError(
            f"Rows of {table.name} cannot be kept: it has no single-column "
            "primary key to tell them by."
        )
    return key_columns[0].name


# A table consumer of read_project: called with each table, and nothing more.
# pylint: disable-next=too-few-public-methods
class _TargetWriter:
    """Insert the rows of each table copied into the target, as they are read.

    Rows are inserted by column name, ids and timestamps included, in the
    order :func:`read_project` reads the tables, which puts every row after
    those it refers to.
    """

    def __init__(self, connection, uncopied_tables, kept_rows):
        """
        Write into the given connection, inside its transaction.

        Args:
            connection:    A connection to the target database, whose
                transaction commits or rolls back the whole copy.

            uncopied_tables(set):    The names of the tables none of whose
                rows are written.

            kept_rows(dict):    By table name, what tells the rows the target
                keeps, whose copies in the source are skipped (see
                :func:`_get_kept_row_test`).
        """

        self._connection = connection
        self._uncopied_tables = uncopied_tables
        self._kept_rows = kept_rows
        # By table name, the keys of the source's rows skipped so far.
        self._skipped_keys = {}
        # SQLite does not enforce the length of a string column; a server
        # would refuse the value, or silently cut it short.
        self._check_lengths = connection.dialect.name != "sqlite"

    def __call__(self, table):
        """Start copying the table; return what to give its rows to."""

        if table.name in self._uncopied_tables:
            return None
        target_table = DataModelBase.metadata.tables[table.name]
        is_kept = _get_kept_row_test(
            target_table, self._kept_rows.get(table.name), self._skipped_keys
        )
        key_column = skipped_keys = None
        if is_kept is not None:
            key_column = _get_key_column(target_table)
            skipped_keys = self._skipped_keys.setdefault(table.name, set())
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
            rows = []
            for row in batch:
                if is_kept is not None and is_kept(row):
                    skipped_keys.add(row[key_column])
                else:
                    rows.append(dict(row))
            if not rows:
                return
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


def _find_kept_keys(connection, uncopied_tables, kept_rows):
    """
    Return the keys of the rows a replacing copy keeps in the target.

    Args:
        connection:    A connection to the target, in the copy's transaction.

        uncopied_tables(set):    The names of the tables kept whole, which
            are not looked at.

        kept_rows(dict):    As for :func:`copy_project_database`.

    Returns:
        dict:    By table name, the primary keys of the rows kept, for every
        table whose rows may be.
    """

    kept_keys = {}
    for table in DataModelBase.metadata.sorted_tables:
        if table.name in uncopied_tables:
            continue
        is_kept = _get_kept_row_test(
            table, kept_rows.get(table.name), kept_keys
        )
        if is_kept is None:
            continue
        key_column = _get_key_column(table)
        kept_keys[table.name] = {
            row[key_column]
            for row in connection.execute(select(table)).mappings()
            if is_kept(row)
        }
    return kept_keys


def copy_project_database(  # pylint: disable=too-many-arguments
    source_engine,
    target_engine,
    *,
    replace=False,
    kept_tables=(),
    kept_rows=None,
    table_consumers=(),
):
    """
    Copy every row of a project's database into another.

    The source is read once, within one transaction, by
    :func:`~autowisp.database.project_rows.read_project`; the target is
    written within one transaction, so a copy that fails leaves it as it was.
    No copy includes the ``experiment`` table: it is the original's record of
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
            included, empty. An SQLite target being replaced checks its
            foreign keys at the commit, so that rows kept may refer to rows
            deleted and inserted again.

        kept_tables:    When replacing, the names of further tables the
            target keeps whole, getting none of the source's rows, e.g. an
            experiment's configuration when its processing is reset. On a
            server, whose foreign keys are checked at once rather than at the
            commit, a kept table may refer only to rows that stay, as the
            ``experiment`` table refers to runs kept by ``kept_rows``;
            otherwise the copy fails, and its transaction rolls back.

        kept_rows(dict):    When replacing, by table name, a callable given
            each row, as a mapping from column name to value, returning
            whether the target keeps it: it stays in the target, and the
            source's copy of it is skipped. A row referring to a kept row
            through a foreign key is kept with it, in either database. The
            tables involved need single-column primary keys.

        table_consumers:    Further consumers of the source's rows (see
            :func:`~autowisp.database.project_rows.read_project`), given them
            in the same pass, e.g. one fingerprinting the source.

    Returns:
        None

    Raises:
        DatabaseError:    If either side is not fully migrated, a new target
            already holds AutoWISP tables, or a string is too long for its
            column in a server target.

        ValueError:    If tables or rows are to be kept without replacing.
    """

    if not replace and (kept_tables or kept_rows):
        raise ValueError("Only a copy replacing rows can keep any.")
    uncopied_tables = {
        # The original's record of its experiments, which no copy carries.
        "experiment",
        # The revision, which the target records when its schema is created.
        "alembic_version",
        *kept_tables,
    }
    kept_rows = kept_rows or {}
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
                if target.dialect.name == "sqlite":
                    # Lasts until the transaction ends, so it must be in one.
                    begin_transaction(target)
                    target.exec_driver_sql("PRAGMA defer_foreign_keys = ON")
                # Found before anything is deleted: a row is kept for
                # referring to a kept row.
                kept_keys = _find_kept_keys(target, uncopied_tables, kept_rows)
                for table in reversed(DataModelBase.metadata.sorted_tables):
                    if table.name in uncopied_tables:
                        continue
                    statement = delete(table)
                    if kept_keys.get(table.name):
                        statement = statement.where(
                            table.c[_get_key_column(table)].not_in(
                                kept_keys[table.name]
                            )
                        )
                    target.execute(statement)
            read_project(
                source,
                [
                    _TargetWriter(target, uncopied_tables, kept_rows),
                    *table_consumers,
                ],
            )
    except BaseException:
        if not replace:
            # Its rows rolled back, but its schema was committed before the
            # copy started: leave it empty, as it was found.
            _drop_project_schema(target_engine)
        raise
