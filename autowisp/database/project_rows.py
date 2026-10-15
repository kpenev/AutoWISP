"""Read every row of a project's database, for whatever consumes them.

The one reader of a whole project: fingerprinting an original
(:mod:`autowisp.database.experiments`) and copying a project
(:mod:`autowisp.database.project_copy`) both consume what it reads. So two
readings of an unchanged project cannot disagree, and a copy can fingerprint
its source in the same pass.
"""

from sqlalchemy import MetaData, select


def begin_transaction(connection):
    """
    Make sure the connection is in a transaction, as its caller assumes.

    A server's connection always is. SQLite does not start one for a read or
    a ``PRAGMA``, so it is started here unless the connection is already in
    one; it ends when the caller's does. Every read on the connection then
    sees one state of the database, and a pragma lasting for the transaction
    (e.g. ``defer_foreign_keys``) lasts for the caller's.
    """

    if (
        connection.dialect.name == "sqlite"
        and not connection.connection.dbapi_connection.in_transaction
    ):
        connection.exec_driver_sql("BEGIN")


def _select_in_key_order(table):
    """
    Return the statement reading a table's rows in a fixed order.

    That is primary key order, or, for a table without a primary key, the
    order of its columns other than ``timestamp``, so that the rows come in
    the same order however their timestamps change.
    """

    order = list(table.primary_key.columns) or [
        column for column in table.columns if column.name != "timestamp"
    ]
    return select(table).order_by(*order)


def read_project(connection, table_consumers, batch_size=10000):
    """
    Feed every row of a project's database to whatever wants it.

    Tables are reflected from the database rather than taken from the models,
    so that a consumer sees whatever the database holds, and are read in
    dependency order, so that a table's rows come after those they refer
    to. Each table's rows come in a fixed order (see
    :func:`_select_in_key_order`), in batches, and a table nothing wants is
    not read at all.

    Args:
        connection:    A connection to the project's database. Everything is
            read within one transaction, so that everything consumed is one
            state of the project.

        table_consumers:    Callables, each given every table, as reflected
            from the database, in turn. Each returns None if it does not want
            the table's rows, or a callable to give each batch of them to: a
            list of mappings from column name to value.

        batch_size(int):    The number of rows read at a time.

    Returns:
        None
    """

    begin_transaction(connection)
    metadata = MetaData()
    metadata.reflect(connection)
    for table in metadata.sorted_tables:
        row_consumers = [
            consume_rows
            for consume_rows in (
                take_table(table) for take_table in table_consumers
            )
            if consume_rows is not None
        ]
        if not row_consumers:
            continue
        result = connection.execute(
            _select_in_key_order(table).execution_options(yield_per=batch_size)
        )
        for batch in result.mappings().partitions():
            for consume_rows in row_consumers:
                consume_rows(batch)
