"""The expression library of the project the browser interface has open.

The library lives in the project database and is read and written through
:mod:`autowisp.diagnostics.expression_library`, which the pipeline uses too,
so that a project's plots and its exclusion rules see the same expressions.
This module only supplies the session: views that just want the library,
as the ``{name: expression}`` dictionary the expression layer takes as an
argument, get it here without opening one themselves.

Every function here needs a project to be open, as every other view reading
the project database does.
"""

from autowisp.database.interface import start_db_session
from autowisp.diagnostics import expression_library


def get_expressions():
    """
    Return the open project's library as ``{name: expression}``.

    Returns:
        dict:    Every stored expression, keyed by name.
    """

    with start_db_session() as db_session:
        return expression_library.get_expressions(db_session)


def get_expression_descriptions():
    """
    Return ``{name: description}`` for the open project's expressions.

    Returns:
        dict:    Every stored expression's description, the empty string
            where one was left blank.
    """

    with start_db_session() as db_session:
        return expression_library.get_expression_descriptions(db_session)
