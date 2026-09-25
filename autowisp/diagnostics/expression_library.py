"""Read and write the diagnostic expression library of a project.

The library lives in the project database, so that the pipeline, which
evaluates exclusion rules over it, and the browser interface, which plots
and edits it, see the same expressions for the same project. Everything
here takes an open session on that database and knows nothing of Django.

Reading hands back the plain ``{name: expression}`` dictionary that
:mod:`autowisp.diagnostics.expressions` takes as an argument. What an
expression means, and what is wrong with a proposed one, is decided there;
this module only stores what it is given, keeping the references between
stored expressions intact when one of them is renamed.
"""

from sqlalchemy import delete, select, update

# pylint: disable=no-name-in-module
from autowisp.database.data_model import DiagnosticExpression

# pylint: enable=no-name-in-module
from autowisp.diagnostics.expressions import (
    get_expression_dependents,
    rename_references,
)


def get_expressions(db_session):
    """
    Return the whole library as ``{name: expression}``.

    The whole library, not the part the project has recorded diagnostics
    for: an expression naming a diagnostic nothing has recorded is not an
    error but a thing to leave unoffered, which callers that care establish
    by counting rows.

    Args:
        db_session:    Open session on the project database.

    Returns:
        dict:    Every stored expression, keyed by name.
    """

    return dict(
        db_session.execute(
            select(DiagnosticExpression.name, DiagnosticExpression.expression)
        ).all()
    )


def get_expression_descriptions(db_session):
    """
    Return ``{name: description}`` for the stored expressions.

    Kept apart from :func:`get_expressions`, which everything that
    *evaluates* an expression consumes: what a quantity is for is of no
    interest to the evaluator.

    Args:
        db_session:    Open session on the project database.

    Returns:
        dict:    Every stored expression's description, the empty string
            where one was left blank.
    """

    return dict(
        db_session.execute(
            select(DiagnosticExpression.name, DiagnosticExpression.description)
        ).all()
    )


def get_expression_entries(names, db_session):
    """
    Return the stored fields of the named expressions, for exporting.

    Args:
        names:    The expressions wanted. Names not in the library are
            skipped.

        db_session:    Open session on the project database.

    Returns:
        list:    ``{"name": ..., "expression": ..., "description": ...}``
            per expression found, ordered by name.
    """

    return [
        {"name": name, "expression": expression, "description": description}
        for name, expression, description in db_session.execute(
            select(
                DiagnosticExpression.name,
                DiagnosticExpression.expression,
                DiagnosticExpression.description,
            )
            .where(DiagnosticExpression.name.in_(list(names)))
            .order_by(DiagnosticExpression.name)
        ).all()
    ]


def store_expression(
    db_session, *, name, expression, description="", replacing=None
):
    """
    Add one expression, or replace a stored one, keeping references intact.

    Replacing under a new name is a rename. Stored expressions referencing
    the old name are rewritten to reference the new one, since leaving them
    pointing at a name that no longer exists would break them, and refusing
    the rename would not help: a dependent cannot be pointed at the new
    name before it exists.

    Nothing is validated here; the caller checks the expression against the
    library first (see :func:`~autowisp.diagnostics.expressions.
    check_expression`), including that *name* does not clash with a stored
    expression other than the one being replaced.

    Args:
        db_session:    Open session on the project database. Changes are
            flushed but not committed.

        name(str):    The name to store the expression under.

        expression(str):    The expression text.

        description(str):    What the expression is for.

        replacing(str or None):    The name of the stored expression this
            one replaces, or ``None`` to add a new one. A name not in the
            library is treated as ``None``.

    Returns:
        list:    The names of the stored expressions rewritten to follow a
            rename, alphabetically; empty if nothing was renamed or nothing
            referenced the old name.
    """

    stored = None
    if replacing is not None:
        stored = db_session.scalar(
            select(DiagnosticExpression).where(
                DiagnosticExpression.name == replacing
            )
        )

    updated = []
    if stored is not None and replacing != name:
        library = get_expressions(db_session)
        for dependent in sorted(get_expression_dependents(replacing, library)):
            db_session.execute(
                update(DiagnosticExpression)
                .where(DiagnosticExpression.name == dependent)
                .values(
                    expression=rename_references(
                        library[dependent], replacing, name
                    )
                )
            )
            updated.append(dependent)

    if stored is None:
        db_session.add(
            DiagnosticExpression(
                name=name, expression=expression, description=description
            )
        )
    else:
        stored.name = name
        stored.expression = expression
        stored.description = description

    db_session.flush()
    return updated


def delete_expressions(names, db_session):
    """
    Delete the named expressions.

    Whether anything still references them is the caller's question: a
    whole chain may be deleted together while one link of it may not be
    deleted alone, which only the caller knows.

    Args:
        names:    The expressions to delete. Names not in the library are
            ignored.

        db_session:    Open session on the project database.

    Returns:
        None
    """

    db_session.execute(
        delete(DiagnosticExpression).where(
            DiagnosticExpression.name.in_(list(names))
        )
    )


def write_expressions(entries, db_session):
    """
    Store *entries*, adding new names and overwriting existing ones.

    For importing, where references are carried by name, so overwriting an
    expression leaves its dependents pointing at the new text as intended.

    Args:
        entries(dict):    ``{name: {"expression": ..., "description": ...}}``.

        db_session:    Open session on the project database. Changes are
            flushed but not committed.

    Returns:
        (int, int):    How many expressions were added, and how many
            replaced.
    """

    stored = {
        row.name: row
        for row in db_session.scalars(
            select(DiagnosticExpression).where(
                DiagnosticExpression.name.in_(list(entries))
            )
        )
    }

    for name, entry in entries.items():
        if name in stored:
            stored[name].expression = entry["expression"]
            stored[name].description = entry["description"]
        else:
            db_session.add(
                DiagnosticExpression(
                    name=name,
                    expression=entry["expression"],
                    description=entry["description"],
                )
            )

    db_session.flush()
    return len(entries) - len(stored), len(stored)
