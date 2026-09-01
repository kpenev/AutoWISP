"""Named expressions over per-image diagnostics, evaluated in dependency order.

This is the whole of what an expression *means*: which names it references,
what order a library of them has to be evaluated in, and what is wrong with
one. It deliberately knows about no database of any kind. The library
arrives as a ``{name: expression}`` dictionary and the data as a
``{diagnostic_name: array}`` dictionary, so the same rules apply whether the
caller is the browser interface reading its own database or a pipeline step
handed a library from a file.

Passing the values in rather than fetching them is what keeps this module
free of a database and cheap to test exhaustively; it is not a facility for
supplying diagnostics by hand. Building those arrays -- NaN-padded onto one
canonical image list so that index *i* is the same image in every one of
them -- belongs to the layer above.
"""

import ast
import functools
import re

import numpy

from autowisp.evaluator import Evaluator, EvaluatorBase
from autowisp.exceptions import PipelineError


@functools.lru_cache(maxsize=1)
def _evaluator_names():
    """Return the names a bare :class:`Evaluator` already defines."""

    return frozenset(Evaluator().symtable)


def get_expression_names(expression):
    """
    Return every name one expression mentions, functions included.

    ``nanmedian(rel)`` gives ``{"nanmedian", "rel"}``: a call by bare name
    is an ``ast.Call`` whose ``func`` is an ``ast.Name``, so there is
    nothing here to tell a function from a variable, and no attempt is made
    to. Which is which cannot be decided from the text anyway -- it depends
    on the open project's diagnostics, on what else the library holds, and
    on the evaluator's symbol table -- so the split is left to the caller,
    which is what :func:`order_expressions` and :func:`check_expression`
    both do.

    Args:
        expression(str):    The expression text.

    Returns:
        set:    The ``ast.Name`` identifiers, whether they are used as
            values or called.

    Raises:
        SyntaxError:    If the text is not a single Python expression.
            ``mode="eval"`` rejects statements, assignments, loops and
            imports, which is worth having but is **not** what makes
            evaluating an expression safe: ``__import__('os').listdir('.')``
            is a perfectly valid expression. Safety comes from
            :class:`autowisp.evaluator.EvaluatorBase` -- asteval refuses
            imports, ``eval``, ``exec``, ``getattr`` and dunder traversal,
            and AutoWISP drops ``open`` and ``print`` on top -- and from
            :func:`check_expression` restricting names to that symbol table
            plus the project's diagnostics.
    """

    return {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    }


def get_bare_aggregates(expression):
    """
    Return the NaN-propagating aggregates an expression calls.

    Every array is NaN-padded to the canonical image list, so a plain
    ``median`` over one goes NaN as soon as a single image lacks the
    diagnostic -- which is usual rather than exceptional. Callers use this
    to warn, not to refuse: a deliberate ``median`` is still a legitimate
    thing to write.

    Args:
        expression(str):    The expression text.

    Returns:
        set:    The names called without their ``nan`` prefix.

    Raises:
        SyntaxError:    As for :func:`get_expression_names`.
    """

    # The NaN-propagating spelling of each aggregate the evaluator defines:
    # `median` where `nanmedian` was meant, and so on. Derived from the
    # evaluator's own list so the two cannot drift apart.
    bare = {nan_name[len("nan") :] for nan_name in EvaluatorBase.nan_aggregates}

    return {
        node.func.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in bare
    }


def get_expression_dependents(name, expressions):
    """
    Return the expressions referencing *name*, for the delete guard.

    Args:
        name(str):    The expression being deleted.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        set:    Names of expressions that would break if *name* went away.
    """

    return {
        other
        for other, expression in expressions.items()
        if other != name and name in get_expression_names(expression)
    }


def _evaluation_order(targets, expressions):
    """Return the expressions to evaluate, each after what it references.

    A depth-first walk from *targets*, appending a name only once the
    expressions it references have been appended. Only what the targets
    reach is visited, so asking for one expression does not drag in the
    whole library.

    Appending on the way *out* is what makes this an order rather than a
    traversal. On the way in, two expressions reached at the same depth are
    indistinguishable even when one references the other: with
    ``a = b + c`` and ``c = b * 2``, both ``b`` and ``c`` are reached from
    ``a`` together, so reversing the order of discovery can place ``c``
    before the ``b`` it needs.

    Raises:
        PipelineError:    On a reference cycle, naming the loop.
    """

    order = []
    done = set()
    path = []

    def visit(name):
        """Append *name*, and first everything it references."""

        if name in done:
            return
        if name in path:
            # The stack from the earlier visit down to here is exactly the
            # loop, so it can be reported as one rather than as a set of
            # suspects.
            cycle = path[path.index(name) :] + [name]
            raise PipelineError(
                "Expressions reference each other in a cycle: "
                + " -> ".join(cycle)
                + ".",
                details={"cycle": cycle},
            )

        path.append(name)
        # Sorted because the references arrive as a set, and the resulting
        # order should not depend on set iteration.
        for referenced in sorted(get_expression_names(expressions[name])):
            if referenced in expressions:
                visit(referenced)
        path.pop()

        done.add(name)
        order.append(name)

    for target in sorted(targets):
        if target in expressions:
            visit(target)

    return order


def _needed_diagnostics(targets, references, expressions, known_names):
    """Return the diagnostics to fetch, rejecting names that resolve to
    nothing.

    Raises:
        PipelineError:    If any referenced name is neither an expression,
            a diagnostic, nor an evaluator builtin.
    """

    builtins = _evaluator_names()
    needed = {name for name in targets if name in known_names}
    unresolved = {}
    for name, referenced in references.items():
        for other in referenced:
            if other in expressions:
                continue
            if other in known_names:
                needed.add(other)
            elif other not in builtins:
                unresolved.setdefault(name, set()).add(other)

    if unresolved:
        raise PipelineError(
            "Unresolvable names in "
            + ", ".join(
                f"{name} ({', '.join(sorted(names))})"
                for name, names in sorted(unresolved.items())
            )
            + ".",
            details={name: sorted(names) for name, names in unresolved.items()},
        )

    return needed


def order_expressions(targets, expressions, known_names):
    """
    Return the order to evaluate *targets* in, and what data that needs.

    Only the dependency subtree of *targets* is walked, so asking for one
    expression does not evaluate the whole library.

    Args:
        targets:    The quantity names wanted. Any that are not expressions
            are diagnostics, and pass through to the returned set.

        expressions(dict):    The library, ``{name: expression}``.

        known_names:    Names that resolve to real data: the diagnostics
            recorded in the open project, plus ``jd``.

    Returns:
        tuple:
            list:    Expression names, each after everything it references.

            set:    The diagnostics that have to be fetched for them.

    Raises:
        PipelineError:    On a reference cycle, or on a name that is
            neither an expression, a diagnostic, nor an evaluator builtin.
    """

    unknown = sorted(
        name
        for name in targets
        if name not in expressions and name not in known_names
    )
    if unknown:
        raise PipelineError(
            "Cannot plot " + ", ".join(unknown) + ": no such diagnostic or "
            "expression.",
            details={"unknown": unknown},
        )

    # The order is also exactly the set of expressions reached, so it
    # doubles as the subtree to collect diagnostics from.
    order = _evaluation_order(targets, expressions)
    references = {
        name: get_expression_names(expressions[name]) for name in order
    }

    return (
        order,
        _needed_diagnostics(targets, references, expressions, known_names),
    )


def _as_series(values, count):
    """Return *values* as an array of *count* entries.

    A constant-valued expression evaluates to a scalar, which still has to
    plot as a series, so it is broadcast to the image count.
    """

    values = numpy.atleast_1d(values)
    if values.size == count:
        return values
    if values.size == 1:
        return numpy.full(count, values.item())
    raise PipelineError(
        f"An expression produced {values.size} values for {count} images.",
        details={"produced": int(values.size), "images": int(count)},
    )


def evaluate_expressions(targets, expressions, values):
    """
    Evaluate the wanted quantities against already-fetched values.

    Every expression in the dependency subtree is computed once, its result
    assigned back into the same symbol table, so one used twice -- or shared
    by several dependents -- is not recomputed. Passing several *targets*
    rather than calling once per target is what extends that across the two
    axes of a plot.

    Args:
        targets:    The quantity names wanted.

        expressions(dict):    The library, ``{name: expression}``.

        values(dict):    ``{diagnostic_name: array}``, all on a common
            index -- in practice one canonical image list.

    Returns:
        dict:    ``{target: array}``, each of the same length as *values*.

    Raises:
        PipelineError:    As for :func:`order_expressions`, or if *values*
            lacks a diagnostic the expressions need.
    """

    order, needed = order_expressions(targets, expressions, set(values))

    missing = sorted(needed - set(values))
    if missing:
        raise PipelineError(
            "No values supplied for " + ", ".join(missing) + ".",
            details={"missing": missing},
        )

    count = numpy.size(next(iter(values.values()))) if values else 0

    evaluate = Evaluator(dict(values))
    for name in order:
        evaluate.symtable[name] = evaluate(expressions[name])

    return {
        target: _as_series(evaluate.symtable[target], count)
        for target in targets
    }


def check_expression(name, expression, expressions, known_names):
    """
    Return what is wrong with a proposed expression, as plain strings.

    Problems are returned rather than raised so that this module stays free
    of any particular presentation: the browser interface turns them into a
    ``ValidationError``, an importer collects them per entry.

    Args:
        name(str):    The proposed name.

        expression(str):    The proposed text.

        expressions(dict):    The library it would join. An existing entry
            of the same name is replaced by the proposed text rather than
            conflicting with it, so an edit can pass the library unchanged.

        known_names:    Names that resolve to real data, which are also the
            names an expression may not take -- an expression cannot shadow
            a diagnostic, since both are variables in the same flat space.

    Returns:
        list:    Descriptions of the problems; empty if there are none.
    """

    problems = []

    # Django's slug charset, spelled out so this module needs no Django. An
    # expression is selected through `image/<slug:x>/vs/<slug:y>`, so a name
    # outside this set could be stored but never plotted.
    if not name or not re.match(r"^[-a-zA-Z0-9_]+\Z", name):
        problems.append(
            f"{name!r} is not a valid name: use letters, digits, hyphens "
            "and underscores, so that it survives being put in a URL."
        )
    if name in known_names:
        problems.append(
            f"{name!r} is already the name of a diagnostic, and an "
            "expression cannot shadow one."
        )

    try:
        referenced = get_expression_names(expression)
    except SyntaxError as error:
        problems.append(
            f"Not a single expression: {error.msg}. Assignments, "
            "statements and imports are not allowed."
        )
        return problems

    unresolvable = sorted(
        other
        for other in referenced
        if other != name
        and other not in known_names
        and other not in expressions
        and other not in _evaluator_names()
    )
    if unresolvable:
        problems.append(
            "Not a diagnostic, an expression or a function: "
            + ", ".join(unresolvable)
            + "."
        )
        # Ordering would only fail on the same names, less legibly.
        return problems

    try:
        order_expressions(
            [name], dict(expressions, **{name: expression}), known_names
        )
    except PipelineError as error:
        problems.append(str(error))

    return problems
