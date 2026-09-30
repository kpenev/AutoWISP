"""Named expressions over per-image diagnostics, evaluated in dependency order.

This is the whole of what an expression *means*: which names it references,
what order a library of them has to be evaluated in, and what is wrong with
one. It deliberately knows about no database of any kind. The library
arrives as a ``{name: expression}`` dictionary and the data as a
``{diagnostic_name: array}`` dictionary, so the same rules apply whether the
caller is the browser interface or the pipeline, both of which read the
project's library through :mod:`autowisp.diagnostics.expression_library`.

An exclusion rule is one more expression, taking at most one channel slot
and one photometry slot, and evaluated alongside the library under
:data:`rule_quantity`.

Passing the values in rather than fetching them is what keeps this module
free of a database and cheap to test exhaustively; it is not a facility for
supplying diagnostics by hand. Building those arrays -- NaN-padded onto one
canonical image list so that index *i* is the same image in every one of
them -- belongs to the layer above.
"""

# Over pylint's 1000-line default, and left there: the split that suggests
# itself -- reading expression text apart from evaluating it -- would put
# most of the reading half on the evaluating half's import list.
# pylint: disable=too-many-lines

import ast
import functools
import re

import numpy

from autowisp.diagnostics.diagnostic_types import (
    is_diagnostic,
    is_known_quantity,
    parse_photometry_literal,
    photometry_diagnostic_names,
    photometry_literal,
    time_quantity,
)
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


def get_indexed_names(expression):
    """
    Return what one expression reads, in which channel and photometry slots.

    A channel slot is written as a subscript -- ``bg_center[0]``, or
    ``sky_color[0,1]`` for a quantity taking two -- and the numbers are
    that expression's own formal parameters, bound to real channels only
    when something is plotted. So this reports the *shape* of what the
    text asks for, and says nothing about channels.

    A slot may instead be a quoted channel name, ``bg_center['G0']``,
    which binding passes through unchanged: it is already bound. That ties
    the text to a camera's channel naming, which is the point in an
    exclusion rule and a convenience in a project whose cameras all name
    their channels alike.

    A quantity taking photometries -- a diagnostic ``fit_magnitudes``
    produces, or an expression reading one -- is read with a second
    subscript for them, numbered in a namespace of its own:
    ``magfit_residual[0][1]`` is channel slot 0 and photometry slot 1, and
    ``magfit_residual[0]['ap4']`` quotes aperture 4. One taking no channel
    is written with an empty channel subscript, ``name[()][0]``.

    Args:
        expression(str):    The expression text.

    Returns:
        list:    ``(name, channel_slots, photometry_slots)`` triples in the
            order they appear, both always tuples however many were
            written, each slot an ``int`` or a quoted ``str``; the
            photometry slots empty where there is no second subscript.
            Repeats are kept: ``sky_color[0,1] - sky_color[1,2]`` reads one
            name at two different slot pairs, which is the whole point of
            the parameters being formal.

    Raises:
        SyntaxError:    As for :func:`get_expression_names`.

        PipelineError:    If a subscript is not a plain name indexed by
            integer or string literals, at most twice. Anything else --
            ``bg_center[i]``, ``bg_center[1:2]``, ``x[0][1][2]`` -- cannot
            name a slot, and saying so here is clearer than letting it fail
            as an unresolvable name later. So is an empty channel subscript
            with no photometries after it.
    """

    tree = ast.parse(expression, mode="eval")
    nested = _nested_reads(tree)
    return [
        _get_read(node)
        for node in _in_written_order(tree)
        if isinstance(node, ast.Subscript) and id(node) not in nested
    ]


def _in_written_order(tree):
    """Return the nodes of *tree* in the order their text starts.

    Rather than the breadth-first order of ``ast.walk``, which puts
    ``y['R']`` before ``x['B']`` in ``2 * x['B'] + y['R']``, being
    shallower.
    """

    return sorted(
        (node for node in ast.walk(tree) if hasattr(node, "col_offset")),
        key=lambda node: (node.lineno, node.col_offset),
    )


def _nested_reads(tree):
    """Return the ids of the subscripts in *tree* that are part of another.

    The ``x[0]`` of ``x[0][1]`` is a subscript node of its own, but it is
    part of the whole read rather than a read.
    """

    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Subscript)
    }


def _literal_slots(index):
    """Return the slots *index* writes as a tuple, or ``None`` if it can't.

    Each slot is an ``int`` or a ``str`` literal; anything else, a name or
    a slice, writes no slot. The tuple may be empty, for the caller to
    judge.
    """

    try:
        written = ast.literal_eval(index)
    except ValueError:
        return None
    if not isinstance(written, tuple):
        written = (written,)
    if all(
        isinstance(value, str)
        or (isinstance(value, int) and not isinstance(value, bool))
        for value in written
    ):
        return written
    return None


def _get_read(read):
    """Return ``(name, channel_slots, photometry_slots)`` for one read.

    *read* is the whole read: ``x[0]``, or ``x[0][1]`` with photometries.
    The photometry slots are ``()`` where there is no second subscript,
    and the channel slots may be empty only where there is one:
    ``name[()][0]`` is how a quantity taking photometries but no channel
    is read.

    Raises:
        PipelineError:    As described in :func:`get_indexed_names`.
    """

    has_photometries = isinstance(read.value, ast.Subscript)
    channel_part = read.value if has_photometries else read
    if not isinstance(channel_part.value, ast.Name):
        raise PipelineError(
            f"{ast.unparse(channel_part)!r} does not name a channel slot: "
            "only a diagnostic or an expression can take one.",
            details={"subscript": ast.unparse(channel_part)},
        )

    channels = _literal_slots(channel_part.slice)
    if channels is None or not (channels or has_photometries):
        raise PipelineError(
            f"{ast.unparse(channel_part)!r} does not name a channel slot: "
            "a slot is written as a whole number, as in bg_center[0], or "
            "as a quoted channel name, as in bg_center['G0'].",
            details={"subscript": ast.unparse(channel_part)},
        )

    if not has_photometries:
        return channel_part.value.id, channels, ()
    photometries = _literal_slots(read.slice)
    if not photometries:
        raise PipelineError(
            f"{ast.unparse(read)!r} does not name a photometry slot: a "
            "slot is written as a whole number, as in "
            "magfit_residual[0][0], or as a quoted photometry, as in "
            "magfit_residual[0]['ap4'].",
            details={"subscript": ast.unparse(read)},
        )
    return channel_part.value.id, channels, photometries


def _get_bare_names(expression):
    """Return the names *expression* reads without a subscript.

    The complement of :func:`get_indexed_names`, so between them every
    ``ast.Name`` is accounted for once. Mostly functions.
    """

    tree = ast.parse(expression, mode="eval")

    # The nodes rather than their ids: one name may be read both ways, and
    # ``bg_center[0] - bg_center`` has to report the second.
    subscripted = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
    }

    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node not in subscripted
    }


def get_channel_parameters(expression):
    """
    Return the channel slots one expression takes, in order.

    Derived from the body rather than declared: the parameters *are* the
    slot numbers it mentions, so there is nothing to keep in step and
    nothing extra to store. ``bg_center[0] / bg_center[1]`` takes ``(0,
    1)``; ``sky_color[0,1] - sky_color[1,2]`` takes ``(0, 1, 2)``.

    Sorted, which is what makes the order well defined when a body writes
    its slots out of sequence -- and the order matters, since a reference's
    arguments are matched onto these positionally.

    A quoted channel is not a parameter: it is already bound. So
    ``x[0] - x['G0']`` takes ``(0,)``, and ``sky_color['B', 'R']`` takes
    none.

    Args:
        expression(str):    The expression text.

    Returns:
        tuple:    The distinct slot numbers, ascending. Empty for an
            expression that reads no channel at all, or only quoted ones.

    Raises:
        SyntaxError, PipelineError:    As for
            :func:`get_indexed_names`.
    """

    return tuple(
        sorted(
            {
                slot
                for _, channels, _ in get_indexed_names(expression)
                for slot in channels
                if not isinstance(slot, str)
            }
        )
    )


def get_photometry_parameters(expression):
    """
    Return the photometry slots one expression takes, in order.

    As :func:`get_channel_parameters` returns the channel slots, from
    the second subscripts instead:
    ``magfit_residual[0][0] - magfit_residual[0][1]`` takes ``(0, 1)``, and
    ``magfit_residual[0]['ap4']``, whose photometry is quoted, none.

    Args:
        expression(str):    The expression text.

    Returns:
        tuple:    The distinct slot numbers, ascending.

    Raises:
        SyntaxError, PipelineError:    As for
            :func:`get_indexed_names`.
    """

    return tuple(
        sorted(
            {
                slot
                for _, _, photometries in get_indexed_names(expression)
                for slot in photometries
                if not isinstance(slot, str)
            }
        )
    )


def _bind_channels(slots, binding):
    """Return the channels *slots* name under *binding*.

    A slot number is looked up; a quoted channel is already one.
    """

    return tuple(
        slot if isinstance(slot, str) else binding[slot] for slot in slots
    )


def _bind_photometries(slots, binding):
    """Return the photometry ids *slots* name under *binding*.

    A slot number is looked up; a quoted photometry is read as the id it
    spells, ``'ap4'`` as 4.
    """

    return tuple(
        (
            parse_photometry_literal(slot)
            if isinstance(slot, str)
            else binding[slot]
        )
        for slot in slots
    )


def get_channel_arity(name, expressions):
    """
    Return how many channels *name* has to be bound to before it can be read.

    Only one of the three answers is data. A diagnostic is recorded once
    per channel, so it takes exactly one; :data:`time_quantity` is recorded
    once per image and takes none; and an expression takes however many
    slots its body mentions, which is the only part worth deriving.

    The table above a plot asks this to know how many channel dropdowns an
    axis needs.

    Args:
        name(str):    A diagnostic, an expression, or
            :data:`time_quantity`.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        int:    The number of channels to bind.

    Raises:
        PipelineError:    If *name* is neither a diagnostic nor an
            expression nor the time.
    """

    if name in expressions:
        return len(get_channel_parameters(expressions[name]))
    if name == time_quantity:
        return 0
    if is_diagnostic(name):
        return 1
    raise PipelineError(
        f"Cannot plot {name}: no such diagnostic or expression.",
        details={"unknown": [name]},
    )


def get_photometry_arity(name, expressions):
    """
    Return how many photometries *name* has to be bound to.

    As :func:`get_channel_arity` counts channels: a diagnostic recorded per
    photometry (see
    :func:`~autowisp.diagnostics.diagnostic_types.photometry_diagnostic_names`)
    takes exactly one; any other diagnostic, and the time, take none; and an
    expression takes however many photometry slots its body mentions.

    Args:
        name(str):    A diagnostic, an expression, or
            :data:`time_quantity`.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        int:    The number of photometries to bind.

    Raises:
        PipelineError:    If *name* is neither a diagnostic nor an
            expression nor the time.
    """

    if name in expressions:
        return len(get_photometry_parameters(expressions[name]))
    if name == time_quantity or is_diagnostic(name):
        return int(name in photometry_diagnostic_names())
    raise PipelineError(
        f"Cannot plot {name}: no such diagnostic or expression.",
        details={"unknown": [name]},
    )


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


def get_logical_keywords(expression):
    """
    Return the Python logical keywords an expression uses.

    ``and``, ``or`` and ``not`` ask for the truth value of their operands,
    which an array of more than one element refuses to have ("the truth
    value of an array is ambiguous"): element-wise logic is ``|``, ``&``
    and ``~``. On a quantity with one value, though, the keywords are
    legitimate, so -- like :func:`get_bare_aggregates` -- this is for
    callers to warn with, never to refuse.

    Args:
        expression(str):    The expression text.

    Returns:
        set:    Those of ``"and"``, ``"or"`` and ``"not"`` that appear.

    Raises:
        SyntaxError:    As for :func:`get_expression_names`.
    """

    keywords = set()
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if isinstance(node, ast.BoolOp):
            keywords.add("and" if isinstance(node.op, ast.And) else "or")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            keywords.add("not")
    return keywords


def rename_references(expression, old_name, new_name):
    """
    Return *expression* with every reference to *old_name* renamed.

    A source-level edit, and the only rewriting anywhere in this feature.
    It does not contradict the rule that nothing is rewritten -- that is
    about *resolution*, where an ``ast.unparse`` roundtrip would leave a
    stored text and a resolved text to keep straight. Here the user has
    asked for the change, and what comes back is what they will read and
    edit from then on.

    So it must not reformat: the new identifier is spliced in at the exact
    source offsets ``ast`` reports, leaving every other byte alone. That is
    also what makes it safe where a textual replace is not -- ``rel``
    inside ``rel_bg``, inside a string literal, or as a keyword argument's
    name is left untouched, because only ``ast.Name`` nodes are moved.

    Args:
        expression(str):    The expression text to rewrite.

        old_name(str):    The name being renamed.

        new_name(str):    What to call it instead.

    Returns:
        str:    The text with those references renamed, and identical
            everywhere else.

    Raises:
        SyntaxError:    As for :func:`get_expression_names`.
    """

    # The offsets `ast` reports count utf-8 bytes rather than characters,
    # so the splicing happens on the encoded text: a non-ASCII character
    # earlier in the line would otherwise shift every offset after it.
    encoded = expression.encode("utf-8")

    line_starts = []
    offset = 0
    for line in encoded.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line)

    spans = sorted(
        (
            line_starts[node.lineno - 1] + node.col_offset,
            line_starts[node.end_lineno - 1] + node.end_col_offset,
        )
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name) and node.id == old_name
    )

    # Right to left, so that an earlier span's offsets are still valid
    # after a later one has changed the length of the text.
    for start, end in reversed(spans):
        encoded = encoded[:start] + new_name.encode("utf-8") + encoded[end:]

    return encoded.decode("utf-8")


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


def _needed_diagnostics(targets, references, expressions):
    """Return the diagnostics to fetch, rejecting names that resolve to
    nothing.

    Raises:
        PipelineError:    If any referenced name is neither an expression,
            a diagnostic, nor an evaluator builtin.
    """

    builtins = _evaluator_names()
    needed = {name for name in targets if is_known_quantity(name)}
    unresolved = {}
    for name, referenced in references.items():
        for other in referenced:
            if other in expressions:
                continue
            if is_known_quantity(other):
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


def order_expressions(targets, expressions):
    """
    Return the order to evaluate *targets* in, and what data that needs.

    Only the dependency subtree of *targets* is walked, so asking for one
    expression does not evaluate the whole library.

    What a name may mean comes from
    :func:`~autowisp.diagnostics.diagnostic_types.is_known_quantity` rather
    than from a project, because it cannot differ between projects: a
    ``diagnostic_type`` row is either seeded from the static catalogue or
    created by the quantile branch that refuses every other name. A
    diagnostic no image here records is therefore accepted and comes back
    all-NaN, which is what the padding is for.

    Args:
        targets:    The quantity names wanted. Any that are not expressions
            are diagnostics, and pass through to the returned set.

        expressions(dict):    The library, ``{name: expression}``.

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
        if name not in expressions and not is_known_quantity(name)
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

    return order, _needed_diagnostics(targets, references, expressions)


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


class QuantityLookUp:
    """
    One name an expression may read, resolved when it is asked for.

    Everything a quantity needs is held here: a diagnostic owns the values
    fetched for it, an expression owns its text and the instantiations it
    has computed. Asking the top-level lookup for its channels is
    therefore the whole of evaluation -- there is no separate pass, and no
    state threaded through one.

    A diagnostic is the degenerate case rather than a second kind of
    thing: every instantiation of it is known before evaluation starts, so
    it arrives with its cache full and its text never consulted.

    Built through :meth:`library` rather than one at a time, because the
    lookups of one evaluation share a binding stack and an evaluator, and
    both are this class's business rather than its caller's.
    """

    def __init__(
        self,
        name,
        shared,
        *,
        definition=None,
        computed=None,
        takes_photometries=False,
    ):
        """
        Args:
            name(str):    What the expressions call it, used for error
                messages only -- nothing resolves by it.

            shared(tuple):    The binding stack and evaluator this
                evaluation's lookups share, from :meth:`library`.

            definition(tuple):    The expression text, and its channel and
                its photometry parameters as a pair, or ``None`` for a
                diagnostic, which has none of them because it is never
                evaluated.

            computed(dict):    ``(channels, photometries) -> array`` known
                in advance: the whole of a diagnostic, and empty for an
                expression.

            takes_photometries(bool):    Whether a diagnostic is read with a
                second subscript, binding a photometry. An expression takes
                photometries if it has photometry parameters.
        """

        self._name = name
        self._stack, self._evaluator = shared
        # The channel and the photometry parameters, paired as a binding
        # and a stack frame are.
        self._text, self._parameters = definition or (None, ((), ()))
        self._takes_photometries = takes_photometries or bool(
            self._parameters[1]
        )
        self._computed = dict(computed or {})

    @classmethod
    def library(cls, expressions, values):
        """
        Return ``{name: lookup}`` for one evaluation, ready to be asked.

        The binding stack and the evaluator are created here and captured
        by the lookups, so neither is named outside this class. The stack
        is how a lookup finds the binding of the body asking it, which is
        nobody else's concern; and it must belong to **one evaluation**
        rather than to the class, or two plots drawn at once would
        interleave their bindings on it and return wrong numbers rather
        than failing.

        The evaluator does not come back out either: handing it over would
        hand over a symbol table full of lookups, and with it a way to
        evaluate arbitrary text with none of the parameter machinery.

        Args:
            expressions(dict):    The library, ``{name: expression}``.

            values(dict):    ``{name: {(channels, photometries): array}}``,
                as fetched for one series, keyed exactly as
                :func:`get_needed_values` asked.

        Returns:
            dict:    A lookup per name, sharing one evaluation's state.
        """

        shared = ([], Evaluator({}))
        per_phot_names = photometry_diagnostic_names()
        lookups = {
            name: cls(
                name,
                shared,
                computed=by_binding,
                takes_photometries=name in per_phot_names,
            )
            for name, by_binding in values.items()
        }
        lookups.update(
            {
                name: cls(
                    name,
                    shared,
                    definition=(
                        text,
                        (
                            get_channel_parameters(text),
                            get_photometry_parameters(text),
                        ),
                    ),
                )
                for name, text in expressions.items()
            }
        )

        symtable = shared[1].symtable
        symtable.update(lookups)

        # A quantity binding neither channels nor photometries -- the time,
        # and any expression whose channels and photometries, if any, are
        # all quoted -- is written without a subscript, there being nothing
        # to subscript it with. A lookup resolves on ``__getitem__`` and a
        # bare name never calls one, so such a quantity has to be its
        # *values* here, or it would reach the arithmetic as the object
        # itself. The time first, then the expressions over it, each after
        # what it reads. One taking photometries but no channel is written
        # ``name[()][0]``, so it stays a lookup.
        for name, by_binding in values.items():
            if ((), ()) in by_binding:
                symtable[name] = by_binding[(), ()]

        parameter_free = {
            name
            for name, text in expressions.items()
            if not get_channel_parameters(text)
            and not get_photometry_parameters(text)
        }
        # The order also holds what they reach by subscript, such as
        # ``sky_color`` in ``sky_color['B', 'R']``, which has parameters to
        # bind and stays a lookup.
        for name in _evaluation_order(parameter_free, expressions):
            if name in parameter_free:
                symtable[name] = lookups[name].at((), ())

        return lookups

    def __getitem__(self, slots):
        """
        Resolve ``name[slots]`` as the body being evaluated means it.

        Python passes ``1`` for ``x[1]`` and ``(1, 2)`` for ``x[1,2]``, so
        the shapes take care of themselves. The binding read is the top of
        the stack, which is always the body asking: a nested evaluation
        finishes in here before the outer body's next operand is touched.
        A quoted channel, ``x['G0']``, is not looked up in it.

        A quantity taking photometries is not resolved yet: its channels
        are bound, and the :class:`_ChannelBound` returned resolves the
        second subscript, which such a quantity always has.
        """

        if not isinstance(slots, tuple):
            slots = (slots,)
        channel_binding, photometry_binding = self._stack[-1]
        channels = _bind_channels(slots, channel_binding)
        if self._takes_photometries:
            return _ChannelBound(self, channels, photometry_binding)
        return self.at(channels, ())

    def at(self, channels, photometries):
        """
        Return this quantity with its parameters bound as given.

        The cache is what makes an instantiation wanted twice -- by two
        references, or by two different expressions -- computed once. For a
        diagnostic it is also the whole of the answer, so a miss means the
        values were never fetched rather than that something needs
        computing, and is reported rather than repaired: it means
        :func:`get_needed_values` and whatever fetched disagree.

        Nothing here guards against a reference cycle:
        :func:`order_expressions` refuses one statically, on names, and
        that is exact rather than conservative, since substitution permutes
        a finite parameter set and introduces no new symbols.

        Args:
            channels(tuple):    One channel per channel parameter, in
                order.

            photometries(tuple):    One photometry id per photometry
                parameter, in order.

        Returns:
            numpy.ndarray:    The values, over the canonical image list.

        Raises:
            PipelineError:    If a diagnostic was not fetched for this
                binding.
        """

        binding = (channels, photometries)
        if binding not in self._computed:
            if self._text is None:
                raise PipelineError(
                    f"No values supplied for {self._name} in "
                    + ", ".join(
                        [*channels, *map(photometry_literal, photometries)]
                    )
                    + ".",
                    details={
                        "missing": [self._name],
                        "channels": channels,
                        "photometries": photometries,
                    },
                )
            self._stack.append(
                tuple(
                    dict(zip(parameters, bound))
                    for parameters, bound in zip(self._parameters, binding)
                )
            )
            try:
                self._computed[binding] = self._evaluator(self._text)
            finally:
                self._stack.pop()
        return self._computed[binding]


class _ChannelBound:  # pylint: disable=too-few-public-methods
    """
    A quantity taking photometries, with its channels bound.

    What ``x[0]`` gives inside ``x[0][1]``: the second subscript then binds
    the photometries, against the binding of the body that wrote it, and
    returns the values. Every read of such a quantity has that second
    subscript (:func:`check_expression` refuses one without), so this never
    reaches the arithmetic in place of values.
    """

    def __init__(self, lookup, channels, photometry_binding):
        """
        Args:
            lookup(QuantityLookUp):    The quantity read.

            channels(tuple):    Its channels, already bound.

            photometry_binding(dict):    The photometry binding of the body
                reading it, taken when its channels were bound: the same
                body's, since nothing is evaluated between the subscripts.
        """

        self._lookup = lookup
        self._channels = channels
        self._photometry_binding = photometry_binding

    def __getitem__(self, slots):
        """Resolve the photometries of ``x[...][slots]``, and read it."""

        if not isinstance(slots, tuple):
            slots = (slots,)
        return self._lookup.at(
            self._channels,
            _bind_photometries(slots, self._photometry_binding),
        )


def _visit_needed(name, channels, photometries, expressions, needed):
    """Add what *name* bound to *channels* and *photometries* reads.

    Recursively, and both kinds of reference are followed. A subscripted
    one is bound through *channels* and *photometries*; a bare one takes
    neither, being the time or an expression whose channels and
    photometries, if any, are all quoted -- and the latter still reads
    diagnostics, so stopping at bare names would leave them unfetched.
    Other bare names are functions.
    """

    expected = get_channel_arity(name, expressions)
    if len(channels) != expected:
        raise PipelineError(
            f"{name} takes {expected} channel(s), not {len(channels)}.",
            details={"quantity": name, "channels": list(channels)},
        )
    expected = get_photometry_arity(name, expressions)
    if len(photometries) != expected:
        raise PipelineError(
            f"{name} takes {expected} photometries, not "
            f"{len(photometries)}.",
            details={"quantity": name, "photometries": list(photometries)},
        )

    if name not in expressions:
        # A diagnostic, or the time -- either way a leaf, and either way
        # named by the binding it is read in, which for the time is empty.
        needed.setdefault(name, set()).add((channels, photometries))
        return

    text = expressions[name]
    channel_binding = dict(zip(get_channel_parameters(text), channels))
    photometry_binding = dict(
        zip(get_photometry_parameters(text), photometries)
    )
    for referenced, channel_slots, photometry_slots in get_indexed_names(text):
        _visit_needed(
            referenced,
            _bind_channels(channel_slots, channel_binding),
            _bind_photometries(photometry_slots, photometry_binding),
            expressions,
            needed,
        )
    for referenced in _get_bare_names(text):
        if referenced in expressions or referenced == time_quantity:
            _visit_needed(referenced, (), (), expressions, needed)


def get_needed_values(wanted, expressions):
    """
    Return the diagnostics to read, and in which channels.

    Runs before anything is evaluated, for two callers that both need the
    answer without it:

    * whatever reads the values does so for a whole series in one query,
      so it cannot discover them as it goes;
    * the table above a plot counts the images recording all of them,
      which is a question about rows and must never evaluate an
      expression -- there is a table row per observing session and image
      type, so evaluating per row would be work proportional to the whole
      image collection.

    It walks what evaluation walks, resolving each reference's arguments
    through the binding of the body holding it, and stops at the
    diagnostics and the time.

    Args:
        wanted(dict):    ``{quantity: set of (channels, photometries)}``,
            each a tuple holding one channel per channel parameter of that
            quantity, and one photometry id per photometry parameter. A
            *set* because one quantity may be wanted at two bindings at
            once: ``bg_center`` in R against ``bg_center`` in B is a plot of
            a diagnostic between channels.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        dict:    ``{name: set of (channels, photometries)}`` -- the same
            shape it takes, which is what it means: what is wanted, resolved
            into what must be read. One entry per diagnostic, holding every
            binding it is read in, plus :data:`time_quantity` with
            ``((), ())`` where an expression reads the time.

    Raises:
        PipelineError:    On a reference cycle, on a name that resolves to
            nothing, or on a binding of the wrong length for what it binds.
    """

    # Refuses a cycle and an unresolvable name, so the walk below cannot
    # recurse for ever and neither it nor the evaluation need check again.
    order_expressions(list(wanted), expressions)

    needed = {}
    for quantity, bindings in wanted.items():
        for channels, photometries in bindings:
            _visit_needed(
                quantity,
                tuple(channels),
                tuple(photometries),
                expressions,
                needed,
            )

    return needed


def _canonical_length(values):
    """Return the length of the image list *values* are padded onto."""

    for by_channels in values.values():
        for array in by_channels.values():
            return numpy.size(array)
    return 0


def evaluate_quantities(wanted, expressions, values):
    """
    Evaluate the quantities of one series, each bound as asked.

    Args:
        wanted(dict):    ``{quantity: set of (channels, photometries)}``,
            as the table bound them and as :func:`get_needed_values` takes
            them. Asking for both axes at once is what lets an
            instantiation they share be computed once.

        expressions(dict):    The library, ``{name: expression}``.

        values(dict):    ``{name: {(channels, photometries): array}}``,
            every array on one canonical image list, holding what
            :func:`get_needed_values` asked for and keyed the same way.

    Returns:
        dict:    ``{quantity: {(channels, photometries): array}}``, all of
            the same length -- the shape *values* arrives in, being the
            same kind of thing: a quantity read in the binding asked for.

    Raises:
        PipelineError:    If a quantity resolves to nothing, if the
            expressions reference each other in a cycle, or if *values*
            lacks something they read.
    """

    # Only what these quantities reach. The library may hold dozens of
    # expressions about other diagnostics entirely, and nothing was fetched
    # for those -- so building lookups for them would at best be waste, and
    # for a channel-free one, which is evaluated on sight, an error about
    # values nobody asked for.
    order, _ = order_expressions(list(wanted), expressions)
    lookups = QuantityLookUp.library(
        {name: expressions[name] for name in order}, values
    )
    count = _canonical_length(values)

    return {
        quantity: {
            (tuple(channels), tuple(photometries)): _as_series(
                lookups[quantity].at(tuple(channels), tuple(photometries)),
                count,
            )
            for channels, photometries in bindings
        }
        for quantity, bindings in wanted.items()
    }


def _spell_channel_slots(count):
    """Return *count* channel slots, spelled for a message."""

    if count == 0:
        return "no channel slot"
    return "1 channel slot" if count == 1 else f"{count} channel slots"


def _spell_photometry_slots(count):
    """Return *count* photometry slots, spelled for a message."""

    if count == 0:
        return "no photometry slot"
    return "1 photometry slot" if count == 1 else f"{count} photometry slots"


def _spell_read(name, channels, photometries):
    """Return a read of *name* at these slots, as an expression writes it.

    The channel subscript is ``[()]`` where there are no channels but
    photometries follow, and the photometry one is left out where there are
    none.
    """

    written = f"{name}[{', '.join(map(repr, channels)) or '()'}]"
    if photometries:
        written += f"[{', '.join(map(repr, photometries))}]"
    return written


def _slot_problems(expression, library):
    """
    Return what is wrong with how *expression* reads its quantities.

    One rule: a quantity is read with exactly as many channel slots and
    photometry slots as it takes, and one taking neither is read bare. The
    first half is what makes a reference's arguments match a definition's
    parameters positionally, and what makes the second subscript present
    exactly where a :class:`_ChannelBound` needs resolving; the second is
    what lets :func:`get_needed_values` and :meth:`QuantityLookUp.library`
    treat a bare name as a value and still know they have seen everything.
    And a quoted photometry must name one.

    Only names resolving to a quantity are judged, in either direction:
    the evaluator's own arrays can be indexed too, and most bare names are
    its functions.

    Args:
        expression(str):    The proposed text.

        library(dict):    The library it would join, ``{name: expression}``,
            *including* the proposed expression, so that a self-reference
            has an arity.

    Returns:
        list:    Descriptions of the problems; empty if there are none.

    Raises:
        PipelineError:    From :func:`get_indexed_names`, if an index does
            not name a channel slot at all.
    """

    def is_quantity(candidate):
        """Whether *candidate* reads data, rather than being a function."""

        return candidate in library or is_known_quantity(candidate)

    problems = []

    for referenced, channels, photometries in get_indexed_names(expression):
        if not is_quantity(referenced):
            continue
        written = _spell_read(referenced, channels, photometries)
        channel_arity = get_channel_arity(referenced, library)
        photometry_arity = get_photometry_arity(referenced, library)
        if not channel_arity and not photometry_arity:
            problems.append(
                f"{referenced} takes no slot, being one value per image: "
                f"write {referenced} rather than {written}."
            )
            continue
        if channel_arity != len(channels):
            problems.append(
                f"{referenced} takes {_spell_channel_slots(channel_arity)}, "
                f"not {len(channels)}: {written}."
            )
        if photometry_arity != len(photometries):
            problems.append(
                f"{referenced} takes "
                f"{_spell_photometry_slots(photometry_arity)}, not "
                f"{len(photometries)}: {written}."
            )
        problems.extend(
            f"{literal!r} names no photometry in {written}: quote one as "
            "'shapefit', or as 'ap' followed by the aperture index, as in "
            "'ap4'."
            for literal in photometries
            if isinstance(literal, str)
            and parse_photometry_literal(literal) is None
        )

    for referenced in sorted(_get_bare_names(expression)):
        if not is_quantity(referenced):
            continue
        channel_arity = get_channel_arity(referenced, library)
        photometry_arity = get_photometry_arity(referenced, library)
        if channel_arity or photometry_arity:
            taken = [
                spell(arity)
                for spell, arity in (
                    (_spell_channel_slots, channel_arity),
                    (_spell_photometry_slots, photometry_arity),
                )
                if arity
            ]
            sample = _spell_read(
                referenced,
                tuple(range(channel_arity)),
                tuple(range(photometry_arity)),
            )
            problems.append(
                f"{referenced} takes {' and '.join(taken)}, so it cannot be "
                f"read on its own: write {sample}."
            )

    return problems


def check_expression(name, expression, current_library):
    """
    Return what is wrong with a proposed expression, as plain strings.

    Problems are returned rather than raised so that this module stays free
    of any particular presentation: the browser interface turns them into a
    ``ValidationError``, an importer collects them per entry.

    **No project is needed.** What a name may mean comes from
    :mod:`autowisp.diagnostics.diagnostic_types`, which is complete: a
    ``diagnostic_type`` row is either seeded from the static catalogue at
    project creation or created by the ``pixel_q*`` branch of
    ``_save_image_diagnostics``, which refuses every other name. So no
    project can contain a diagnostic this does not know, and an expression
    means the same thing everywhere -- which is what lets expressions be
    exported from one project and imported into another.

    Whether an expression is *usable* in a particular project is a
    different question, about whether rows have been recorded, and is
    answered by counting them rather than here.

    Args:
        name(str):    The proposed name.

        expression(str):    The proposed text.

        current_library(dict):    The library it would join, as it stands.
            An existing entry of the same name is replaced by the proposed
            text rather than conflicting with it, so an edit can pass the
            library unchanged.

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
    if is_known_quantity(name):
        problems.append(
            f"{name!r} already names a diagnostic, and an expression "
            "cannot shadow one: the two share a name space, which is what "
            "lets a selector and a URL treat them alike."
        )

    return problems + _body_problems(name, expression, current_library)


def _body_problems(name, expression, current_library):
    """
    Return what is wrong with the text of an expression or a rule.

    Everything :func:`check_expression` judges apart from the name, which
    is the whole of what applies to an exclusion rule as well.

    Args:
        name(str):    What the text is evaluated under: the proposed name,
            or :data:`rule_quantity`.

        expression(str):    The proposed text.

        current_library(dict):    As for :func:`check_expression`.

    Returns:
        list:    Descriptions of the problems; empty if there are none.
    """

    problems = []

    try:
        referenced = get_expression_names(expression)
    except SyntaxError as error:
        problems.append(
            f"Not a single expression: {error.msg}. Assignments, "
            "statements and imports are not allowed."
        )
        return problems

    # What the library would become if this were saved, which is what the
    # two checks below judge against: an expression may reference itself
    # right up to the cycle check that refuses it.
    new_library = dict(current_library, **{name: expression})

    # Before the unresolvable names, so ``bg_center[i]`` is told what a
    # slot looks like rather than that ``i`` is not a diagnostic. Safe in
    # that order because only names that resolve are asked for an arity.
    try:
        problems.extend(_slot_problems(expression, new_library))
    except PipelineError as error:
        problems.append(str(error))
        return problems

    unresolvable = sorted(
        other
        for other in referenced
        if other != name
        and not is_known_quantity(other)
        and other not in current_library
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
        order_expressions([name], new_library)
    except PipelineError as error:
        problems.append(str(error))

    return problems


#: What an exclusion rule is evaluated under, as one more entry in the
#: library: ``dict(library, **{rule_quantity: rule})``. A rule taking a
#: channel parameter (see :func:`get_channel_parameters`) is bound to each
#: channel being decided for, ``(channel,)``, and excludes per channel;
#: one taking none reads only quoted channels, is bound to ``()``, and its
#: verdict applies to every channel of the image. Photometries likewise
#: (see :func:`get_photometry_parameters`): a rule taking a photometry
#: parameter excludes per photometry. Not a slug, so
#: :func:`check_expression` lets no stored expression take the name, and it
#: reads sensibly in an error message.
rule_quantity = "<exclusion rule>"


def get_quoted_channel_order(quantity, expressions):
    """
    Return the channels *quantity* quotes, in the order its text is written.

    Following the expressions it references where they come, whether read
    bare or through a subscript: ``gcol = sky_color['B', 'R']`` quotes
    ``B`` and then ``R``, and so does anything reading ``gcol``. Each
    channel is listed once, where it is first quoted.

    The order is the text's rather than any other because two things are
    laid out by it. A series whose channels are all quoted binds none of
    its own, and is shown in the first -- its colour, and the frame a click
    on a point opens. And a quoted channel a magfit diagnostic is read in
    gets a column of the series table to choose its photometric reference
    in, and those columns follow the order the expression mentions them.

    Args:
        quantity(str):    A diagnostic, an expression, or the time.

        expressions(dict):    The library, ``{name: expression}``, which
            :func:`order_expressions` has found free of cycles.

    Returns:
        list:    The channels, empty if nothing *quantity* reaches quotes
            one -- as for anything but an expression.

    Raises:
        SyntaxError, PipelineError:    As for :func:`get_indexed_names`.
    """

    if quantity not in expressions:
        return []

    tree = ast.parse(expressions[quantity], mode="eval")
    nested = _nested_reads(tree)
    found = []
    # A subscript starts where its name does and comes first, so its own
    # quoted slots are looked at before the expression it subscripts.
    for node in _in_written_order(tree):
        if isinstance(node, ast.Subscript):
            if id(node) in nested:
                continue
            quoted = [
                channel
                for channel in _get_read(node)[1]
                if isinstance(channel, str)
            ]
        elif isinstance(node, ast.Name):
            quoted = get_quoted_channel_order(node.id, expressions)
        else:
            continue
        for channel in quoted:
            if channel not in found:
                found.append(channel)

    return found


def check_rule(rule, library):
    """
    Return what is wrong with a proposed exclusion rule, as plain strings.

    A rule is judged as an expression would be, apart from having no name
    of its own, and with one more constraint: it takes at most one channel
    slot, which is bound to the channel being decided for, and at most one
    photometry slot, bound to the photometry being decided for. Any other
    channel or photometry it reads is named by quoting it.

    Whether the quoted channels and photometries exist is a question about
    the cameras and the configuration of a project, and is not answered
    here.

    Args:
        rule(str):    The proposed rule text.

        library(dict):    The project's library, ``{name: expression}``.

    Returns:
        list:    Descriptions of the problems; empty if there are none.
    """

    problems = _body_problems(rule_quantity, rule, library)

    try:
        channels = get_channel_parameters(rule)
        photometries = get_photometry_parameters(rule)
    except SyntaxError, PipelineError:
        # Reported above already.
        return problems

    if len(channels) > 1:
        problems.append(
            "An exclusion rule takes at most one channel slot, for the "
            "channel being decided for, not "
            + ", ".join(map(str, channels))
            + ": name any other channel by quoting it, as in "
            "bg_center['G0']."
        )
    if len(photometries) > 1:
        problems.append(
            "An exclusion rule takes at most one photometry slot, for the "
            "photometry being decided for, not "
            + ", ".join(map(str, photometries))
            + ": name any other photometry by quoting it, as in "
            "magfit_residual[0]['ap4']."
        )

    return problems
