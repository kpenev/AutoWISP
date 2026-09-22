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
    Return what one expression reads, and in which channel slots.

    A slot is written as a subscript -- ``bg_center[0]``, or
    ``sky_color[0,1]`` for a quantity taking two -- and the numbers are
    that expression's own formal parameters, bound to real channels only
    when something is plotted. So this reports the *shape* of what the
    text asks for, and says nothing about channels.

    Args:
        expression(str):    The expression text.

    Returns:
        list:    ``(name, slots)`` pairs in the order they appear, *slots*
            always a tuple however many were written. Repeats are kept:
            ``sky_color[0,1] - sky_color[1,2]`` reads one name at two
            different slot pairs, which is the whole point of the
            parameters being formal.

    Raises:
        SyntaxError:    As for :func:`get_expression_names`.

        PipelineError:    If a subscript is not a plain name indexed by
            integer literals. Anything else -- ``bg_center[i]``,
            ``bg_center[1:2]``, ``sky_color[1][2]`` -- cannot name a
            channel slot, and saying so here is clearer than letting it
            fail as an unresolvable name later.
    """

    references = []
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            raise PipelineError(
                f"{ast.unparse(node)!r} does not name a channel slot: only "
                "a diagnostic or an expression can take one.",
                details={"subscript": ast.unparse(node)},
            )
        try:
            slots = ast.literal_eval(node.slice)
        except ValueError:
            # Non-literal channel indices is a sub-case of what is handled
            # below.
            slots = None
        if not isinstance(slots, tuple):
            slots = (slots,)
        if not slots or not all(
            isinstance(slot, int) and not isinstance(slot, bool)
            for slot in slots
        ):
            raise PipelineError(
                f"{ast.unparse(node)!r} does not name a channel slot: a "
                "slot is written as a whole number, as in bg_center[0].",
                details={"subscript": ast.unparse(node)},
            )
        references.append((node.value.id, slots))

    return references


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


def get_expression_parameters(expression):
    """
    Return the channel slots one expression takes, in order.

    Derived from the body rather than declared: the parameters *are* the
    slot numbers it mentions, so there is nothing to keep in step and
    nothing extra to store. ``bg_center[0] / bg_center[1]`` takes ``(0,
    1)``; ``sky_color[0,1] - sky_color[1,2]`` takes ``(0, 1, 2)``.

    Sorted, which is what makes the order well defined when a body writes
    its slots out of sequence -- and the order matters, since a reference's
    arguments are matched onto these positionally.

    Args:
        expression(str):    The expression text.

    Returns:
        tuple:    The distinct slot numbers, ascending. Empty for an
            expression that reads no channel at all.

    Raises:
        SyntaxError, PipelineError:    As for
            :func:`get_indexed_names`.
    """

    return tuple(
        sorted(
            {
                slot
                for _, slots in get_indexed_names(expression)
                for slot in slots
            }
        )
    )


def get_quantity_arity(name, expressions):
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
        return len(get_expression_parameters(expressions[name]))
    if name == time_quantity:
        return 0
    if is_diagnostic(name):
        return 1
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

    def __init__(self, name, shared, *, definition=None, computed=None):
        """
        Args:
            name(str):    What the expressions call it, used for error
                messages only -- nothing resolves by it.

            shared(tuple):    The binding stack and evaluator this
                evaluation's lookups share, from :meth:`library`.

            definition(tuple):    The expression text and its parameters,
                or ``None`` for a diagnostic, which has neither because it
                is never evaluated.

            computed(dict):    ``channels -> array`` known in advance: the
                whole of a diagnostic, and empty for an expression.
        """

        self._name = name
        self._stack, self._evaluator = shared
        self._text, self._parameters = definition or (None, ())
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

            values(dict):    ``{name: {channels: array}}``, as fetched for
                one series, keyed exactly as
                :func:`get_needed_values` asked.

        Returns:
            dict:    A lookup per name, sharing one evaluation's state.
        """

        shared = ([], Evaluator({}))
        lookups = {
            name: cls(name, shared, computed=by_channels)
            for name, by_channels in values.items()
        }
        lookups.update(
            {
                name: cls(
                    name,
                    shared,
                    definition=(text, get_expression_parameters(text)),
                )
                for name, text in expressions.items()
            }
        )

        symtable = shared[1].symtable
        symtable.update(lookups)

        # A quantity binding no channels -- the time, and any expression
        # over it alone -- is written without a subscript, there being
        # nothing to subscript it with. A lookup resolves on
        # ``__getitem__`` and a bare name never calls one, so such a
        # quantity has to be its *values* here, or it would reach the
        # arithmetic as the object itself. The time first, then the
        # expressions over it, each after what it reads.
        for name, by_channels in values.items():
            if () in by_channels:
                symtable[name] = by_channels[()]

        channel_free = [
            name
            for name, text in expressions.items()
            if not get_expression_parameters(text)
        ]
        for name in _evaluation_order(channel_free, expressions):
            symtable[name] = lookups[name].at(())

        return lookups

    def __getitem__(self, slots):
        """
        Resolve ``name[slots]`` as the body being evaluated means it.

        Python passes ``1`` for ``x[1]`` and ``(1, 2)`` for ``x[1,2]``, so
        the shapes take care of themselves. The binding read is the top of
        the stack, which is always the body asking: a nested evaluation
        finishes in here before the outer body's next operand is touched.
        """

        if not isinstance(slots, tuple):
            slots = (slots,)
        binding = self._stack[-1]
        return self.at(tuple(binding[slot] for slot in slots))

    def at(self, channels):
        """
        Return this quantity with its parameters bound to *channels*.

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
            channels(tuple):    One channel per parameter, in order.

        Returns:
            numpy.ndarray:    The values, over the canonical image list.

        Raises:
            PipelineError:    If a diagnostic was not fetched for this
                channel.
        """

        if channels not in self._computed:
            if self._text is None:
                raise PipelineError(
                    f"No values supplied for {self._name} in "
                    + ", ".join(channels)
                    + ".",
                    details={"missing": [self._name], "channels": channels},
                )
            self._stack.append(dict(zip(self._parameters, channels)))
            try:
                self._computed[channels] = self._evaluator(self._text)
            finally:
                self._stack.pop()
        return self._computed[channels]


def _visit_needed(name, channels, expressions, needed):
    """Add what *name* bound to *channels* reads, recursively."""

    expected = get_quantity_arity(name, expressions)
    if len(channels) != expected:
        raise PipelineError(
            f"{name} takes {expected} channel(s), not {len(channels)}.",
            details={"quantity": name, "channels": list(channels)},
        )

    if name not in expressions:
        # A diagnostic, or the time -- either way a leaf, and either way
        # named by the channels it is read in, which for the time is none.
        needed.setdefault(name, set()).add(channels)
        return

    text = expressions[name]
    binding = dict(zip(get_expression_parameters(text), channels))
    for referenced, slots in get_indexed_names(text):
        _visit_needed(
            referenced,
            tuple(binding[slot] for slot in slots),
            expressions,
            needed,
        )


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
    diagnostics.

    Args:
        wanted(dict):    ``{quantity: set of channel tuples}``, each tuple
            holding one channel per parameter of that quantity. A *set*
            because one quantity may be wanted at two bindings at once:
            ``bg_center`` in R against ``bg_center`` in B is a plot of a
            diagnostic between channels.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        dict:    ``{name: set of channel tuples}`` -- the same shape it
            takes, which is what it means: what is wanted, resolved into
            what must be read. One entry per diagnostic, holding every
            combination it is read in, plus :data:`time_quantity` with the
            empty tuple where an expression reads the time.

    Raises:
        PipelineError:    On a reference cycle, on a name that resolves to
            nothing, or on a binding of the wrong length for what it binds.
    """

    # Refuses a cycle and an unresolvable name, so the walk below cannot
    # recurse for ever and neither it nor the evaluation need check again.
    _, channel_free = order_expressions(list(wanted), expressions)

    needed = {}

    # The time is the one quantity read without a subscript, so the walk
    # below never reaches it however deep it lies: ``bg_center[0] * night``
    # reads it only through ``night``. Taking no channel is also why no
    # walk is needed -- order_expressions already reports it, having
    # flattened bare and subscripted names alike -- and why nothing else
    # can hide behind a bare reference, reading a diagnostic taking a
    # subscript that would give its expression a parameter.
    if time_quantity in channel_free:
        needed[time_quantity] = {()}

    for quantity, bindings in wanted.items():
        for channels in bindings:
            _visit_needed(quantity, tuple(channels), expressions, needed)

    return needed


def _canonical_length(values):
    """Return the length of the image list *values* are padded onto."""

    for by_channels in values.values():
        for array in by_channels.values():
            return numpy.size(array)
    return 0


def evaluate_quantities(wanted, expressions, values):
    """
    Evaluate the quantities of one series, each bound to its channels.

    Args:
        wanted(dict):    ``{quantity: set of channel tuples}``, as the
            table bound them and as :func:`get_needed_values` takes them.
            Asking for both axes at once is what lets an instantiation
            they share be computed once.

        expressions(dict):    The library, ``{name: expression}``.

        values(dict):    ``{name: {channels: array}}``, every array on one
            canonical image list, holding what
            :func:`get_needed_values` asked for and keyed the same way.

    Returns:
        dict:    ``{quantity: {channels: array}}``, all of the same
            length -- the shape *values* arrives in, being the same kind
            of thing: a quantity read in the channels asked for.

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
            tuple(channels): _as_series(
                lookups[quantity].at(tuple(channels)), count
            )
            for channels in bindings
        }
        for quantity, bindings in wanted.items()
    }


def _spell_slots(count):
    """Return *count* channel slots, spelled for a message."""

    return "1 channel slot" if count == 1 else f"{count} channel slots"


def _slot_problems(expression, library):
    """
    Return what is wrong with how *expression* reads its quantities.

    One rule: a quantity is read with exactly as many channel slots as it
    takes, and one taking none is read bare. The first half is what makes
    a reference's arguments match a definition's parameters positionally;
    the second is what lets :func:`get_needed_values` and
    :meth:`QuantityLookUp.library` treat a bare name as a value and still
    know they have seen everything.

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

    for referenced, slots in get_indexed_names(expression):
        if not is_quantity(referenced):
            continue
        written = f"{referenced}[{', '.join(str(slot) for slot in slots)}]"
        arity = get_quantity_arity(referenced, library)
        if arity == 0:
            problems.append(
                f"{referenced} takes no channel slot, being one value per "
                f"image: write {referenced} rather than {written}."
            )
        elif arity != len(slots):
            problems.append(
                f"{referenced} takes {_spell_slots(arity)}, not "
                f"{len(slots)}: {written}."
            )

    for referenced in sorted(_get_bare_names(expression)):
        if not is_quantity(referenced):
            continue
        arity = get_quantity_arity(referenced, library)
        if arity:
            sample = ", ".join(str(slot) for slot in range(arity))
            problems.append(
                f"{referenced} takes {_spell_slots(arity)}, so it cannot be "
                f"read on its own: write {referenced}[{sample}]."
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
    means the same thing everywhere -- which is what lets one library be
    shared by every project.

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
