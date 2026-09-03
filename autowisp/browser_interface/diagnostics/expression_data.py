"""Where the expression library comes from, when it comes from the BUI.

Tier 3 of the expression layer, and the only place outside the views that
knows Django. Tiers 1 and 2 take the library as an argument -- a plain
``{name: expression}`` dictionary -- precisely so that they need not know
whether it was stored by the browser interface, read out of an exported
file by ``run_pipeline``, or written down in a test. This module is the
first of those sources.

It is deliberately thin. Everything one might be tempted to put here --
what an expression means, which order to evaluate a library in, what is
wrong with a proposed one -- belongs to
:mod:`autowisp.diagnostics.expressions`, where it can be tested without a
database of either kind, and is reached from here only by callers that
already have both.
"""

from .models import DiagnosticExpression


def get_expressions():
    """
    Return the stored library as ``{name: expression}``.

    The whole library, not the part that is usable in the open project: an
    expression naming a diagnostic this project never recorded is not an
    error but a thing to leave unoffered. Every expression is *valid*
    everywhere -- the vocabulary is the same in all projects, see
    :mod:`autowisp.diagnostics.diagnostic_types` -- so what varies is only
    whether rows exist, which callers that care establish by counting them.

    Returns:
        dict:    Every stored expression, keyed by name.
    """

    return dict(DiagnosticExpression.objects.values_list("name", "expression"))
