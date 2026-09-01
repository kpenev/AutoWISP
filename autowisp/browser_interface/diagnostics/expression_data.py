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

    The whole library, not the part that resolves in the open project: an
    expression naming a diagnostic this project never recorded is not an
    error but a thing to leave unoffered, and deciding that needs the
    project's names, which this module does not have. Callers that care ask
    :func:`autowisp.diagnostics.expression_series.get_known_names` and let
    tier 1 do the resolving.

    Returns:
        dict:    Every stored expression, keyed by name.
    """

    return dict(DiagnosticExpression.objects.values_list("name", "expression"))
