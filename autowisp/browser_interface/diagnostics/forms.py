"""The form behind the diagnostic expression management page.

The library lives in the project database rather than in a Django model,
so this is a plain ``Form``: Django still does the per-field checks and
error plumbing, while the one check a model used to supply -- that the name
is not already taken -- is made here against the library.

What Django cannot know is whether the expression *means* anything.  That
is :func:`~autowisp.diagnostics.expressions.check_expression`, which
returns its complaints as plain strings rather than raising so that it
stays usable with no Django at all.  Turning those strings into
``ValidationError``\\ s is this module's whole reason to exist, and the only
place that adaptation happens.
"""

from django import forms

from autowisp.diagnostics.expressions import (
    check_expression,
    get_bare_aggregates,
)


class DiagnosticExpressionForm(forms.Form):
    """
    Validate one proposed expression against the library it would join.

    Validity does not depend on what the project has recorded -- see
    :mod:`autowisp.diagnostics.diagnostic_types` -- so only the library is
    consulted; whether the diagnostics an expression needs exist is a
    separate question, answered by counting rows elsewhere.
    """

    # Selected through ``image/<slug:x>/vs/<slug:y>``, so a name outside the
    # slug charset could be stored but never plotted.
    name = forms.SlugField(
        max_length=100,
        help_text="Name shown in the diagnostics selectors",
    )
    # Neither has a useful length limit, but both are written on one line,
    # so a textarea would be a misleading amount of room.
    expression = forms.CharField(
        widget=forms.TextInput(),
        help_text="Python expression over per-image diagnostic names",
    )
    description = forms.CharField(
        required=False,
        widget=forms.TextInput(),
        help_text="What the expression is for",
    )

    def __init__(self, *args, expressions=None, replacing=None, **kwargs):
        """
        Args:
            expressions(dict):    The library, ``{name: expression}``, this
                one would join.  The view has it and the form does not, so
                it arrives as a keyword argument.

            replacing(str or None):    The name of the stored expression
                being edited, or ``None`` when adding.  An entry of that
                name is treated as the one being replaced rather than as a
                conflict, so an edit can pass the library unchanged.
        """

        super().__init__(*args, **kwargs)
        self.expressions = dict(expressions or {})
        self.replacing = replacing or None

        #: The NaN-propagating aggregates the accepted expression calls,
        #: for the view to warn about.  Not an error: a deliberate
        #: ``median`` is a legitimate thing to write, it is merely almost
        #: never what was meant.
        self.bare_aggregates = set()

    def clean(self):
        """Report what ``check_expression`` says, against the field at fault."""

        cleaned_data = super().clean()
        name = cleaned_data.get("name")
        expression = cleaned_data.get("expression")
        if not name or not expression:
            # A missing field has already failed; complaining about the
            # pair as well would only repeat that.
            return cleaned_data

        if name != self.replacing and name in self.expressions:
            self.add_error("name", f"An expression named {name} exists.")
            return cleaned_data

        # check_expression reports on the pair, but its complaints have to
        # land on the field that caused them.  The problems a name has on
        # its own are exactly those it still has beside an expression that
        # references nothing, which sorts them without matching on the
        # message text.
        name_problems = check_expression(name, "0", {})
        for problem in check_expression(name, expression, self.expressions):
            self.add_error(
                "name" if problem in name_problems else "expression", problem
            )

        if not self.errors:
            self.bare_aggregates = get_bare_aggregates(expression)

        return cleaned_data
