"""The form behind the diagnostic expression management page.

A ``ModelForm`` rather than hand-written POST handling, which is a new
pattern in this interface and a deliberate one.  The name charset, the
uniqueness of the name, the create-versus-update branch and the per-field
error plumbing are all things Django already does correctly, and each one
written out by hand would be another thing to keep right.

What Django cannot know is whether the expression *means* anything.  That
is :func:`~autowisp.diagnostics.expressions.check_expression`, which
returns its complaints as plain strings rather than raising so that it
stays usable with no Django at all -- import, and one day the command
line, reach it by the same path.  Turning those strings into a
``ValidationError`` is this module's whole reason to exist, and the only
place that adaptation happens.
"""

from django import forms

from autowisp.diagnostics.expressions import (
    check_expression,
    get_bare_aggregates,
)

from .models import DiagnosticExpression


class DiagnosticExpressionForm(forms.ModelForm):
    """
    Validate one proposed expression against the library it would join.

    No project is involved.  An expression is valid or not in every project
    alike -- see :mod:`autowisp.diagnostics.diagnostic_types` -- and
    whether the open project has *recorded* what it needs is a separate
    question, answered by counting rows elsewhere.  So this form works with
    no project open, which is part of what makes one global library
    coherent.
    """

    # A ModelForm is configuration plus one hook; the base class supplies
    # the rest of the interface.
    # pylint: disable=too-few-public-methods
    class Meta:
        """The three user-editable columns of the model."""

        model = DiagnosticExpression
        fields = ["name", "expression", "description"]
        # Both are ``TextField`` because neither has a useful length limit,
        # but both are written on one line, so a textarea would be a
        # misleading amount of room.
        widgets = {
            "expression": forms.TextInput(),
            "description": forms.TextInput(),
        }

    # pylint: enable=too-few-public-methods

    def __init__(self, *args, expressions=None, **kwargs):
        """
        Args:
            expressions(dict):    The library, ``{name: expression}``, this
                one would join.  The view has it and the form does not, so
                it arrives as a keyword argument.  An entry of the same
                name is treated as the one being replaced rather than as a
                conflict, so an edit can pass the library unchanged.
        """

        super().__init__(*args, **kwargs)
        self.expressions = dict(expressions or {})

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
