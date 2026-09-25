"""Define the diagnostic expression library table."""

from sqlalchemy import Column, String, Text

from autowisp.database.data_model.base import DataModelBase

__all__ = ["DiagnosticExpression"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class DiagnosticExpression(DataModelBase):
    """
    A named expression over the per-image diagnostics.

    Expressions share a flat name space with the ``DiagnosticType`` names
    and with ``jd``, so that one can be selected for either axis of a
    diagnostics plot exactly like a recorded diagnostic, and referenced by
    the exclusion rules that decide which images a fit leaves out.

    The library belongs to the project rather than to the browser interface:
    once expressions drive processing they are part of a project's
    configuration, and a library shared between projects would let an edit
    made for one silently change what an old one excludes and plots.
    Projects share expressions by export and import instead.

    Nothing here resolves the names an expression uses. One naming a
    diagnostic the project has never recorded is not an error -- it is
    simply not offered -- and what an expression means is the business of
    :mod:`autowisp.diagnostics.expressions`.
    """

    __tablename__ = "diagnostic_expression"

    name = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="Name shown in the diagnostics selectors and used by other "
        "expressions and exclusion rules to reference this one.",
    )
    expression = Column(
        Text,
        nullable=False,
        doc="Python expression over per-image diagnostic names.",
    )
    description = Column(
        Text,
        nullable=False,
        doc="What the expression is for; empty if left blank.",
    )

    def __repr__(self):
        return f"({self.id}) {self.name} = {self.expression}"


# pylint: enable=too-few-public-methods
