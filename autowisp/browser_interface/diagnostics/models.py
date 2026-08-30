"""Models for user-defined diagnostic expressions."""

from django.db import models

from core.models import BuiModelBase


class DiagnosticExpression(BuiModelBase):
    """
    A named expression over the per-image diagnostics.

    Expressions share a flat name space with the ``DiagnosticType`` names of
    whichever project is open, and with ``jd``, so that one can be selected
    for either axis exactly like a recorded diagnostic.  They live here, in
    the browser-interface database, rather than in a project database:
    an expression is a way of looking at data rather than data itself, and
    is worth having available in every project.

    A consequence of that reach is that an expression may name diagnostics
    the open project has never recorded.  That is not an error -- it is
    simply not offered there -- so nothing here constrains the names used;
    resolving them is the business of ``expression_data``.
    """

    name = models.SlugField(
        max_length=100,
        unique=True,
        help_text="Name shown in the diagnostics selectors",
    )
    expression = models.TextField(
        help_text="Python expression over per-image diagnostic names",
    )
    description = models.TextField(
        blank=True,
        help_text="What the expression is for",
    )

    # A Django Meta is declaration only, so it has no methods to count.
    # pylint: disable=too-few-public-methods
    class Meta:
        """Order by name, which is how the management page lists them."""

        ordering = ["name"]

    # pylint: enable=too-few-public-methods

    def __str__(self):
        return str(self.name)
