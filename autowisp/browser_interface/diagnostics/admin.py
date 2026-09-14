"""Admin registration for the diagnostics app."""

from django.contrib import admin

from .models import DiagnosticExpression


@admin.register(DiagnosticExpression)
class DiagnosticExpressionAdmin(admin.ModelAdmin):
    """Inspect and repair expressions without going through the app.

    The management page is where expressions are meant to be edited, since
    it validates them against the open project.  This is the fallback for
    when that is what needs fixing.
    """

    list_display = ("name", "expression", "modified")
    search_fields = ("name", "expression", "description")
    readonly_fields = ("created", "modified")
