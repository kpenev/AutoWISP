"""Define the URL paths used by the diagnostics BUI app."""

from django.urls import path, register_converter

from . import views

app_name = "diagnostics"


class SlugListConverter:
    """One or more slugs, comma separated: the y quantities of a page.

    A plot may draw several quantities against one x, and the address bar
    says which -- so that a page can be bookmarked, reloaded or sent to
    somebody as what is on the screen. ``image/jd/vs/bg_center,smooth_bg``
    is two sections against the time.

    The order is the sections' order, which the page keeps meaningful: it
    decides which default marker each section takes, and the order of the
    legend.
    """

    regex = r"[-a-zA-Z0-9_]+(?:,[-a-zA-Z0-9_]+)*"

    def to_python(self, value):
        """Return the names in order, keeping the first of any repeat.

        A repeated name would be a second section drawing exactly what an
        earlier one draws, which is not worth refusing the whole address
        over: the sections are what the user asked for, minus the one
        that would have been a duplicate.
        """

        return list(dict.fromkeys(value.split(",")))

    def to_url(self, value):
        """Return the path segment naming *value*.

        A string passes through whole, because the selector bar reverses
        this route with a placeholder standing in for the names and
        substitutes the real ones in the browser -- so what arrives here
        is sometimes a marker rather than a list.
        """

        if isinstance(value, str):
            return value

        return ",".join(dict.fromkeys(value))


# Registered as `slug_list` rather than `slugs`: a route reads
# `<slug:x_quantity>` beside it, and names a letter apart are the kind
# that get misread rather than noticed.
register_converter(SlugListConverter, "slug_list")

urlpatterns = [
    path(
        "detrending/<slug:step>/<slug:imtype>",
        views.display_detrending_diagnostics,
        name="diagnostics",
    ),
    path(
        "detrending/<slug:step>/<slug:imtype>/<slug:master_ids>",
        views.display_detrending_diagnostics,
        name="diagnostics",
    ),
    path(
        "display_detrending_diagnostics",
        views.display_detrending_diagnostics,
        name="display_detrending_diagnostics",
    ),
    path(
        "refresh_detrending_diagnostics",
        views.refresh_detrending_diagnostics,
        name="refresh_diagnostics",
    ),
    path(
        "update_detrending_diagnostics_plot",
        views.update_detrending_diagnostics_plot,
        name="update_diagnostics_plot",
    ),
    path(
        "download_detrending_diagnostics_plot",
        views.download_detrending_diagnostics_plot,
        name="download_diagnostics_plot",
    ),
    path(
        "image/<slug:x_quantity>/vs/<slug_list:y_quantities>",
        views.display_diagnostics,
        name="display_diagnostics",
    ),
    # One section, fetched on its own when the y selector adds one.
    path(
        "image/<slug:x_quantity>/section/<slug:y_quantity>",
        views.diagnostics_section,
        name="diagnostics_section",
    ),
    # No y on either: each posted row names the quantity it draws, so all
    # these two need naming is the x every row shares.
    path(
        "image/<slug:x_quantity>/update_plot",
        views.update_diagnostics_plot,
        name="update_diagnostics_plot",
    ),
    path(
        "image/<slug:x_quantity>/download_plot",
        views.download_diagnostics_plot,
        name="download_diagnostics_plot",
    ),
    path(
        "expressions",
        views.list_expressions,
        name="list_expressions",
    ),
    # Under `edit/` rather than `expressions/<name>`, so that an
    # expression legitimately named "save" or "delete" cannot collide with
    # a literal route below.
    path(
        "expressions/edit/<slug:name>",
        views.list_expressions,
        name="edit_expression",
    ),
    path(
        "expressions/save",
        views.save_expression,
        name="save_expression",
    ),
    path(
        "expressions/delete",
        views.delete_expressions,
        name="delete_expressions",
    ),
    path(
        "expressions/export",
        views.export_expressions,
        name="export_expressions",
    ),
    path(
        "expressions/import",
        views.import_expressions,
        name="import_expressions",
    ),
    path(
        "expressions/import/confirm",
        views.confirm_import_expressions,
        name="confirm_import_expressions",
    ),
    path(
        "preview_calibrated/<int:image_id>/<slug:color_channel>",
        views.preview_calibrated_image,
        name="preview_calibrated_image",
    ),
    path(
        "preview_calibrated/<int:image_id>/<slug:color_channel>"
        "/overlay/<slug:overlay_type>",
        views.get_image_overlay,
        name="get_image_overlay",
    ),
]
