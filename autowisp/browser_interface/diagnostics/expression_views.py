"""Views for defining, listing and moving diagnostic expressions.

The library is global -- one set of expressions shared by every project --
because an expression is a way of *looking* at data rather than data
itself.  That is also why this page works with no project open: validity
does not depend on one (see
:mod:`autowisp.diagnostics.diagnostic_types`), and the only thing that
does -- whether the diagnostics an expression needs are recorded here --
is reported as availability rather than as brokenness.  Today's single
status conflates "meaningless" with "nothing recorded yet"; these are not
the same complaint and do not read as one here.
"""

import json
from io import StringIO

from django.contrib import messages
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from autowisp.database.interface import start_db_session
from autowisp.diagnostics.diagnostic_types import time_quantity
from autowisp.diagnostics.expressions import (
    check_expression,
    get_expression_dependents,
    get_expression_names,
    order_expressions,
    rename_references,
)

from .expression_data import get_expressions
from .forms import DiagnosticExpressionForm
from .image_diagnostics_views import get_recorded_diagnostics
from .models import DiagnosticExpression

#: Marks an export file as ours and says which shape it is in.  A file
#: without the key, or carrying a version this code does not know, is
#: refused rather than guessed at.
_format_key = "autowisp_diagnostic_expressions"

#: The only export shape there has been so far.  It is meant to become the
#: configuration format a command-line run is pointed at, which is why it
#: is versioned and why a subset export pulls in what it depends on.
_format_version = 1


def _references(name, expressions):
    """
    Return the library entries one expression names directly.

    Tolerates an unparseable expression by reporting no references: what is
    wrong with it is :func:`check_expression`'s to say, and a page listing
    expressions must not fail to render because one of them is broken.

    Args:
        name(str):    The expression to look at.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        set:    The names of the expressions it references.  Diagnostics
            and functions are left out; they are not links to follow.
    """

    try:
        referenced = get_expression_names(expressions[name])
    except SyntaxError:
        return set()

    return {other for other in referenced if other in expressions} - {name}


def _reachable(names, expressions):
    """
    Return *names* and every expression they reach, transitively.

    Deliberately not :func:`order_expressions`, which refuses a library
    with a cycle or an unresolvable name in it: exporting is one way a user
    moves expressions somewhere they can be repaired, so it has to work on
    a library that does not validate.

    Args:
        names(iterable):    The selected expression names.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        list:    The closure, alphabetically.
    """

    reached = set()
    pending = [name for name in names if name in expressions]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(_references(name, expressions))

    return sorted(reached)


def describe_expression(name, expressions, recorded):
    """
    Return one row of the management table.

    The two columns that can complain say different things, and the
    distinction is the point of the page.  *Problems* is whether the
    expression means anything, which is the same answer in every project,
    while *missing* is whether this one has recorded what it needs, which
    is not.  An expression naming a diagnostic this project never produced
    is unavailable here and perfectly sound; only a typo is broken.

    Args:
        name(str):    The expression to describe.

        expressions(dict):    The library, ``{name: expression}``.

        recorded(set):    The diagnostic names in use in the open project,
            or ``None`` if no project is open.

    Returns:
        dict:    The fields ``diagnostic_expressions.html`` renders.
    """

    problems = check_expression(name, expressions[name], expressions)

    missing = None
    if not problems and recorded is not None:
        _, needed = order_expressions([name], expressions)
        # jd is known for every image of the canonical list, so it never
        # counts against availability.
        missing = sorted(needed - {time_quantity} - recorded)

    return {
        "name": name,
        "expression": expressions[name],
        "depends_on": sorted(_references(name, expressions)),
        "problems": problems,
        "missing": missing,
    }


def _render_list(request, form, edit_name=""):
    """
    Render the management page around *form*, bound or blank.

    The library comes from the form rather than being fetched again: the
    form was built with the library this request is about, and on a failed
    save the table must show what is stored rather than what was typed.

    Args:
        request:    The Django request.

        form(DiagnosticExpressionForm):    The form to render above the
            table, blank when adding and filled when editing.

        edit_name(str):    The name of the row being replaced, which the
            template posts back so a rename stays an edit.  Empty when
            adding.
    """

    expressions = form.expressions

    # Availability is the one thing here that needs a project; validity is
    # not, so with none open the page still lists and still validates, and
    # simply says nothing about what has been recorded.
    recorded = None
    if request.session.get("project_home"):
        with start_db_session() as db_session:
            recorded = set(get_recorded_diagnostics(db_session))

    # The description is the one stored column no rule is derived from, so
    # it is merged in here rather than threaded through the library, which
    # is a name-to-expression mapping everywhere else in the feature.
    # pylint: disable=no-member
    descriptions = dict(
        DiagnosticExpression.objects.values_list("name", "description")
    )
    # pylint: enable=no-member

    return render(
        request,
        "diagnostics/diagnostic_expressions.html",
        {
            "form": form,
            "edit_name": edit_name,
            "have_project": recorded is not None,
            "expression_rows": [
                dict(
                    describe_expression(name, expressions, recorded),
                    description=descriptions.get(name, ""),
                )
                for name in sorted(expressions)
            ],
        },
    )


def list_expressions(request, name=None):
    """
    Show the library, with a form for adding to it or editing one row.

    Editing is a URL rather than a click that fills the form in place, so
    that it needs no JavaScript, survives a refresh, and can be linked to.
    ``Http404`` for a name that is not there is the right answer to a stale
    link, and Django's own -- the error middleware deliberately leaves it
    alone.

    Args:
        request:    The Django request.

        name(str):    The expression to open for editing, or ``None`` to
            show the blank form.
    """

    instance = (
        None
        if name is None
        else get_object_or_404(DiagnosticExpression, name=name)
    )

    return _render_list(
        request,
        DiagnosticExpressionForm(
            instance=instance, expressions=get_expressions()
        ),
        edit_name=name or "",
    )


def _carry_dependents_through_rename(old_name, new_name, expressions):
    """
    Rewrite everything referencing *old_name* to reference *new_name*.

    A rename would otherwise orphan its dependents -- the delete guard's
    hazard reached from the other side -- and refusing it, as deletion is
    refused, is not a workable answer: unlike a delete there is no gesture
    that makes it legal, because pointing a dependent at the new name will
    not validate while that name does not yet exist.  So the rename carries
    them with it, and the caller says which ones moved.

    Args:
        old_name(str):    The name as it was.

        new_name(str):    The name as it now is.

        expressions(dict):    The library as it was *before* the rename,
            which is what the dependents are read from.

    Returns:
        list:    The names updated, alphabetically; empty if nothing
            referenced *old_name*.
    """

    updated = []
    for dependent in sorted(get_expression_dependents(old_name, expressions)):
        # QuerySet.update rather than save(): `modified` is maintained by a
        # database trigger precisely so that it survives this, see
        # core.models.BuiModelBase.
        # pylint: disable=no-member
        DiagnosticExpression.objects.filter(name=dependent).update(
            expression=rename_references(
                expressions[dependent], old_name, new_name
            )
        )
        # pylint: enable=no-member
        updated.append(dependent)

    return updated


def save_expression(request):
    """
    Create or update one expression.

    Editing is keyed by ``edit_name`` rather than by primary key, so that
    the page is driven entirely by the names it displays and a rename is an
    edit rather than a delete followed by a create.
    """

    assert request.method == "POST"

    expressions = get_expressions()
    edit_name = request.POST.get("edit_name", "")
    # pylint: disable=no-member
    instance = DiagnosticExpression.objects.filter(name=edit_name).first()
    # pylint: enable=no-member

    form = DiagnosticExpressionForm(
        request.POST, instance=instance, expressions=expressions
    )
    if not form.is_valid():
        # Re-rendered rather than redirected, so the complaints stay
        # attached to the fields that caused them -- and still in edit
        # mode, so correcting one does not silently create a second row.
        return _render_list(request, form, edit_name=edit_name)

    with transaction.atomic():
        expression = form.save()
        updated = (
            _carry_dependents_through_rename(
                edit_name, expression.name, expressions
            )
            if edit_name and edit_name != expression.name
            else []
        )

    if updated:
        messages.info(
            request,
            f"Renamed {edit_name} to {expression.name}, and updated "
            + ", ".join(updated)
            + " to match.",
        )

    for aggregate in sorted(form.bare_aggregates):
        messages.warning(
            request,
            f"{expression.name} calls {aggregate}(), which goes NaN as soon "
            f"as one image of a series lacks a diagnostic. Did you mean "
            f"nan{aggregate}()?",
        )

    return redirect("diagnostics:list_expressions")


def delete_expressions(request):
    """
    Delete the checked expressions, unless something still needs them.

    Dependents are judged against what will *remain*, so a whole chain may
    be deleted together while the bottom of it may not be deleted alone.
    """

    assert request.method == "POST"

    selected = set(request.POST.getlist("expression_names"))
    if not selected:
        messages.info(request, "Nothing was ticked, so nothing was deleted.")
        return redirect("diagnostics:list_expressions")

    expressions = get_expressions()

    blocked = {}
    for name in selected:
        dependents = get_expression_dependents(name, expressions) - selected
        if dependents:
            blocked[name] = sorted(dependents)

    if blocked:
        messages.error(
            request,
            "Nothing was deleted: "
            + "; ".join(
                f"{name} is used by {', '.join(dependents)}"
                for name, dependents in sorted(blocked.items())
            )
            + ".",
        )
    else:
        # pylint: disable=no-member
        DiagnosticExpression.objects.filter(name__in=selected).delete()
        # pylint: enable=no-member

    return redirect("diagnostics:list_expressions")


def export_expressions(request):
    """
    Download the library, or a selection of it, as JSON.

    A selection is extended with everything it depends on, since a file
    naming an expression it does not carry cannot be imported anywhere
    else -- nor read by a command-line run, which is what this format is
    ultimately for.

    A POST rather than a link, because the selection is the same set of
    checkboxes the delete button reads: one form serves both, with
    ``formaction`` sending each button here or there.  Nothing is lost by
    it -- the URL of a download whose content depends on what is ticked is
    not worth bookmarking.
    """

    assert request.method == "POST"

    expressions = get_expressions()
    selected = request.POST.getlist("expression_names")
    names = (
        _reachable(selected, expressions) if selected else sorted(expressions)
    )

    # pylint: disable=no-member
    rows = DiagnosticExpression.objects.filter(name__in=names).values(
        "name", "expression", "description"
    )
    # pylint: enable=no-member

    with StringIO() as export_stream:
        json.dump(
            {
                _format_key: _format_version,
                "expressions": sorted(rows, key=lambda row: row["name"]),
            },
            export_stream,
            indent=4,
        )
        return HttpResponse(
            export_stream.getvalue().encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Content-Disposition": (
                    'attachment; filename="diagnostic_expressions.json"'
                ),
            },
        )


def _staged_expressions(entries):
    """
    Return ``{name: fields}`` for the entries of an import file.

    Args:
        entries:    Whatever the file's ``expressions`` key held, which is
            not to be trusted to be a list of anything in particular.

    Returns:
        dict:    ``{name: {"expression": …, "description": …}}``, ready to
            be laid over the stored library.

    Raises:
        ValueError:    If the entries are not objects carrying a name and
            an expression.
    """

    staged = {}
    for entry in entries:
        if not isinstance(entry, dict) or not {"name", "expression"} <= set(
            entry
        ):
            raise ValueError(
                "every entry must be an object with a name and an expression"
            )
        staged[str(entry["name"])] = {
            "expression": str(entry["expression"]),
            "description": str(entry.get("description", "")),
        }

    return staged


def import_expressions(request):
    """
    Add expressions from a JSON file written by :func:`export_expressions`.

    The whole file is staged over the stored library before anything is
    checked, so that expressions referencing each other validate whatever
    order they appear in.  Entries are then written one at a time, and one
    that does not validate is reported rather than aborting the rest.
    """

    assert request.method == "POST"

    try:
        document = json.load(request.FILES["expressions-import"])
        if document.get(_format_key) != _format_version:
            raise ValueError(
                f"expected {_format_key!r} to be {_format_version}, which is "
                "what this version of AutoWISP writes"
            )
        staged = _staged_expressions(document.get("expressions", []))
    except (ValueError, UnicodeDecodeError, AttributeError) as error:
        messages.error(request, f"Not a diagnostic expression file: {error}.")
        return redirect("diagnostics:list_expressions")

    overwrite = bool(request.POST.get("overwrite"))
    stored = get_expressions()
    # What each entry is checked against: the file laid over the library,
    # so an intra-file reference resolves whether or not its target has
    # been written yet.
    library = dict(
        stored,
        **{name: entry["expression"] for name, entry in staged.items()},
    )

    added, updated, skipped, refused = 0, 0, [], []
    for name, entry in staged.items():
        problems = check_expression(name, entry["expression"], library)
        if problems:
            refused.append(f"{name} ({' '.join(problems)})")
        elif name in stored and not overwrite:
            skipped.append(name)
        else:
            # pylint: disable=no-member
            _, created = DiagnosticExpression.objects.update_or_create(
                name=name, defaults=entry
            )
            # pylint: enable=no-member
            if created:
                added += 1
            else:
                updated += 1

    # Only what actually happened: "imported 0, updated 0" beside a list of
    # what was kept instead says the same thing twice, the second time
    # wrongly.
    if added or updated:
        messages.info(
            request, f"Imported {added} expression(s), updated {updated}."
        )
    elif not staged:
        messages.info(request, "That file listed no expressions.")

    if skipped:
        messages.warning(
            request,
            "Kept the stored version of "
            + ", ".join(sorted(skipped))
            + "; tick 'overwrite existing' to replace them instead.",
        )
    if refused:
        messages.error(request, "Refused " + "; ".join(sorted(refused)) + ".")

    return redirect("diagnostics:list_expressions")
