"""The catalogue of per-image diagnostics AutoWISP defines.

The vocabulary of diagnostic names, kept apart from the database that
stores values against them.  Two quite different things want it:

* :func:`autowisp.database.initialize_database._init_diagnostic_types`
  seeds a new project's ``diagnostic_type`` table from it, which is why
  the descriptions live here too rather than only the names.
* :func:`autowisp.diagnostics.expressions.check_expression` needs to know
  what a name in an expression may mean -- and needs to know it *without*
  a project, since a diagnostic expression is defined once and used in
  every project.

That second caller is the reason this module exists.  The catalogue is a
static list: whatever seeds one project seeds them all, so requiring an
open database to discover it was a false dependency.  The database module
is now a consumer of the vocabulary rather than its owner.

The one thing genuinely not knowable in advance is the ``pixel_q*``
family, created at run time by ``calibrate`` rather than seeded.  Those
are a *pattern* rather than a list, and no amount of enumeration would
have captured them -- hence :func:`is_quantile_diagnostic`, which both
the code creating those rows and the code validating expressions against
them ask, so the two cannot drift apart.

Everything here is exposed as a cached function returning an immutable
value rather than as a module-level constant, so that a caller cannot
mutate the catalogue out from under every other caller in the process.
"""

import functools
import re
from types import MappingProxyType


@functools.lru_cache(maxsize=1)
def standard_diagnostic_types():
    """
    Return every diagnostic seeded into a new project.

    A mapping rather than a sequence of pairs because that is what the
    table is: ``DiagnosticType.name`` is unique, so pairs would admit
    duplicates that only fail later, at insert time.

    Returns:
        MappingProxyType:    Diagnostic name to its description. Read-only,
            so it is safe to hand the same object to every caller.
    """

    return MappingProxyType(
        {
            "num_extracted_src": ("The number of extracted stars in the image"),
            **{
                f"{param}_center": (
                    f"The smoothed source extraction {param.upper()} "
                    "parameter at the center of the image"
                )
                for param in ("s", "d", "k")
            },
            **{
                f"{param}_map_residual": (
                    f"RMS difference between source extraction "
                    f"{param.upper()} and smoothed {param.upper()} map"
                )
                for param in ("s", "d", "k")
            },
            "bg_center": (
                "The smoothed background level at the center of the image"
            ),
            "bg_map_residual": (
                "RMS difference between background and smoothed background "
                "map"
            ),
            **{
                f"{param}_center": (
                    f"The {descr} the center of the image according "
                    "to the astrometric solution"
                )
                for param, descr in (
                    ("ra", "right ascension of"),
                    ("dec", "declination of"),
                    ("z", "zenith distance of"),
                )
            },
            "diagonal_fov": (
                "The mean angular distance from the image center to its "
                "four corners on the sky, used as a scale-independent "
                "measure of the field of view"
            ),
            "pointing_offset": (
                "The angular distance between the target and the center of "
                "the image according to the astrometric solution"
            ),
            "matched_fraction": (
                "The fraction of extracted sources that were matched to "
                "the reference catalog"
            ),
            "astrom_residual": (
                "The RMS distance between matched extracted sources and "
                "their projected positions"
            ),
            "srcextract_mag_zeropt": (
                "The zeropoint of the transformation between source "
                "extraction flux and catalog magnitude (the magnitude "
                "corresponding to a flux of 1 ADU)"
            ),
            "magfit_residual": (
                "The RMS difference between best fit correction using the "
                "final master photometric reference."
            ),
            "photometry_mag_offset": (
                "The best-fit offset between the image magnitude and "
                "the reference magnitude in magnitude fit."
            ),
            "mag_fit_num_stars": (
                "The number of stars used in the last magnitude fit "
                "iteration for this image"
            ),
        }
    )


@functools.lru_cache(maxsize=1)
def standard_diagnostic_names():
    """
    Return just the names, which is all validating an expression needs.

    Returns:
        frozenset:    The seeded diagnostic names.
    """

    return frozenset(standard_diagnostic_types())


#: The one diagnostic family created at run time rather than seeded.
#: ``calibrate`` records one per configured quantile, so which exist
#: depends on how a project was configured and cannot be listed ahead of
#: time. Digits are required and the match anchored, so ``pixel_q999`` is
#: recognised while a plausible future diagnostic such as ``pixel_quality``
#: is not swallowed.
_quantile_pattern = re.compile(r"pixel_q\d+\Z")


def is_quantile_diagnostic(name):
    """
    Whether *name* is one of the run-time quantile diagnostics.

    The single definition of what a quantile diagnostic is called. Both
    the code that creates the rows and the code that validates expressions
    against them ask here, so the two cannot drift into disagreeing.

    Args:
        name(str):    The name to test.

    Returns:
        bool:    Whether ``calibrate`` would record under this name.
    """

    return _quantile_pattern.match(name) is not None


def is_diagnostic(name):
    """
    Whether *name* is a diagnostic AutoWISP can record, in any project.

    The complete vocabulary, and knowable without opening a database: a
    ``diagnostic_type`` row can only come from
    :func:`standard_diagnostic_types` at project creation or from the
    quantile branch of ``ImageProcessingManager._save_image_diagnostics``,
    which refuses every other name. So no project can hold a diagnostic
    this does not recognise, and validating an expression needs no project.

    This is *not* the question of whether anything has been recorded here,
    which is per-project and answered by counting rows.

    Args:
        name(str):    The name to test.

    Returns:
        bool:    Whether the name refers to a diagnostic.
    """

    return name in standard_diagnostic_names() or is_quantile_diagnostic(name)


#: The name ``Image.jd`` is plotted and referenced under. Not a diagnostic
#: -- it is a column of the image row rather than an ``image_diagnostics``
#: value -- but it is a variable in the same flat name space, and the only
#: one that is never NaN, since the canonical image list is defined by
#: ``jd IS NOT NULL``. Lives here rather than in ``expression_series`` so
#: that validating an expression needs nothing from the database tier.
time_quantity = "jd"


def is_known_quantity(name):
    """
    Whether *name* resolves to data an expression may read.

    The whole readable vocabulary: every diagnostic, plus the time. This is
    what tier 1 checks a referenced name against.

    Args:
        name(str):    The name to test.

    Returns:
        bool:    Whether the name refers to something readable.
    """

    return name == time_quantity or is_diagnostic(name)


#: The pseudo-quantity offered in place of the individual quantiles, which
#: expands to one plotted series per ``pixel_q*`` rather than standing for
#: values of its own. Deliberately spelled without digits, so that it does
#: not match :func:`is_quantile_diagnostic` -- it names the family, and is
#: never one of its members.
quantiles_quantity = "pixel_quantiles"


def is_reserved_name(name):
    """
    Whether something already answers to *name*, so an expression may not.

    Wider than :func:`is_known_quantity` by exactly one entry, and the
    asymmetry is the point: :data:`quantiles_quantity` cannot be *read* by
    an expression, since it stands for a family rather than for values, but
    neither may it be shadowed by one. Expressions, diagnostics and the
    family name share one flat name space -- it is what lets a selector and
    a URL treat them alike -- so a name meaning one thing to the selector
    and another inside an expression would be ambiguous in both.

    Args:
        name(str):    The proposed expression name.

    Returns:
        bool:    Whether the name is taken.
    """

    return name == quantiles_quantity or is_known_quantity(name)
