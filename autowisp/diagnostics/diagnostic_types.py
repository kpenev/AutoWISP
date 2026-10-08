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


@functools.lru_cache(maxsize=1)
def magfit_diagnostic_names():
    """
    Return the diagnostics ``fit_magnitudes`` produces.

    Their values depend on the photometric reference each image was fit
    against, not only on the image, so a population they are read over has
    to be split by reference as well. Everything deciding that asks here.

    Returns:
        frozenset:    The names, a subset of
            :func:`standard_diagnostic_names`.
    """

    return frozenset(
        ("photometry_mag_offset", "magfit_residual", "mag_fit_num_stars")
    )


@functools.lru_cache(maxsize=1)
def photometry_diagnostic_names():
    """
    Return the diagnostics recorded per photometry.

    Stored in ``photometry_diagnostics``, one value per shape fit and per
    aperture, rather than in ``image_diagnostics``, so an expression reads
    one in a photometry as well as a channel. Not the same question as
    :func:`magfit_diagnostic_names`, although today the same names answer
    both: a diagnostic ``fit_magnitudes`` recorded once per image would
    depend on the reference and still take no photometry.

    Returns:
        frozenset:    The names, a subset of
            :func:`magfit_diagnostic_names`.
    """

    return frozenset(
        ("photometry_mag_offset", "magfit_residual", "mag_fit_num_stars")
    )


#: The photometry id of the shape fit. Apertures are numbered by their index,
#: from 0, so the shape fit takes a value no aperture can have.
shapefit_photometry = -1


def get_photometry_id(position, has_shape_fit):
    """
    Return the id recorded for the photometry at *position* in magfit's arrays.

    ``fit_magnitudes`` holds an image's photometries in one array: the shape
    fit first, where the image has a usable one, and then every aperture.
    So a position is the shape fit on one image and aperture 0 on the next,
    and recording it would let a series pinned to one id mix photometries
    without saying so. The id is the same on every image instead: the
    aperture index, which is what the DR files number apertures by, or
    :data:`shapefit_photometry`.

    Args:
        position(int):    The index into magfit's photometry arrays.

        has_shape_fit(bool):    Whether those arrays start with a shape fit,
            as ``get_magfit_sources`` decided when building them.

    Returns:
        int:    The photometry id.
    """

    if not has_shape_fit:
        return position
    return shapefit_photometry if position == 0 else position - 1


def parse_photometry_literal(literal):
    """
    Return the photometry id a quoted photometry names, or ``None``.

    An expression quotes a photometry as ``'shapefit'`` or as ``'ap'``
    followed by the aperture index, ``'ap4'``, just as it quotes a channel
    by name. :func:`photometry_literal` spells an id the same way.

    Args:
        literal(str):    The quoted text.

    Returns:
        int or None:    The id, or ``None`` if *literal* names no
            photometry.
    """

    if literal == "shapefit":
        return shapefit_photometry
    aperture = re.fullmatch(r"ap([0-9]+)", literal)
    return int(aperture.group(1)) if aperture else None


def photometry_literal(photometry_id):
    """Return how an expression quotes the photometry *photometry_id*."""

    if photometry_id == shapefit_photometry:
        return "shapefit"
    return f"ap{photometry_id}"


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
