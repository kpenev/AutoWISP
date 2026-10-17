#!/usr/bin/env python3

"""Plot a star's lightcurves, from one or more projects, on one figure.

Each lightcurve is given as ``<project home>:<star>``. The project's database
maps the lightcurve's quantities to its datasets, by the layout version the
file records, and its ``lc-fname`` locates the star's file, so each project
is opened, read-only, while its lightcurve is read.

Points are drawn in series (``--series``), by default one per lightcurve,
and optionally binned within each series (``--bin``). The aperture of each
lightcurve, per photometric reference, is the one minimizing
``--minimize``: by default the scatter of the magnitudes, or with a model
(``--model``), their deviation from it.

Run from a checkout: ``python scripts/plot_lc.py <project home>:<star> ...``.
"""

from argparse import ArgumentTypeError
from copy import deepcopy
import logging
from os import path
import sys

import numpy
import pandas
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
from matplotlib import pyplot

from autowisp.catalog import resolve_gaia_id
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.interface import set_project_home
from autowisp.diagnostics.get_from_lc import (
    DEFAULT_MODELS,
    get_plot_data,
    parse_range,
)

_logger = logging.getLogger(__name__)


def parse_lightcurve(lightcurve):
    """
    Parse a lightcurve argument: ``<project home>:<star>``.

    Split at the last colon, so that a Windows home keeps its drive's.

    Returns:
        (str, str):    The project home and the star.
    """

    project_home, separator, star = lightcurve.rpartition(":")
    if not (separator and project_home and star):
        raise ArgumentTypeError(
            f"{lightcurve!r} is not of the form <project home>:<star>."
        )
    return project_home, star


def parse_model_param(param):
    """Parse a value of ``--model-param``: ``<name>=<expression>``."""

    name, separator, expression = param.partition("=")
    if not (separator and name.strip() and expression.strip()):
        raise ArgumentTypeError(
            f"{param!r} is not of the form <name>=<expression>."
        )
    return name.strip(), expression.strip()


def get_bin_expression(bin_spec):
    """
    Return the expression whose equal values form one bin.

    Args:
        bin_spec(str):    ``colors`` for the channels of one raw image; a
            duration ``astropy.units.Quantity`` reads, e.g. ``5min``, for
            bins of that width in time; anything else is the expression.

    Returns:
        str:    The expression.
    """

    if bin_spec == "colors":
        return "fitsheader.rawfname"
    try:
        width = units.Quantity(bin_spec)
    except (TypeError, ValueError):
        return bin_spec
    if width.unit.physical_type != "time":
        return bin_spec
    return f"(skypos.BJD / {float(width.to_value(units.day))!r}).astype(int)"


def get_model(config):
    """
    Return the model to fit, as :func:`get_plot_data` takes it, or None.

    The type's default parameters, as lightcurve expressions, overridden by
    those given with ``--model-param``. The model is of the plotted
    quantity, shifted to its median.
    """

    if config.model is None:
        return None
    kwargs = {
        name: value if isinstance(value, str) else repr(value)
        for name, value in DEFAULT_MODELS[config.model].items()
    }
    kwargs.update(config.model_param)
    return {
        "type": config.model,
        "kwargs": kwargs,
        "quantity": "y",
        "shift_to": True,
    }


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "lightcurves",
        nargs="+",
        type=parse_lightcurve,
        metavar="<project home>:<star>",
        help="The lightcurves to plot: each a project's home and a star, "
        "which may be given as ``GDR3 <id>``, ``TIC <id>``, ``TOI-<number>`` "
        "or a name the NASA Exoplanet Archive knows. Every point carries the "
        "lightcurve it came from as ``lightcurve``, the argument as given, "
        "which any expression may use.",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        is_config_file=True,
        help="A configuration file giving any of the options. Options given "
        "on the command line override it.",
    )
    parser.add_argument(
        "-x",
        default="skypos.BJD",
        help="The x expression to plot. For example: ``skypos.BJD %% 3.14`` "
        "plots the lightcurve folded at a period of 3.14 days.",
    )
    parser.add_argument(
        "-y",
        default="{mode}.magfit.magnitude - nanmedian({mode}.magfit.magnitude)",
        help="The y expression to plot. By default the magnitude fitting "
        "magnitudes' offset from their median. Replace ``magfit`` with "
        "``epd`` or ``tfa`` for EPD or TFA magnitudes. ``{mode}`` is replaced "
        "by the photometry mode (e.g. ``apphot``) per ``--photometry-modes``.",
    )
    parser.add_argument(
        "--series",
        default="lightcurve",
        help="The expression whose equal values form one series, drawn in "
        "one colour with one legend entry. By default each lightcurve is a "
        "series of its own; a constant (``\"'all'\"``) merges them, and "
        "``fitsheader.cfg.clrchnl`` tells channels apart.",
    )
    parser.add_argument(
        "--bin",
        nargs="+",
        default=None,
        metavar=("SPEC", "AGGREGATE"),
        help="Bin the points of each series, plotting one point per bin. SPEC "
        "is ``colors`` (the channels of one raw image), a duration such as "
        "``5min`` or ``0.5h`` (bins of that width in time), or an expression "
        "whose equal values form a bin. AGGREGATE is the numpy function "
        "combining a bin's points (default ``nanmedian``). Bins never span "
        "series.",
    )
    parser.add_argument(
        "--selection",
        default=None,
        help="An expression selecting the points to use; all by default.",
    )
    parser.add_argument(
        "--photometry-modes",
        nargs="+",
        default=["apphot"],
        help="The photometry modes to search for the best lightcurve.",
    )
    parser.add_argument(
        "--magfit-iteration",
        type=int,
        default=-1,
        help="The magnitude fitting iteration to plot; negative counts from "
        "the last.",
    )
    parser.add_argument(
        "--find-best",
        nargs="+",
        type=parse_range,
        default=None,
        help="Lightcurve substitutions to choose by minimizing "
        "``--minimize``, each as ``<substitution>:<min>..<max>``; every "
        "combination is tried. By default every aperture of each "
        "lightcurve's project.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(DEFAULT_MODELS),
        default=None,
        help="A model of the plotted quantity, drawn over each series. The "
        "best aperture is then the one deviating least from it.",
    )
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_model_param,
        default=[],
        help="A parameter of ``--model``, as ``<name>=<expression>``: a "
        "number, a list, or a lightcurve expression such as ``skypos.BJD``. "
        "Can be repeated. Parameters not given keep their defaults: "
        + "; ".join(
            f"{model}: "
            + ", ".join(f"{name}={value!r}" for name, value in params.items())
            for model, params in DEFAULT_MODELS.items()
        ).replace("%", "%%"),
    )
    parser.add_argument(
        "--minimize",
        default=None,
        help="The expression the best substitutions minimize. By default the "
        "median absolute deviation of the magnitudes from their median, or "
        "with ``--model`` that of their deviation from the model "
        "(``model_diff``).",
    )
    parser.add_argument(
        "--fmt",
        default="o",
        help="The matplotlib format of the points. A colour given here "
        "colours every series alike.",
    )
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        default=None,
        help="The lower and upper bounds of the x axis.",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        default=None,
        help="The lower and upper bounds of the y axis.",
    )
    parser.add_argument(
        "--plot-fname",
        "--plot",
        "-o",
        default=None,
        help="The file to save the plot to. By default it is shown in an "
        "interactive window.",
    )
    config = parser.parse_args()
    if config.bin is not None and len(config.bin) > 2:
        parser.error("--bin takes a spec and, optionally, an aggregate.")
    config.model_param = dict(config.model_param)
    if config.model_param and config.model is None:
        parser.error("--model-param needs --model.")
    if config.minimize is None:
        config.minimize = (
            "nanmedian(abs({mode}.magfit.magnitude - "
            "nanmedian({mode}.magfit.magnitude)))"
            if config.model is None
            else "nanmedian(abs(model_diff - nanmedian(model_diff)))"
        )
    return config


def read_lightcurve(project_home, star, config):
    """
    Read the points to plot from one project's lightcurve of a star.

    Args:
        project_home(str):    The project's home.

        star(str):    The star, as :func:`resolve_gaia_id` takes it.

        config:    The command line configuration.

    Returns:
        pandas.DataFrame or None:    One row per point, with columns ``x``,
        ``y``, ``series``, ``bin`` if binning and ``best_model`` with a
        model; None if the project has no lightcurve of the star.
    """

    set_project_home(project_home)
    param_values = ImageProcessingManager(
        pipeline_run_id=None
    ).get_param_values({1}, ["lc-fname", "apertures"])
    lc_fname = param_values["lc-fname"].format(
        resolve_gaia_id(star), PROJHOME=project_home
    )
    if not path.exists(lc_fname):
        _logger.warning(
            "%s has no lightcurve of %s: skipped.", project_home, star
        )
        return None

    expressions = {"x": config.x, "y": config.y, "series": config.series}
    if config.bin is not None:
        expressions["bin"] = get_bin_expression(config.bin[0])
    data_by_sphotref = get_plot_data(
        lc_fname,
        expressions,
        {
            "lc_substitutions": {"magfit_iteration": config.magfit_iteration},
            "selection": config.selection,
            "find_best": config.find_best
            or [
                (
                    "aperture_index",
                    range(len(param_values["apertures"].split(","))),
                )
            ],
            "minimize": config.minimize,
            "photometry_modes": config.photometry_modes,
        },
        deepcopy(get_model(config)),
        {"lightcurve": f"{project_home}:{star}"},
    )
    frames = []
    for sphotref_data in data_by_sphotref.values():
        num_points = len(sphotref_data["x"])
        frames.append(
            pandas.DataFrame(
                {
                    # A constant expression, e.g. the lightcurve, is one value
                    # for all the points.
                    column: numpy.broadcast_to(
                        sphotref_data[column], (num_points,)
                    )
                    for column in list(expressions) + ["best_model"]
                    if sphotref_data.get(column) is not None
                }
            )
        )
    return pandas.concat(frames, ignore_index=True)


def plot_series(points, config):
    """Draw each series of the points, binned if asked, with its model."""

    if config.bin is not None:
        aggregate = getattr(
            numpy, config.bin[1] if len(config.bin) > 1 else "nanmedian"
        )
        points = (
            points.groupby(["series", "bin"], sort=False)
            .agg(aggregate)
            .reset_index()
        )
    for series, series_points in points.groupby("series", sort=False):
        (line,) = pyplot.plot(
            series_points["x"],
            series_points["y"],
            config.fmt,
            label=series.decode() if isinstance(series, bytes) else series,
        )
        if "best_model" in series_points:
            ordered = series_points.sort_values("x")
            model = ordered["best_model"].to_numpy()
            pyplot.plot(
                ordered["x"],
                model + numpy.nanmedian(ordered["y"].to_numpy() - model),
                "-",
                color=line.get_color(),
            )
    if points["series"].nunique() > 1:
        pyplot.legend()


def main():
    """Plot the lightcurves per the command line."""

    config = parse_command_line()
    lightcurves = [
        read_lightcurve(project_home, star, config)
        for project_home, star in config.lightcurves
    ]
    lightcurves = [points for points in lightcurves if points is not None]
    if not lightcurves:
        sys.exit("None of the lightcurves were found.")
    plot_series(pandas.concat(lightcurves, ignore_index=True), config)
    if config.x_range is not None:
        pyplot.xlim(*config.x_range)
    if config.y_range is not None:
        pyplot.ylim(*config.y_range)
    if config.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(config.plot_fname)


if __name__ == "__main__":
    main()
