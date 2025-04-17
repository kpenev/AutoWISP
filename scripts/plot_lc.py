#!/usr/bin/env python3

"""Plot brightness measurements vs time for a star."""

from matplotlib import use as set_mpl_backend
from matplotlib import pyplot

import numpy
from configargparse import ArgumentParser, DefaultsFormatter

from autowisp.diagnostics.plot_lc import get_plot_data, calculate_combined
from autowisp import LightCurveFile, DataReductionFile


def parse_range(range_str):
    """Parse optimize option value."""

    key, range_str = range_str.split(":")
    start, stop = map(int, range_str.split(".."))
    return key, range(start, stop + 1)


def parse_command_line():
    """Return the command line arguments as attributes of an object."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["plot_lc.cfg"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )

    parser.add_argument(
        "lightcurve", help="The filenames of the lightcurve to plot."
    )
    parser.add_argument(
        "--config-file",
        "-c",
        is_config_file=True,
        help="Specify a configuration file in liu of using command line "
        "options. Any option can still be overriden on the command line.",
    )
    parser.add_argument(
        "-x",
        default="skypos.BJD - skypos.BJD.min()",
        help="The x expression to plot. By default plots time from first LC "
        "data point. For example: ``skypos.BJD %% 3.14`` will plot a phase "
        "folded lightcurve folded at a period of 3.14 days.",
    )
    parser.add_argument(
        "-y",
        default="{mode}.magfit.magnitude - nanmedian({mode}.magfit.magnitude)",
        help="The y expression to plot. By default plots the magnitude offset "
        "from the median for magnitude fitting magnitudes. Replace ``magfit``"
        "with ``epd`` or ``tfa`` to plot EPD or TFA magnitudes respectively or "
        "specify entirely different expression to plot. May include ``{mode}`` "
        "substitution which will be replaced by the photometry mode (e.g. "
        "``shapefit`` or ``apphot``) per ``--photometry-modes`` argument.",
    )
    parser.add_argument(
        "--bin",
        default=None,
        nargs=2,
        help="Binning to apply. Should be an expression of lightcurve "
        "quantities followed by a binning function. Some examples include:\n"
        "\t* ``fitsheader.rawfname`` ``nanmedian`` to plot the color "
        "median of the measurements in different channels of an each image.\n"
        "\t* ``(skypos.BJD * 24 * 12).astype(int)`` ``nanmedian`` to bin in 5 "
        "min increments. By default no binning is applied.",
    )
    parser.add_argument(
        "--photometry-modes",
        nargs="+",
        default=["apphot"],
        help="The photometry modes to search for best lightcurve.",
    )
    parser.add_argument(
        "--optimize",
        nargs="+",
        default={"aperture_index": range(46)},
        type=parse_range,
        help="Lightcurve substitutions to optimize to find the smallest value "
        "of the expression specified in ``--minimize-expression``. The format "
        "is ``substitution_key:<min>..<max>``. If multiple keys are given all "
        "possible combinations are tested.",
    )
    parser.add_argument(
        "--minimize-expression",
        default=(
            "nanmedian(abs({mode}.magfit.magnitude - "
            "nanmedian({mode}.magfit.magnitude)))"
        ),
        help="The expression to minimize by iterating over all possibilities "
        "specified by ``--optimize``.",
    )
    parser.add_argument(
        "--fmt",
        default="ok",
        help="The format to use for plotting the lightcurve points. See "
        "matplotlib for description.",
    )
    parser.add_argument(
        '--x-range',
        default=None,
        nargs=2,
        type=float,
        help='If specified, the plot x range is set to the given lower and '
        'upper boundaries.'
    )
    parser.add_argument(
        '--y-range',
        default=None,
        nargs=2,
        type=float,
        help='If specified, the plot y range is set to the given lower and '
        'upper boundaries.'
    )

    parser.add_argument(
        "--plot-fname",
        "--plot",
        "-o",
        help="The filename under which to save the plot. By default the plot "
        "is display in an interactive window.",
    )
    return parser.parse_args()


def main(config):
    """Actually create the plot."""

    if config.plot_fname is None:
        set_mpl_backend("TkAgg")
    elif config.plot_fname.lower().endswith("pdf"):
        set_mpl_backend("PDF")
    else:
        set_mpl_backend("Agg")
    expressions = {var: getattr(config, var) for var in ["x", "y"]}
    if config.bin is not None:
        expressions["bin"] = config.bin[0]

    data_by_sphotref, _ = get_plot_data(
        config.lightcurve,
        expressions=expressions,
        configuration={
            "lc_substitutions": {},
            "selection": None,
            "find_best": dict(config.optimize),
            "minimize": config.minimize_expression,
            "photometry_modes": config.photometry_modes,
        },
    )
    nrows = 2 if len(data_by_sphotref) <= 4 else 3
    if config.bin is None:
        for subfig_id, (sphotref_fname, single_data) in enumerate(
            data_by_sphotref.items()
        ):
            print(f"Single data: {single_data!r}")
            pyplot.subplot(nrows, nrows, subfig_id + 1)
            pyplot.plot(single_data["x"], single_data["y"], config.fmt)
            # pyplot.plot(single_data["x"], single_data["best_model"], "-k")
            with DataReductionFile(sphotref_fname, "r") as dr_file:
                pyplot.title(dr_file.get_frame_header()["CLRCHNL"])

            if config.x_range is not None:
                pyplot.xlim(*config.x_range)
            if config.y_range is not None:
                pyplot.ylim(*config.y_range)

    else:
        data_combined = calculate_combined(
            data_by_sphotref, "bin", getattr(numpy, config.bin[1])
        )

        pyplot.plot(data_combined["x"], data_combined["y"], config.fmt)

        if config.x_range is not None:
            pyplot.xlim(*config.x_range)
        if config.y_range is not None:
            pyplot.ylim(*config.y_range)

    if config.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(config.plot_fname)


if __name__ == "__main__":
    main(parse_command_line())
