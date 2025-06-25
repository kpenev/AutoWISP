"""Views for displaying the lightcurve of a star."""

from itertools import product
from copy import deepcopy
from io import StringIO, BytesIO
import json

import matplotlib
from matplotlib import pyplot, gridspec, rcParams
import numpy
from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from autowisp.bui_util import hex_color
from autowisp.evaluator import Evaluator
from autowisp.diagnostics.plot_lc import get_plot_data, calculate_combined
from autowisp.database.image_processing import ImageProcessingManager

_custom_aggregators = {"len": len}

_default_models = {
    "transit": {
        "times": "skypos.BJD",
        "k": 0.1,  # the planet-star radius ratio
        "ldc": [0.8, 0.7],  # limb darkening coeff
        "t0": 2455787.553228,  # the zero epoch,
        "p": 3.0,  # the orbital period,
        "a": 10.0,  # the orbital semi-major divided by R*,
        "i": numpy.pi / 2,  # the orbital inclination in rad,
        "e": 0.0,  # the orbital eccentricity (optional, can be left out if
        # assuming circular a orbit), and
        "w": 0.0,  # the argument of periastron in radians (also optional,
        # can be left out if assuming circular a orbit).
    }
}


def _init_session(request):
    """Initialize the session for displaying lightcurve with defaults."""

    color_map = matplotlib.colormaps.get_cmap("tab10")
    request.session["lc_plotting"] = {
        "lc_fname_template": ImageProcessingManager(
            dummy=True
        ).get_param_values({1}, ["lc-fname"])["lc-fname"],
        "target_fname": "",
        "color_map": [hex_color(color_map(i)) for i in range(10)],
        "data_select": [
            {
                "lc_substitutions": {"magfit_iteration": -1},
                "find_best": {"aperture_index": "0..40"},
                "minimize": (
                    "nanmedian(abs({mode}.tfa.magnitude - "
                    "nanmedian({mode}.tfa.magnitude)))"
                ),
                "photometry_modes": ["apphot"],
                "selection": None,
                "model": None,
                "expressions": {
                    "magnitude": (
                        "{mode}.tfa.magnitude - "
                        "nanmedian({mode}.tfa.magnitude)"
                    ),
                    "bjd": "skypos.BJD - skypos.BJD.min()",
                    "rawfname": "fitsheader.rawfname",
                },
                "plot_config": [
                    [
                        {
                            "sphotref_selector": "*",
                            "x_aggregate": "nanmedian",
                            "y_aggregate": "nanmedian",
                            "x": "bjd",
                            "y": "magnitude",
                            "match_by": "rawfname",
                            "curve_label": "tfa",
                            "plot_kwargs": {
                                "marker": "o",
                                "markersize": 3,
                                "markeredgecolor": "none",
                                "markerfacecolor": hex_color(color_map(0)),
                                "linestyle": "none",
                                "linewidth": 0,
                                "color": "#ffffff",
                            },
                        }
                    ]
                ],
            }
        ],
        "plot_layout": [
            {
                "width_ratios": [1.0],
                "height_ratios": [1.0],
                "wspace": rcParams["figure.subplot.wspace"],
                "hspace": rcParams["figure.subplot.hspace"],
            },
            [0],
        ],
        "plot_decorations": [
            {
                "x_label": "BJD",
                "y_label": "magnitude",
                "title": "GDR3 {GaiaID}",
                "xmin": None,
                "xmax": None,
                "ymin": None,
                "ymax": None,
            }
        ],
        "figure_config": {},
    }
    print(f'Color map: {request.session["lc_plotting"]["color_map"]}')


def _jsonify_plot_data(plot_data):
    """Re-format plot data for single sphotref for storing in JSON format."""

    result = {}
    for key, value in plot_data.items():
        if value is None:
            result[key] = value
        else:
            if isinstance(value[0], bytes):
                result[key] = [entry.decode() for entry in value]
            else:
                result[key] = value.tolist()
    return result


def _unjsonify_plot_data(json_data):
    """Undo `_jsonify_plot_data()`."""

    result = {}
    for key, value in json_data.items():
        if value is None:
            result[key] = value
        else:
            if isinstance(value[0], str):
                result[key] = numpy.array(
                    [entry.encode("ascii") for entry in value]
                )
            else:
                result[key] = numpy.array(value)
    return result


def _convert_plot_data_json(plot_data, reverse):
    """Re-format plot data for storing in JSON format or reverse conversion."""

    transform = _unjsonify_plot_data if reverse else _jsonify_plot_data
    return {fname: transform(data) for fname, data in plot_data.items()}


def _parse_optimizables(optimizables):
    """Convert user shorthands to lists of allowed values for ``find_best``."""

    result = {}
    for substitution, user_value in optimizables.items():
        if ".." in user_value:
            result[substitution] = list(
                range(*map(int, user_value.split("..")))
            )
        else:
            result[substitution] = [
                int(s)
                for s in user_value.split("," if "," in user_value else None)
            ]
    return result


def _add_lightcurve_to_session(plotting_info, lightcurve_fname, select=True):
    """Add to the browser session a new entry for the given lightcurve."""

    plotting_info[lightcurve_fname] = []

    for data_select in plotting_info["data_select"]:
        print("Data select pre getting data: " + repr(data_select))
        plot_data, best_substitutions = get_plot_data(
            lightcurve_fname,
            data_select["expressions"],
            {
                k: (
                    _parse_optimizables(data_select[k])
                    if k == "find_best"
                    else data_select[k]
                )
                for k in [
                    "lc_substitutions",
                    "find_best",
                    "minimize",
                    "photometry_modes",
                    "selection",
                ]
            },
            data_select.get("model"),
        )
        print("Data select post getting data: " + repr(data_select))

        plotting_info[lightcurve_fname].append(
            {
                "plot_data": _convert_plot_data_json(plot_data, False),
                "best_substitutions": best_substitutions,
            }
        )
    if select:
        plotting_info["target_fname"] = lightcurve_fname


def plot(target_info, plot_config, plot_decorations):
    """Make a single plot of the spceified lighturve."""

    plot_data = {}
    print(f"Plot config (len:{len(plot_config)}): {plot_config!r}")
    print(f"Plot decorations : {plot_decorations!r}")
    print(f"len(target_info): {len(target_info)}")
    assert len(plot_config) == len(target_info)
    for dataset, dataset_plot_configs in zip(target_info, plot_config):
        plot_data = _convert_plot_data_json(dataset["plot_data"], True)
        print(f"Plot configs: {dataset_plot_configs!r}")
        for curve_config in dataset_plot_configs:
            if curve_config["sphotref_selector"] == "*":
                curve_data = plot_data
            else:
                if isinstance(curve_config["sphotref_selector"], list):
                    curve_data = {
                        sphotref_fname: plot_data[sphotref_fname]
                        for sphotref_fname in curve_config["sphotref_selector"]
                    }
                else:
                    curve_data = {
                        sphotref_fname: plot_data[sphotref_fname]
                        for sphotref_fname in plot_data.keys()
                        if Evaluator(sphotref_fname)(
                            curve_config["sphotref_selector"]
                        )
                    }
            # TODO: optimize to combine only coord and match_by
            curve_data = {
                coord: calculate_combined(
                    curve_data,
                    curve_config["match_by"],
                    (
                        _custom_aggregators.get(
                            curve_config[coord + "_aggregate"]
                        )
                        or getattr(numpy, curve_config[coord + "_aggregate"])
                    ),
                )
                for coord in "xy"
            }
            pyplot.plot(
                curve_data["x"][curve_config["x"]],
                curve_data["y"][curve_config["y"]],
                *curve_config.get("plot_args", []),
                label=curve_config["curve_label"],
                **curve_config.get("plot_kwargs", {}),
            )
    pyplot.legend()
    pyplot.xlim(plot_decorations["xmin"], plot_decorations["xmax"])
    pyplot.ylim(plot_decorations["ymin"], plot_decorations["ymax"])

    pyplot.xlabel(plot_decorations["x_label"])
    pyplot.ylabel(plot_decorations["y_label"])
    pyplot.title(plot_decorations["title"])


def _create_subplots(plotting_info, splits, children, parent, figure):
    """Recursively walks the plot layout tree creating subplots as needed."""

    args = (len(splits["height_ratios"]), len(splits["width_ratios"]))
    kwargs = {
        k: splits[k] for k in splits if k not in ["x_label", "y_label", "title"]
    }

    if parent is None:
        grid = gridspec.GridSpec(*args, figure=figure, **kwargs)
    else:
        grid = gridspec.GridSpecFromSubplotSpec(
            *args, subplot_spec=parent, **kwargs
        )

    assert len(children) == args[0] * args[1]
    for child, subplot in zip(children, grid):
        if isinstance(child, int):
            pyplot.sca(figure.add_subplot(subplot))
            plot(
                plotting_info["target_info"],
                [
                    plot_config[child]
                    for plot_config in plotting_info["plot_config"]
                ],
                plotting_info["plot_decorations"][child],
            )
        else:
            _create_subplots(plotting_info, *child, subplot, figure)


def _get_subplot_boundaries(splits, children, x_offset, y_offset, result):
    """Return coords of horizontal and vertical boundaries between plots."""

    x_bounds = numpy.cumsum([x_offset] + splits["width_ratios"])
    y_bounds = numpy.cumsum([y_offset] + splits["height_ratios"])
    cell_indices = product(
        range(len(splits["height_ratios"])), range(len(splits["width_ratios"]))
    )
    for child, (y_ind, x_ind) in zip(children, cell_indices):
        if isinstance(child, int):
            result[child] = {
                "left": x_bounds[x_ind],
                "right": x_bounds[x_ind + 1],
                "top": y_bounds[y_ind],
                "bottom": y_bounds[y_ind + 1],
            }
        else:
            _get_subplot_boundaries(
                *child, x_bounds[x_ind], y_bounds[y_ind], result
            )


def _subdivide_figure(plot_config, new_splits, current_splits, children):
    """Sub-divide all plots with entries in new_splits accordingly."""

    for child_ind, child in enumerate(children):
        if isinstance(child, int):
            child_splits = new_splits.get(str(child))
            orig_width = current_splits["width_ratios"][
                child_ind % len(current_splits["width_ratios"])
            ]
            orig_height = current_splits["height_ratios"][
                child_ind // len(current_splits["width_ratios"])
            ]
            if child_splits is not None:
                child_splits = {
                    side: child_splits.get(side, [1.0])
                    for side in ["top", "left"]
                }
                num_subplots = len(child_splits["top"]) * len(
                    child_splits["left"]
                )
                children[child_ind] = [
                    {
                        "height_ratios": [
                            s * orig_height for s in child_splits["left"]
                        ],
                        "width_ratios": [
                            s * orig_width for s in child_splits["top"]
                        ],
                        "wspace": (
                            current_splits["wspace"] * len(child_splits["top"])
                        ),
                        "hspace": (
                            current_splits["hspace"] * len(child_splits["left"])
                        ),
                    },
                    [child]
                    + list(
                        range(
                            len(plot_config[0]),
                            len(plot_config[0]) + num_subplots - 1,
                        )
                    ),
                ]
                for config in plot_config:
                    config.extend(
                        (
                            deepcopy(config[child])
                            for _ in range(num_subplots - 1)
                        )
                    )
                print(f"After subdividing, children: {children!r}.")
        else:
            _subdivide_figure(plot_config, new_splits, *child)


def update_subplot(plotting_session, updates):
    """Change a given sub-plot (and/or add plot quantities)."""

    print(f'Updating plot {updates["plot_id"]} with: {updates!r}')
    plot_id = int(updates.pop("plot_id"))
    if len(plotting_session["data_select"]) < len(updates["data_select"]):
        plotting_session["data_select"].extend(
            {
                "plot_config": (
                    [None]
                    * len(plotting_session["data_select"][0]["plot_config"])
                )
            }
            for _ in range(
                len(updates["data_select"])
                - len(plotting_session["data_select"])
            )
        )
    elif len(plotting_session["data_select"]) > len(updates["data_select"]):
        plotting_session["data_select"] = plotting_session["data_select"][
            : len(updates["data_select"])
        ]
    for original, updated in zip(
        plotting_session["data_select"], updates["data_select"]
    ):
        for k in updated:
            if k == "plot_config":
                original[k][plot_id] = deepcopy(updated[k])
                print(f"Updated plotting info: {original[k]}")
            else:
                original[k] = deepcopy(updated[k])
    for decoration in ["x_label", "y_label", "title"]:
        plotting_session["plot_decorations"][plot_id][decoration] = updates[
            decoration
        ]
    for decoration in [
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    ]:
        plotting_session["plot_decorations"][plot_id][decoration] = (
            float(updates[decoration])
            if updates[decoration] and updates[decoration] != "None"
            else None
        )
    if updates["star_id_type"] == "GDR3":
        gaia_id = updates["star_id"]
    elif updates["star_id_type"] == "TIC":
        gaia_id = Catalogs.query_criteria(
            catalog="Tic", ID=int(updates["star_id"])
        )["GAIA"]
    else:
        gaia_id = NasaExoplanetArchive.query_object(updates["star_id"])[
            "gaia_id"
        ]
        gaia_id = numpy.unique(gaia_id)
        assert len(gaia_id) == 1
        gaia_id = gaia_id[0].split()[-1]

    plotting_session["target_fname"] = plotting_session[
        "lc_fname_template"
    ].format(int(gaia_id))
    _add_lightcurve_to_session(
        plotting_session, plotting_session["target_fname"]
    )


def _update_plotting_info(plotting_session, updates):
    """Modify the currently set-up figure per user input from BUI."""

    print(f"Updates to apply: {updates!r}")
    modified_session = False
    if "applySplits" in updates:
        _subdivide_figure(
            [
                data_select["plot_config"]
                for data_select in plotting_session["data_select"]
            ]
            + [plotting_session["plot_decorations"]],
            updates["applySplits"],
            *plotting_session["plot_layout"],
        )
        modified_session = True
    if "rcParams" in updates:
        for param, value in updates["rcParams"].items():
            rcParams[param] = value.strip("[]")
    if "subplot" in updates:
        update_subplot(plotting_session, updates["subplot"])
        modified_session = True
    return modified_session


def update_lightcurve_figure(request):
    """Generate and return a new figure for the current lightcurve."""

    request.session.modified = _update_plotting_info(
        request.session["lc_plotting"], json.loads(request.body.decode())
    )

    matplotlib.use("svg")
    pyplot.style.use("dark_background")

    figure = pyplot.figure(**request.session["lc_plotting"]["figure_config"])
    plotting_info = request.session["lc_plotting"]
    if plotting_info["target_fname"]:
        _create_subplots(
            {
                "target_info": plotting_info[plotting_info["target_fname"]],
                "plot_layout": plotting_info["plot_layout"],
                "plot_config": [
                    data_select["plot_config"]
                    for data_select in plotting_info["data_select"]
                ],
                "plot_decorations": plotting_info["plot_decorations"],
            },
            *request.session["lc_plotting"]["plot_layout"],
            None,
            figure,
        )

    with StringIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches="tight", format="svg")
        subplot_boundaries = {}
        _get_subplot_boundaries(
            *request.session["lc_plotting"]["plot_layout"],
            0,
            0,
            subplot_boundaries,
        )
        return JsonResponse(
            {
                "plot_data": image_stream.getvalue(),
                "boundaries": subplot_boundaries,
            }
        )


def edit_subplot(request, plot_id):
    """Set the view to allow editing the selected plot."""

    plotting_info = request.session["lc_plotting"]
    data_select = [
        {
            param: value[plot_id] if param == "plot_config" else value
            for param, value in data_select_entry.items()
        }
        for data_select_entry in plotting_info["data_select"]
    ]

    print("Sub-plot data_select: " + repr(data_select))
    return render(
        request,
        "results/subplot_config.html",
        {
            "data_select": data_select,
            "plot_decorations": plotting_info["plot_decorations"][plot_id],
            "figure_config": plotting_info["figure_config"],
        },
    )


def _sanitize_rcparams():
    """Return list of matplotlib rcParams that can be set through BUI."""

    result = []
    for param, value in rcParams.items():
        if param == "lines.dash_capstyle":
            value = "butt"
        elif param == "lines.solid_capstyle":
            value = "projecting"
        elif param.endswith("_joinstyle"):
            value = "round"
        if (
            # Cannot be reduced in a readable way
            # pylint: disable=too-many-boolean-expressions
            not param.endswith("prop_cycle")
            and param not in ["savefig.bbox", "backend"]
            and not param.startswith("animation")
            and not param.startswith("keymap")
            and not param.startswith("figure.subplot")
            and param[0] != "_"
            # pylint: enable=too-many-boolean-expressions
        ):
            result.append((param, value))
    return result


def edit_rcparams(request):
    """Set the view to allow editing rcParams."""

    return render(
        request,
        "results/rcParams_config.html",
        {"config": _sanitize_rcparams()},
    )


def display_lightcurve(request):
    """Display plots of a single lightcurve to the user."""

    if "lc_plotting" not in request.session:
        rcParams["figure.subplot.bottom"] = 0.0
        rcParams["figure.subplot.top"] = 1.0
        rcParams["figure.subplot.left"] = 0.0
        rcParams["figure.subplot.right"] = 1.0
        _init_session(request)
        if request.session["lc_plotting"]["target_fname"]:
            _add_lightcurve_to_session(
                request.session["lc_plotting"],
                request.session["lc_plotting"]["target_fname"],
            )

    return render(request, "results/display_lightcurves.html", {"config": None})


def edit_model(request, model_type, data_select_index):
    """Add controls to edit the model parameters."""

    return render(
        request,
        "results/edit_model.html",
        {
            "expressions": request.session["lc_plotting"]["data_select"][
                data_select_index
            ]["expressions"].keys(),
            "model": _default_models[model_type],
        },
    )


def clear_lightcurve_buffer(request):
    """Remove buffered lightcurve data from the session."""

    if "lc_plotting" in request.session:
        del request.session["lc_plotting"]
    return redirect("/results")


def download_lightcurve_figure(request):
    matplotlib.use("pdf")
    pyplot.style.use("default")
    plotting_info = request.session["lc_plotting"]
    figure_config = dict(plotting_info["figure_config"])
    figure_config["facecolor"] = "white"
    figure = pyplot.figure(**figure_config)
    if plotting_info["target_fname"]:
        _create_subplots(
            {
                "target_info": plotting_info[plotting_info["target_fname"]],
                "plot_layout": plotting_info["plot_layout"],
                "plot_config": [
                    data_select["plot_config"]
                    for data_select in plotting_info["data_select"]
                ],
                "plot_decorations": plotting_info["plot_decorations"],
            },
            *plotting_info["plot_layout"],
            None,
            figure,
        )
    for ax in figure.get_axes():
        ax.set_facecolor("white")
    with BytesIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches="tight", format="pdf", facecolor="white", edgecolor="white")
        image_stream.seek(0)
        return HttpResponse(
            image_stream.read(),
            content_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="lightcurve.pdf"'
            }
        )