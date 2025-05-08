#!/usr/bin/env python3
"""Utilities for plotting individual lightcurves."""

from itertools import product
from functools import partial

from matplotlib import pyplot
import numpy
import pandas
from pytransit import RoadRunnerModel

from autowisp import LightCurveFile, DataReductionFile
from autowisp.evaluator import LightCurveEvaluator

# TODO:Document all expected entries in configuration for `get_plot_data()`


def evaluate_model(model, lc_eval, expression_params, shift_to=None):
    """Return the model evaluated for the given lightcurve."""

    args = [
        lc_eval(expression.format_map(expression_params))
        for expression in model.get("args", [])
    ]
    kwargs = {
        arg_name: lc_eval(expression.format_map(expression_params))
        for arg_name, expression in model.get("kwargs", {}).items()
    }
    model_values = globals()[model["type"] + "_model"](*args, **kwargs)
    if shift_to is not None:
        shift_to = lc_eval(shift_to.format_map(expression_params))
        assert len(shift_to) == len(times)
        model_values += numpy.nanmedian(shift_to - model_values)
    return model_values


def optimize_substitutions(
    lc_eval, *, find_best, minimize, y_expression, model, expression_params
):
    """
    Find the values of LC substitution params that minimize an expression.

    Updates ``lc_evals.lc_substitutions`` with the best values found.

    Args:
        lc_eval(LightCurveEvaluator):    Allows evaluating the expression to
            minimize.

        find_best(iterable):    Iterable of 2-tuples with the first entry
            in each tuple being a substitution parameters that need to be
            optimized and the second entry containing an iterable of all
            possible values for that parameter. All possible combinations
            are tried.

        minimize(str):    Expression that is evaluated for each combination of
            values from ``find_best`` to select the combination for which
            ``minimize`` evaluates to the smallest value.

    Returns:
        the smallest value of the ``minimize`` expression found.
    """

    key_order = [key for key, _ in find_best]
    best_combination = None
    best_found = None
    for combination in product(*(values for _, values in find_best)):
        lc_eval.update_substitutions(zip(key_order, combination))
        if model is not None:
            model_values = evaluate_model(model, lc_eval, expression_params)
            lc_eval.symtable["model_diff"] = (
                lc_eval(y_expression.format_map(expression_params))
                - model_values
            )

        minimize_val = lc_eval(
            minimize.format_map(expression_params), raise_errors=True
        )
        if best_found is None or minimize_val < best_found:
            best_found = minimize_val
            best_combination = combination
            best_model = None if model is None else model_values
    print(f"Best substitutions: {dict(zip(key_order, best_combination))!r}")
    print(f"Best value: {best_found!r}")
    lc_eval.lc_substitutions.update(zip(key_order, best_combination))
    return best_found, best_model


def get_plot_data(lc_fname, expressions, configuration, model=None):
    """Read relevant data from the lightcurve."""

    result = {}
    if model and model.get("shift_to") is True:
        model["shift_to"] = expressions[model["quantity"]]
    with LightCurveFile(lc_fname, "r") as lightcurve:
        lc_eval = LightCurveEvaluator(
            lightcurve, **configuration["lc_substitutions"]
        )
        lc_eval.update_substitutions({"aperture_index": 0})
        all_sphotref_fnames = set()
        for photometry_mode in configuration["photometry_modes"]:
            all_sphotref_fnames |= set(
                lightcurve.get_dataset(
                    photometry_mode + ".magfit.cfg.single_photref",
                    aperture_index=0,
                )
            )

        for single_photref_fname in all_sphotref_fnames:
            print(f"Single photref: {single_photref_fname!r}")
            best_minimize = None
            sphotref_result = {}
            for photometry_mode in configuration["photometry_modes"]:
                sphotref_dset_key = (
                    photometry_mode + ".magfit.cfg.single_photref"
                )

                lc_eval.lc_points_selection = None

                lc_points_selection = lc_eval(
                    sphotref_dset_key + " == " + repr(single_photref_fname),
                    raise_errors=True,
                )
                if configuration["selection"] is not None:
                    lc_points_selection = numpy.logical_and(
                        lc_eval(
                            configuration["selection"] or "True",
                            raise_errors=True,
                        ),
                        lc_points_selection,
                    )

                lc_eval.lc_points_selection = lc_points_selection
                if (
                    configuration["lc_substitutions"].get("magfit_iteration", 0)
                    < 0
                ):
                    lc_eval.update_substitutions(
                        {
                            "magfit_iteration": configuration[
                                "lc_substitutions"
                            ]["magfit_iteration"]
                            + lightcurve.get_num_magfit_iterations(
                                photometry_mode,
                                lc_eval.lc_points_selection,
                                **lc_eval.lc_substitutions,
                            )
                        }
                    )
                (minimize_value, sphotref_result["best_model"]) = (
                    optimize_substitutions(
                        lc_eval,
                        find_best=configuration["find_best"].items(),
                        minimize=configuration["minimize"],
                        y_expression=(
                            None
                            if model is None
                            else expressions[model["quantity"]]
                        ),
                        model=model,
                        expression_params={"mode": photometry_mode},
                    )
                )
                lc_eval.symtable["best_model"] = sphotref_result["best_model"]
                if best_minimize is None or minimize_value < best_minimize:
                    best_minimize = minimize_value
                    for var_name, var_expr in expressions.items():
                        sphotref_result[var_name] = lc_eval(
                            var_expr.format(mode=photometry_mode),
                            raise_errors=True,
                        )
                    best_substitutions = lc_eval.lc_substitutions

            result[single_photref_fname.decode()] = sphotref_result

    return result, best_substitutions


def calculate_combined(plot_data, match_id_key, aggregation_function):
    """Create a specified plot of the given lightcurve."""

    fixed_order_data = list(plot_data.values())
    combined_data = {}

    match_ids = pandas.concat(
        (pandas.Series(data[match_id_key]) for data in fixed_order_data),
        ignore_index=True,
    )

    for var_name in fixed_order_data[0].keys():
        try:
            combined_data[var_name] = (
                pandas.concat(
                    (
                        pandas.Series(data[var_name])
                        for data in fixed_order_data
                    ),
                    ignore_index=True,
                )
                .groupby(match_ids)
                .agg(aggregation_function)
                .to_numpy()
            )
        except TypeError:
            pass

    return combined_data


def transit_model(times, **params):
    """Calculate the magnitude change of exoplanet with given parameters."""

    model = RoadRunnerModel("quadratic")
    model.set_data(times)
    print(
        f"Evaluating transit model for parameters: {params!r} "
        f"for times: {times!r}."
    )
    mag_change = -2.5 * numpy.log10(model.evaluate(**params))
    return mag_change


def main():
    """Avoid polluting global scope."""

    combined_figure_id = pyplot.figure(0, dpi=300).number
    individual_figures_id = pyplot.figure(1, dpi=300).number
    transit_params = {
        "k": 0.1215,  # the planet-star radius ratio
        "ldc": [0.79272802, 0.72786169],  # limb darkening coeff
        "t0": 2455787.553228,  # the zero epoch,
        "p": 3.94150468,  # the orbital period,
        "a": 11.04,  # the orbital semi-major divided by R*,
        "i": 1.5500269086961642,  # the orbital inclination in rad,
        # e: the orbital eccentricity (optional, can be left out if assuming
        #   circular a orbit), and
        # w: the argument of periastron in radians (also optional, can be left
        #   out if assuming circular a orbit).
    }

    for detrend, fmt in [("magfit", "ob")]:  # ,
        # ('epd', 'or'),
        # ('tfa', 'ob')]:
        data_by_sphotref, _ = get_plot_data(
            "/mnt/md1/DSLR_DATA/PANOPTES/LC/GDR3_2876391245114999040.h5",
            expressions={
                "y": (
                    f"{{mode}}.{detrend}.magnitude - "
                    f"nanmedian({{mode}}.{detrend}.magnitude)"
                ),
                "x": "skypos.BJD - skypos.BJD.min()",
                "frame": "fitsheader.rawfname",
                "bin5min": "(skypos.BJD * 24 * 12).astype(int)",
            },
            configuration={
                "lc_substitutions": {},
                "selection": None,
                "find_best": {"aperture_index": range(46)},
                "minimize": (
                    f"nanmedian(abs({{mode}}.{detrend}.magnitude - "
                    f"nanmedian({{mode}}.{detrend}.magnitude)))"
                ),
                #           "nanmedian(abs(model_diff))",
                "photometry_modes": ["apphot"],
            },
            model=None,
            # {
            #    'type': 'transit',
            #    'quantity': 'y',
            #    'shift_to': True,
            #    'kwargs': {
            #        'times': 'skypos.BJD',
            #        **{k: repr(v) for k, v in transit_params.items()}
            #    }
            # }
        )

        pyplot.figure(combined_figure_id)
        for combine_by, markersize, label in [
            ("frame", 2, "raw"),
            # ("bin5min", 10, "5 min bins"),
        ]:
            data_combined = calculate_combined(
                data_by_sphotref, combine_by, numpy.nanmedian
            )

            pyplot.plot(
                data_combined["x"],
                data_combined["y"],
                fmt,
                label=detrend,
                markersize=markersize,
                markeredgecolor="black" if markersize > 5 else "none",
            )
        # pyplot.plot(
        #    data_combined["x"],
        #    data_combined["best_model"]
        #    + numpy.nanmedian(data_combined["y"] - data_combined["best_model"]),
        #    #                    transit_model(plot_data['x'],
        #    #                                  shift_to=plot_data['y'],
        #    #                                  **transit_params),
        #    "-k",
        #    linewidth=3,
        # )

        pyplot.figure(individual_figures_id)
        for subfig_id, (sphotref_fname, single_data) in enumerate(
            data_by_sphotref.items()
        ):
            print(f"Single data: {single_data!r}")
            pyplot.subplot(2, 2, subfig_id + 1)
            pyplot.plot(
                single_data["x"],
                single_data["y"],
                fmt,
                label=label,
                markersize=1,
            )
            # pyplot.plot(single_data["x"], single_data["best_model"], "-k")
            with DataReductionFile(sphotref_fname, "r") as dr_file:
                pyplot.title(dr_file.get_frame_header()["CLRCHNL"])
            pyplot.legend()
            pyplot.ylim(0.1, -0.1)

    pyplot.figure(combined_figure_id)
    pyplot.xlabel("Time [days]")
    pyplot.ylabel("Magnitude")
    pyplot.ylim(0.05, -0.05)
    pyplot.legend()
    pyplot.savefig("XO-1_combined.pdf")
    pyplot.figure(individual_figures_id)
    pyplot.savefig("XO-1_individual.pdf")


if __name__ == "__main__":
    main()
