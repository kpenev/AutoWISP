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

#TODO:Document all expected entries in configuration for `get_plot_data()`


def evaluate_model(model, lc_eval, expression_params):
    """Return the model evaluated for the given lightcurve."""

    args = [lc_eval(expression.format_map(expression_params))
            for expression in model.get('args', [])]
    kwargs = {
        arg_name: lc_eval(expression.format_map(expression_params))
        for arg_name, expression in model.get('kwargs', {})
    }
    return model['evaluate'](*args, **kwargs)


def optimize_substitutions(lc_eval,
                           *,
                           find_best,
                           minimize,
                           y_expression,
                           model,
                           expression_params):
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
        model_values = evaluate_model(model, lc_eval, expression_params)
        lc_eval.symtable['model_diff'] = (
            lc_eval(y_expression.format_map(expression_params))
            -
            model_values
        )

        minimize_val = lc_eval(minimize, raise_errors=True)
        if best_found is None or minimize_val < best_found:
            best_found = minimize_val
            best_combination = combination
            best_model = model_values
    print(f'Best substitutions: {dict(zip(key_order, best_combination))!r}')
    print(f'Best value: {best_found!r}')
    lc_eval.lc_substitutions.update(zip(key_order, best_combination))
    return best_found, best_model


def get_plot_data(lc_fname, configuration):
    """Create a specified plot of the given lightcurve."""

    plot_data = {coord: pandas.Series() for coord in 'xy'}
    plot_data['by_sphotref'] = {}
    match_ids = pandas.Series()
    with LightCurveFile(lc_fname, 'r') as lightcurve:
        lc_eval = LightCurveEvaluator(lightcurve,
                                      configuration['lc_substitutions'])
        lc_eval.update_substitutions({'aperture_index': 0})
        all_sphotref_fnames = set()
        for photometry_mode in configuration['photometry_modes']:
            all_sphotref_fnames |= set(
                lightcurve.get_dataset(
                    photometry_mode + '.magfit.cfg.single_photref',
                    aperture_index=0
                )
            )

        for single_photref_fname in all_sphotref_fnames:
            print(f'Single photref: {single_photref_fname!r}')
            best_minimize = None
            plot_data['by_sphotref'][single_photref_fname] = {}
            sphotref_data = plot_data['by_sphotref'][single_photref_fname]
            for photometry_mode in configuration['photometry_modes']:
                sphotref_dset_key = (photometry_mode
                                     +
                                     '.magfit.cfg.single_photref')

                lc_eval.lc_points_selection = None
                lc_eval.lc_points_selection = lc_eval(
                    sphotref_dset_key
                    +
                    ' == '
                    +
                    repr(single_photref_fname),
                    raise_errors=True
                )
                if configuration['selection'] is not None:
                    lc_eval.lc_points_selection = numpy.logical_and(
                        lc_eval(configuration['selection'] or 'True',
                                raise_errors=True),
                        lc_eval.lc_points_selection
                    )
                if configuration[
                        'lc_substitutions'
                ].get(
                    'magfit_iteration', 0
                ) < 0:
                    lc_eval.update_substitutions({
                        'magfit_iteration': configuration[
                            'lc_substitutions'
                        ][
                            'magfit_iteration'
                        ] + lightcurve.get_num_magfit_iterations(
                            photometry_mode,
                            lc_eval.lc_points_selection,
                            **lc_eval.lc_substitutions
                        )
                    })
                (
                    minimize_value,
                    sphotref_data['best_model']
                ) = optimize_substitutions(
                    lc_eval,
                    find_best=configuration['find_best'],
                    minimize=configuration['minimize'],
                    y_expression=configuration['y_expression'],
                    model=configuration['model'],
                    expression_params={'mode': photometry_mode}
                )
                if best_minimize is None or minimize_value < best_minimize:
                    best_minimize = minimize_value
                    for coord in 'xy':
                        sphotref_data[coord] = lc_eval(
                            configuration[
                                coord + '_expression'
                            ].format(
                                mode=photometry_mode
                            ),
                            raise_errors=True
                        )
                        new_match_ids = lc_eval(
                            configuration[
                                'match_by'
                            ].format(
                                mode=photometry_mode
                            ),
                            raise_errors=True
                        )
            for coord in 'xy':
                plot_data[coord] = pandas.concat(
                    [
                        plot_data[coord],
                        pandas.Series(sphotref_data[coord])
                    ],
                    ignore_index=True
                )
            match_ids = pandas.concat(
                [
                    match_ids,
                    pandas.Series(new_match_ids)
                ],
                ignore_index=True
            )

    print('X: ' + repr(plot_data['x']))
    print('Y: ' + repr(plot_data['y']))
    print('Match by: ' + repr(match_ids))


    for coord in 'xy':
        plot_data[coord] = pandas.Series(
            plot_data[coord]
        ).groupby(
            match_ids
        ).agg(
            getattr(numpy, configuration['aggregation_func'])
        ).to_numpy()

    return plot_data


def transit_model(bjd, shift_to=None, **params):
    """Calculate the magnitude change of exoplanet with given parameters."""

    model = RoadRunnerModel('quadratic')
    model.set_data(bjd)
    print(
        f'Evaluating transit model for parameters: {params!r} for BJD: {bjd!r}'
    )
    mag_change = -2.5 * numpy.log10(model.evaluate(**params))
    if shift_to is not None:
        assert len(shift_to) == len(bjd)
        mag_change += numpy.nanmedian(shift_to - mag_change)
    return mag_change


def main():
    """Avoid polluting global scope."""

    combined_figure_id = pyplot.figure(0, dpi=300).number
    individual_figures_id = pyplot.figure(1, dpi=300).number
    transit_params={
        'k': 0.1326, #the planet-star radius ratio
        'ldc': [0.79272802, 0.72786169], #limb darkening coeff
        't0': 2455787.553228,# the zero epoch,
        'p': 3.94150468,# the orbital period,
        'a': 11.24,# the orbital semi-major divided by R*,
        'i': 1.5500269086961642,# the orbital inclination in rad,
        #e: the orbital eccentricity (optional, can be left out if assuming circular a orbit), and
        #w: the argument of periastron in radians (also optional, can be left out if assuming circular a orbit).
    }

    for detrend, fmt in [('magfit', 'ob'), ('epd', 'or'), ('tfa', 'og')]:
        plot_data = get_plot_data(
            '/mnt/md1/EW/LC/GDR3_1316708918505350528.h5',
            {
                'lc_substitutions': {'magfit_iteration': -1},
                'selection': None,
                'find_best': [('aperture_index', range(41))],
                'minimize': 'nanmedian(abs(model_diff))',
#                (f'nanmedian(abs({{mode}}.{detrend}.magnitude - '
#                             f'nanmedian({{mode}}.{detrend}.magnitude)))'),
                'photometry_modes': ['apphot'],
                'y_expression': (f'{{mode}}.{detrend}.magnitude - '
                                 f'nanmedian({{mode}}.{detrend}.magnitude)'),
                'x_expression': 'skypos.BJD',
                'match_by': 'fitsheader.rawfname',
                'aggregation_func': 'nanmedian',
                'model': {
                    'evaluate': partial(
                        transit_model,
                        **transit_params
                    ),
                    'args': [
                        'skypos.BJD',
                        (
                            f'{{mode}}.{detrend}.magnitude - '
                            f'nanmedian({{mode}}.{detrend}.magnitude)'
                        )
                    ]
                }
            }
        )

        pyplot.figure(combined_figure_id)
        pyplot.plot(plot_data['x'],
                    plot_data['y'],
                    fmt,
                    label=detrend,
                    markersize=2)
        pyplot.plot(plot_data['x'],
                    transit_model(plot_data['x'],
                                  shift_to=plot_data['y'],
                                  **transit_params),
                    '-k')

        pyplot.figure(individual_figures_id)
        for subfig_id, (sphotref_fname, single_data) in enumerate(
                plot_data['by_sphotref'].items()
        ):
            print(f'Single data: {single_data!r}')
            pyplot.subplot(2, 2, subfig_id + 1)
            pyplot.plot(single_data['x'],
                        single_data['y'],
                        fmt,
                        label=detrend,
                        markersize=1)
            pyplot.plot(single_data['x'],
                        single_data['best_model'],
                        '-k')
            with DataReductionFile(sphotref_fname, 'r') as dr_file:
                pyplot.title(dr_file.get_frame_header()['CLRCHNL'])
            pyplot.legend()
            pyplot.ylim(0.1, -0.1)


    pyplot.figure(combined_figure_id)
    pyplot.ylim(0.1, -0.1)
    pyplot.legend()
    pyplot.savefig('XO-1_combined.pdf')
    pyplot.figure(individual_figures_id)
    pyplot.savefig('XO-1_individual.pdf')


if __name__ == '__main__':
    main()
