"""Views for displaying the lightcurve of a star."""

from functools import partial
from itertools import product
from copy import deepcopy
from io import StringIO, BytesIO
import json

import matplotlib
from matplotlib import pyplot, gridspec, rcParams
import numpy
from asteval import Interpreter

from django.shortcuts import render, redirect
from django.http import JsonResponse

from autowisp.data_reduction import DataReductionFile
from autowisp.diagnostics.plot_lc import\
    get_plot_data,\
    calculate_combined,\
    transit_model


_custom_aggregators = {
    'len': len
}


def _init_session(request):
    """Initialize the session for displaying lightcurve with defaults."""

    request.session['lc_plotting'] = {
        'target_fname': '/mnt/md1/EW/LC/GDR3_1316708918505350528.h5',
        'expressions': {
            'magnitude': ('{{mode}}.{detrend}.magnitude - '
                          'nanmedian({{mode}}.{detrend}.magnitude)'),
            'bjd': 'skypos.BJD - skypos.BJD.min()',
            'rawfname': 'fitsheader.rawfname',
        },
        'configuration': {
            'lc_substitutions': {'magfit_iteration': -1},
            'selection': None,
            'find_best': [('aperture_index', list(range(41)))],
            'minimize': ('nanmedian(abs({{mode}}.{detrend}.magnitude - '
                         'nanmedian({{mode}}.{detrend}.magnitude)))'),
            'photometry_modes': ['apphot'],
        },
        'detrending_modes': [('tfa', ('og',), {})],
        'plot_layout': [
            {
                'width_ratios': [1.0],
                'height_ratios': [1.0],
                'wspace': rcParams['figure.subplot.wspace'],
                'hspace': rcParams['figure.subplot.hspace']
            },
            [0]
        ],
        'plot_config': [
            {
                'aggregate': 'nanmedian',
                'x_quantity': 'bjd',
                'y_quantity': 'magnitude',
                'match_by': 'rawfname',
                'plot_model': ['-r', {'label': 'model'}],
                'x_label': 'BJD',
                'y_label': 'magnitude',
                'title': 'GDR3 {GaiaID}'
            }
        ],
        'figure_config': {}
    }


def _jsonify_plot_data(plot_data):
    """Re-format plot data for single sphotref for storing in JSON format."""

    result = {}
    for key, value in plot_data.items():
        if key == 'best_model':
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
        if key == 'best_model':
            result[key] = value
        else:
            if isinstance(value[0], str):
                result[key] = numpy.array(
                    [entry.encode('ascii') for entry in value]
                )
            else:
                result[key] = numpy.array(value)
    return result


def _convert_plot_data_json(plot_data, reverse):
    """Re-format plot data for storing in JSON format or reverse conversion."""

    transform = (_unjsonify_plot_data if reverse else _jsonify_plot_data)
    return {fname: transform(data) for fname, data in plot_data.items()}


def _add_lightcurve_to_session(request, lightcurve_fname, select=True):
    """Add to the browser session a new entry for the given lightcurve."""

    plotting_info = request.session['lc_plotting']
    plotting_info[lightcurve_fname] = {}
    for detrend, _, _ in plotting_info['detrending_modes']:
        configuration = plotting_info['configuration']
        configuration['minimize'] = (
            configuration['minimize'].format(detrend=detrend)
        )
        plot_data, best_substitutions = get_plot_data(
            lightcurve_fname,
            {
                name: expression.format(detrend=detrend)
                for name, expression in
                plotting_info['expressions'].items()
            },
            configuration
        )

        plotting_info[lightcurve_fname][detrend] = {
            'configuration': configuration,
            'plot_data': _convert_plot_data_json(plot_data, False),
            'best_substitutions': best_substitutions
        }
    plotting_info[lightcurve_fname] = plotting_info[lightcurve_fname]
    if select:
        plotting_info['target_fname'] = lightcurve_fname


def plot(target_info, plot_config, detrending_modes_config):
    """Make a single plot of the spceified lighturve."""

    plot_data = {}
    for detrend, plot_args, plot_kwargs in detrending_modes_config:
        plot_data = _convert_plot_data_json(target_info[detrend]['plot_data'],
                                            True)
        if plot_config.get('sphotref_fname') is not None:
            plot_data = plot_data[plot_config['sphotref_fname']]
        else:
            plot_data = calculate_combined(
                plot_data,
                plot_config['match_by'],
                (
                    _custom_aggregators.get(plot_config['aggregate'])
                    or
                    getattr(numpy, plot_config['aggregate'])
                )
            )
        pyplot.plot(
            plot_data[plot_config['x_quantity']],
            plot_data[plot_config['y_quantity']],
            label=detrend,
            *plot_args,
            **plot_kwargs
        )
        if (
            plot_config.get('plot_model')
            and
            'model' in target_info[detrend]['configuration']
        ):
            pyplot.plot(
                plot_data[plot_config['x_quantity']],
                plot_data['best_model'],
                *plot_config['plot_model'][0],
                **plot_config['plot_model'][1]
            )
    pyplot.legend()
    pyplot.xlabel(plot_config['x_label'])
    pyplot.ylabel(plot_config['y_label'])
    pyplot.title(plot_config['title'])


def _create_subplots(plotting_info, splits, children, parent, figure):
    """Recursively walks the plot layout tree creating subplots as needed."""

    args = (len(splits['height_ratios']), len(splits['width_ratios']))
    if parent is None:
        grid = gridspec.GridSpec(*args, figure=figure, **splits)
    else:
        grid = gridspec.GridSpecFromSubplotSpec(*args,
                                                subplot_spec=parent,
                                                **splits)

    assert len(children) == args[0] * args[1]
    for child, subplot in zip(children, grid):
        if isinstance(child, int):
            pyplot.sca(figure.add_subplot(subplot))
            plot(plotting_info['target_info'],
                 plotting_info['plot_config'][child],
                 plotting_info['detrending_modes_config'])
        else:
            _create_subplots(
                plotting_info,
                *child,
                subplot,
                figure
            )


def _get_subplot_boundaries(splits,
                            children,
                            x_offset,
                            y_offset,
                            result):
    """Return coords of horizontal and vertical boundaries between plots."""

    x_bounds = numpy.cumsum([x_offset] + splits['width_ratios'])
    y_bounds = numpy.cumsum([y_offset] + splits['height_ratios'])
    cell_indices = product(
        range(len(splits['height_ratios'])),
        range(len(splits['width_ratios']))
    )
    for child, (y_ind, x_ind) in zip(children, cell_indices):
        if isinstance(child, int):
            result[child] = {
                'left': x_bounds[x_ind],
                'right': x_bounds[x_ind + 1],
                'top': y_bounds[y_ind],
                'bottom': y_bounds[y_ind + 1]
            }
        else:
            _get_subplot_boundaries(
                *child,
                x_bounds[x_ind],
                y_bounds[y_ind],
                result
            )


def _subdivide_figure(plot_config, new_splits, current_splits, children):
    """Sub-divide all plots with entries in new_splits accordingly."""

    for child_ind, child in enumerate(children):
        if isinstance(child, int):
            child_splits = new_splits.get(str(child))
            orig_width = current_splits[
                'width_ratios'
            ][
                child_ind % len(current_splits['width_ratios'])
            ]
            orig_height = current_splits[
                'height_ratios'
            ][
                child_ind // len(current_splits['width_ratios'])
            ]
            if child_splits is not None:
                child_splits = {
                    side: child_splits.get(side, [1.0])
                    for side in ['top', 'left']
                }
                num_subplots = (len(child_splits['top'])
                                *
                                len(child_splits['left']))
                children[child_ind] = [
                    {
                        'height_ratios': [s * orig_height
                                          for s in child_splits['left']],
                        'width_ratios': [s * orig_width
                                         for s in child_splits['top']],
                        'wspace': (current_splits['wspace']
                                   *
                                   len(child_splits['top'])),
                        'hspace': (current_splits['hspace']
                                   *
                                   len(child_splits['left']))
                    },
                    [child] + list(range(len(plot_config),
                                         len(plot_config) + num_subplots - 1))
                ]
                plot_config.extend((deepcopy(plot_config[child])
                                    for _ in range(num_subplots - 1)))
        else:
            _subdivide_figure(plot_config, new_splits, *child)


def _update_plotting_info(session, updates):
    """Modify the currently set-up figure per user input from BUI."""

    modified_session = False
    if 'applySplits' in updates:
        _subdivide_figure(session['lc_plotting']['plot_config'],
                          updates['applySplits'],
                          *session['lc_plotting']['plot_layout'])
        modified_session = True
    if 'rcParams' in updates:
        for param, value in updates['rcParams'].items():
            rcParams[param] = value.strip('[]')
    return modified_session


def update_lightcurve_figure(request):
    """Generate and return a new figure for the current lightcurve."""

    request.session.modified = _update_plotting_info(
        request.session,
        json.loads(request.body.decode())
    )

    matplotlib.use('svg')
    pyplot.style.use('dark_background')

    figure = pyplot.figure(
        **request.session['lc_plotting']['figure_config']
    )
    plotting_info = request.session['lc_plotting']
    _create_subplots(
        {
            'target_info': plotting_info[plotting_info['target_fname']],
            'plot_config': plotting_info['plot_config'],
            'detrending_modes_config': plotting_info['detrending_modes']
        },
        *request.session['lc_plotting']['plot_layout'],
        None,
        figure
    )

    with StringIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches='tight', format='svg')
        subplot_boundaries = {}
        _get_subplot_boundaries(*request.session['lc_plotting']['plot_layout'],
                                0,
                                0,
                                subplot_boundaries)
        return JsonResponse({
            'plot_data': image_stream.getvalue(),
            'boundaries': subplot_boundaries
        })


def edit_subplot(request, plot_id):
    """Set the view to allow editing the selected plot."""

    plotting_info = request.session['lc_plotting']
    sub_plot_config = dict(plotting_info['plot_config'][plot_id])
    sub_plot_config['plot_id'] = plot_id
    sub_plot_config['expressions'] = list(
        plotting_info['expressions'].items()
    )
    sub_plot_config['model'] = plotting_info.get(
        'model',
        {
            'k': 0.1, #the planet-star radius ratio
            'ldc': [0.8, 0.7], #limb darkening coeff
            't0': 2455787.553228,# the zero epoch,
            'p': 3.0,# the orbital period,
            'a': 10.0,# the orbital semi-major divided by R*,
            'i': numpy.pi / 2,# the orbital inclination in rad,
            'e': None, # the orbital eccentricity (optional, can be left out if
                     # assuming circular a orbit), and
            'w': None  # the argument of periastron in radians (also optional,
                       # can be left out if assuming circular a orbit).
        }
    )
    sub_plot_config['model']['enabled'] = 'model' in plotting_info
    return render(
        request,
        'results/subplot_config.html',
        {'config': sub_plot_config}
    )

def _sanitize_rcparams():
    """Return list of matplotlib rcParams that can be set through BUI."""

    result = []
    for param, value in rcParams.items():
        if param == 'lines.dash_capstyle':
            value = 'butt'
        elif param == 'lines.solid_capstyle':
            value = 'projecting'
        elif param.endswith('_joinstyle'):
            value = 'round'
        if (
            not param.endswith('prop_cycle')
            and
            param not in ['savefig.bbox', 'backend']
            and
            not param.startswith('animation')
            and
            not param.startswith('keymap')
            and
            not param.startswith('figure.subplot')
            and
            param[0] != '_'
        ):
            result.append((param, value))
    return result


def edit_rcparams(request):
    """Set the view to allow editing rcParams."""

    return render(
        request,
        'results/rcParams_config.html',
        {'config': _sanitize_rcparams()}
    )

def display_lightcurve(request):
    """Display plots of a single lightcurve to the user."""

    if 'lc_plotting' not in request.session:
        rcParams['figure.subplot.bottom'] = 0.0
        rcParams['figure.subplot.top'] = 1.0
        rcParams['figure.subplot.left'] = 0.0
        rcParams['figure.subplot.right'] = 1.0
        _init_session(request)
        _add_lightcurve_to_session(
            request,
            request.session['lc_plotting']['target_fname']
        )

    return render(
        request,
        'results/display_lightcurves.html',
        {'config': None}
    )


def clear_lightcurve_buffer(request):
    """Remove buffered lightcurve data from the session."""

    if 'lc_plotting' in request.session:
        del request.session['lc_plotting']
    return redirect('/results')


def download_lightcurve_figure(request):
    """Creates and send to the user the currently setup figure as a file."""
