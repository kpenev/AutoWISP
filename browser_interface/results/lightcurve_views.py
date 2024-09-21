"""Views for displaying the lightcurve of a star."""

from functools import partial
from itertools import product
from copy import deepcopy
from io import StringIO, BytesIO
import json

import matplotlib
from matplotlib import pyplot, gridspec
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
        'plot_layout': [([1.0], [1.0]), [0]],
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


def create_subplots(plotting_info, splits, children, parent, figure):
    """Recursively walks the plot layout tree creating subplots as needed."""

    args = tuple(len(s) for s in splits)
    kwargs= {'width_ratios': splits[0], 'height_ratios': splits[1]}
    if parent is None:
        grid = gridspec.GridSpec(*args, figure=figure, **kwargs)
    else:
        grid = gridspec.GridSpecFromSubplotSpec(*args,
                                                subplot_spec=parent,
                                                **kwargs)

    assert len(children) == args[0] * args[1]
    for child, subplot in zip(children, grid):
        if isinstance(child, int):
            pyplot.sca(figure.add_subplot(subplot))
            plot(plotting_info['target_info'],
                 plotting_info['plot_config'][child],
                 plotting_info['detrending_modes_config'])
        else:

            create_subplots(
                plotting_info,
                *child,
                subplot
            )


def get_subplot_boundaries(splits,
                           children,
                           x_offset,
                           y_offset,
                           result):
    """Return coords of horizontal and vertical boundaries between plots."""

    x_bounds = numpy.cumsum([x_offset] + splits[0])
    y_bounds = numpy.cumsum([y_offset] + splits[1])
    cell_indices = product(range(len(splits[0])), range(len(splits[1])))
    for child, (x_ind, y_ind) in zip(children, cell_indices):
        if isinstance(child, int):
            result[child] = {
                'left': x_bounds[x_ind],
                'right': x_bounds[x_ind + 1],
                'top': y_bounds[y_ind],
                'bottom': y_bounds[y_ind + 1]
            }
        else:
            get_subplot_boundaries(
                *child,
                x_bounds[x_ind],
                y_bounds[y_ind],
                result
            )


def update_lightcurve_figure(request):
    """Generate and return a new figure for the current lightcurve."""

    matplotlib.use('svg')
    #pyplot.style.use('dark_background')

    figure = pyplot.figure(
        **request.session['lc_plotting']['figure_config']
    )
    plotting_info = request.session['lc_plotting']
    create_subplots(
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
        get_subplot_boundaries(*request.session['lc_plotting']['plot_layout'],
                               0,
                               0,
                               subplot_boundaries)
        return JsonResponse({
            'plot_data': image_stream.getvalue(),
            'boundaries': subplot_boundaries
        })


def display_lightcurve(request):
    """Display plots of a single lightcurve to the user."""

    if 'lc_plotting' not in request.session or True:
        _init_session(request)
        _add_lightcurve_to_session(
            request,
            request.session['lc_plotting']['target_fname']
        )

    return render(
        request,
        'results/display_lightcurves.html',
        {}
    )


def clear_lightcurve_buffer(request):
    """Remove buffered lightcurve data from the session."""

    if 'lc_plotting' in request.session:
        del request.session['lc_plotting']
    return redirect('/results')


def download_lightcurve_figure(request):
    """Creates and send to the user the currently setup figure as a file."""
