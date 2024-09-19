"""Views for displaying the lightcurve of a star."""

from functools import partial
from copy import deepcopy

from matplotlib import pyplot, gridspec
import numpy

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
            'lc_substitutions': {'magfit_iteration': '-1'},
            'selection': 'None',
            'find_best': [('aperture_index', [str(i) for i in range(41)])],
            'minimize': ('nanmedian(abs({mode}}.{detrend}.magnitude - '
                         'nanmedian({{mode}}.{detrend}.magnitude)))'),
            'photometry_modes': ['apphot'],
        },
        'detrending_modes': [('tfa', ('og',), {})],
        'plot_layout': [['1.0'], ['1.0'], ['0']],
        'plot_config': [
            {
                'aggregate': 'nanmedian',
                'x_quantity': 'magnitude',
                'match_by': 'rawfname',
                'plot_y': 'bjd',
                'plot_model': ['-r', {'label': 'model'}],
                'x_label': 'BJD',
                'y_label': 'magnitude',
                'title': 'GDR3 {GaiaID}'
            }
        ],
        'figure_config': {}
    }


def _add_lightcurve_to_session(request, lightcurve_fname, select=True):
    """Add to the browser session a new entry for the given lightcurve."""

    plotting_info = request.session['lc_plotting']
    plotting_info[lightcurve_fname] = {}
    for detrend in plotting_info['detrending_modes']:
        configuration = deepcopy(plotting_info['configuration'])
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
            'plot_data': plot_data,
            'best_substitutions': best_substitutions
        }
    if select:
        plotting_info['target_name'] = lightcurve_fname


def plot(request, plot_index):
    """Make a single plot of the spceified lighturve."""

    plotting_info = request.session['lc_plotting']
    target_info = plotting_info[plotting_info['target_fname']]
    plot_config = plotting_info['plot_config'][plot_index]

    plot_data = {}
    for detrend, plot_args, plot_kwargs in plotting_info['detrending_modes']:
        if plot_config.get('sphotref_fname') is not None:
            plot_data = target_info[
                detrend
            ][
                plot_config['sphotref_fname']
            ]
        else:
            plot_data = calculate_combined(
                target_info[detrend],
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
            'model' in target_info['configuration']
        ):
            pyplot.plot(
                plot_data[plot_config['x_quantity']],
                plot_data['best_model'],
                *plot_config['plot_model'][0],
                **plot_config['plot_model'][1]
            )
    pyplot.xlabel(plot_config['x_label'])
    pyplot.ylabel(plot_config['y_label'])
    pyplot.title(plot_config['title'])


def create_subplots(request, widths, heights, children, parent):
    """Recursively walks the plot layout tree creating subplots as needed."""

    args = (len(heights), len(widths))
    kwargs= {'width_ratios': [float(w) for w in widths],
             'height_ratios': [float(h) for h in heights]}
    if isinstance(parent, gridspec.GridSpecBase):
        grid = gridspec.GridSpecFromSubplotSpec(*args,
                                                subplot_spec=parent,
                                                **kwargs)
    else:
        grid = gridspec.GridSpec(*args, figure=parent, **kwargs)

    assert len(children) == args[0] * args[1]
    for child, subplot in zip(children, grid):
        if isinstance(child, str):
            pyplot.sca(subplot)
            plot(request, int(child))
        else:
            create_subplots(request, *child, subplot)


def display_lightcurve(request):
    """Display plots of a single lightcurve to the user."""

    figure = pyplot.figure(**request.session['lc_plotting']['figure_config'])
    create_subplots(request,
                    *request.session['lc_plotting']['plot_layout'],
                    figure)


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
        #e: the orbital eccentricity (optional, can be left out if assuming
        #   circular a orbit), and
        #w: the argument of periastron in radians (also optional, can be left
        #   out if assuming circular a orbit).
    }

    for detrend, fmt in [('magfit', 'ob'), ('epd', 'or'), ('tfa', 'og')]:
        data_by_sphotref = get_plot_data(
            '/mnt/md1/EW/LC/GDR3_1316708918505350528.h5',
            {
                'y': (f'{{mode}}.{detrend}.magnitude - '
                      f'nanmedian({{mode}}.{detrend}.magnitude)'),
                'x': 'skypos.BJD - skypos.BJD.min()',
                'match_ids': '(skypos.BJD * 24 * 12).astype(int)',
                #'fitsheader.rawfname',
            },
            {
                'lc_substitutions': {'magfit_iteration': -1},
                'selection': None,
                'find_best': [('aperture_index', range(41))],
                'minimize': 'nanmedian(abs(model_diff))',
#                (f'nanmedian(abs({{mode}}.{detrend}.magnitude - '
#                             f'nanmedian({{mode}}.{detrend}.magnitude)))'),
                'photometry_modes': ['apphot'],
                'model': {
                    'quantity': 'magnitude',
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
        data_combined = calculate_combined(data_by_sphotref,
                                           'match_ids',
                                           numpy.nanmedian)

        pyplot.figure(combined_figure_id)
        pyplot.plot(data_combined['x'],
                    data_combined['y'],
                    fmt,
                    label=detrend,
                    markersize=2)
        pyplot.plot(data_combined['x'],
                    data_combined['best_model'],
#                    transit_model(plot_data['x'],
#                                  shift_to=plot_data['y'],
#                                  **transit_params),
                    '-k')

        pyplot.figure(individual_figures_id)
        for subfig_id, (sphotref_fname, single_data) in enumerate(
                data_by_sphotref.items()
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
