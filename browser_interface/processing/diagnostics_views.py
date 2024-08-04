"""Views for displaying diagnostics for the calibration steps."""

from io import StringIO, BytesIO
import json

import numpy
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from autowisp import Evaluator
from autowisp.database.interface import Session
from autowisp.diagnostics.detrending import get_magfit_performance_data
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    MasterType,\
    MasterFile,\
    Condition,\
    ConditionExpression
#pylint: enable=no-name-in-module


def _guess_labels(mphotref_entries):
    """Guess what would make good labels for plotting."""

    num_expr = len(mphotref_entries[0][1])
    print(
        'Expression sets: '
        +
        repr([
            set(entry[1][i] for entry in mphotref_entries)
            for i in range(num_expr)
        ])
    )
    use_expr = [
        len(set(entry[1][i] for entry in mphotref_entries)) > 1
        for i in range(num_expr)
    ]
    print(f'Use expr flags: {use_expr!r}')
    for entry in mphotref_entries:
        entry[5] = ':'.join(
            expr for expr, use in zip(entry[1], use_expr) if use
        )


def _init_magfit_session(request):
    """Add to browser session which magfit runs can be diagnosed."""

    #False positive
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        master_photref_fnames = db_session.execute(
            select(
                MasterFile.id,
                MasterFile.filename
            ).join(
                MasterType
            ).where(
                MasterType.name == 'master_photref'
            ).order_by(
                MasterFile.progress_id
            )
        ).all()
        match_expressions = db_session.scalars(
            select(
                ConditionExpression.expression
            ).join_from(
                MasterType,
                Condition,
                MasterType.condition_id == Condition.id
            ).join(
                ConditionExpression
            ).where(
                MasterType.name == 'master_photref'
            )
        ).all()
    color_map = matplotlib.colormaps.get_cmap('tab10')
    request.session['diagnostics'] = {
        'magfit': {
            'match_expressions': match_expressions,
            'mphotref': [
                [
                    str(mphotref_id),
                    tuple(
                        str(Evaluator(mphotref_fname)(expr))
                        for expr in match_expressions
                    ),
                    (
                        '#'
                        +
                        ''.join([
                            f'{int(numpy.round(c * 255)):02x}'
                            for c in color_map(mphotref_index % color_map.N)[:3]
                        ])
                    ),
                    '',
                    '0.8',
                    ''
                ]
                for mphotref_index, (mphotref_id, mphotref_fname) in enumerate(
                    master_photref_fnames
                )
            ],
            'plot_config': {
                'x_range': ['', ''],
                'y_range': ['', ''],
                'mag_expression': 'phot_g_mean_mag'
            }
        }
    }
    _guess_labels(request.session['diagnostics']['magfit']['mphotref'])


def refresh_diagnostics(request):
    """Reset all diagnostics related entries in the BUI session """

    if 'diagnostics' in request.session:
        del request.session['diagnostics']
    return redirect('/processing/display_magfit_diagnostics')


def display_magfit_diagnostics(request):
    """View displaying the scatter after magnitude fitting."""


    print('Using session with keys: ' + repr(request.session.keys()))

    if (
        'diagnostics' not in request.session
        or
        'magfit' not in request.session['diagnostics']
    ):
        print('Refreshing session')
        _init_magfit_session(request)


    print('Using session: ' + repr(request.session['diagnostics']['magfit']))

    return render(
        request,
        'processing/detrending_diagnostics.html',
        request.session['diagnostics']['magfit'],
    )


def display_diagnostics(request, step, imtype):
    """Common interface to all diagnostic views."""

    if step == 'fit_magnitudes':
        return display_magfit_diagnostics(request)
    return redirect('processing/progress')


def create_plot(session_magfit):
    """Create the diagnostic plot per configuration in session."""

    pyplot.clf()
    pyplot.cla()

    plot_config = session_magfit['plot_config']
    for mphotref_info in session_magfit['mphotref']:
        if not mphotref_info[3]:
            continue

        data = get_magfit_performance_data(int(mphotref_info[0]),
                                           float(mphotref_info[4]),
                                           plot_config['mag_expression'],
                                           True)
        pyplot.semilogy(
            data['magnitudes'],
            data['best_scatter'],
            linestyle='none',
            marker=mphotref_info[3],
            markeredgecolor=(
                mphotref_info[2] if mphotref_info[3] in 'x+.,1234|_'
                else 'none'
            ),
            markerfacecolor=mphotref_info[2],
            label=mphotref_info[5]
        )

    try:
        pyplot.xlim(*(float(v) for v in plot_config['x_range']))
    except ValueError:
        pass

    try:
        pyplot.ylim(*(float(v) for v in plot_config['y_range']))
    except ValueError:
        pass

    pyplot.xlabel(plot_config['mag_expression'])
    pyplot.ylabel('MAD')
    pyplot.grid(True, which='both', linewidth=0.2)
    pyplot.legend()


def update_diagnostics_plot(request):
    """Generate and respond with update plot, also update session."""

    plot_config = json.loads(request.body.decode())
    print('Plot config: ' + repr(plot_config))

    matplotlib.use('svg')
    pyplot.style.use('dark_background')

    session_magfit = request.session['diagnostics']['magfit']
    session_magfit['plot_config']['x_range'] = plot_config['x_range']
    session_magfit['plot_config']['y_range'] = plot_config['y_range']
    session_magfit['plot_config']['mag_expression'] = (
        plot_config['mag_expression']
    )

    to_update = session_magfit['mphotref']
    for mphotref_info in to_update:
        this_config = plot_config['datasets'].get(str(mphotref_info[0]))
        if this_config is None:
            continue
        mphotref_info[2] = this_config['color']
        mphotref_info[3] = this_config['marker']
        mphotref_info[4] = this_config['min_fraction']
        mphotref_info[5] = this_config['label']


    print('Updated session: ' + repr(request.session['diagnostics']['magfit']))

    request.session.modified = True

    create_plot(session_magfit)

    with StringIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches='tight', format='svg')
        return JsonResponse({
            'plot_data': image_stream.getvalue(),
            'plot_config': (
                request.session['diagnostics']['magfit']['plot_config']
            )
        })


def download_diagnostics_plot(request):
    """Send the user the diagnostics plot as a PDF file."""

    matplotlib.use('pdf')
    pyplot.style.use('default')

    create_plot(request.session['diagnostics']['magfit'])

    with BytesIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches='tight', format='pdf')
        return HttpResponse(
            image_stream.getvalue(),
            headers={
                'Content-Type': 'application/pdf',
                'Content-Disposition': (
                    'attachment; filename="detrending_performance.pdf"'
                )
            }
        )
